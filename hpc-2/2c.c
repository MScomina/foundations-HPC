#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>

#define MPI_MASTER_TO_WORKER 1
#define MPI_WORKER_TO_MASTER_TILE 2
#define MPI_WORKER_TO_MASTER_DATA 3
#define MPI_WORK_TERM_TAG 4

/*
 *  Forward declaration of the PGM helper functions defined in
 *  read_write_pgm_image.c. The actual definitions are compiled
 *  alongside this file, so we only need the prototype here.
 */
void write_pgm_image(void *image, int maxval, int xsize, int ysize, const char *image_name);
void swap_image(void *image, int xsize, int ysize, int maxval);

static MPI_Datatype pixel_type;

typedef struct {
    int x;
    int y;
    unsigned short data;
} pixel;

typedef struct {
    unsigned int start_row;
    unsigned int end_row;
} tile_t;


/**
 * Helper function for parsing arguments to long double.
 */
static bool parse_ld(const char* s, long double* out) {
    char* endptr;
    errno = 0;
    long double val = strtold(s, &endptr);

    if (errno != 0)               return false;
    if (endptr == s)              return false;
    if (*endptr != '\0')          return false;

    *out = val;
    return true;
}

static tile_t* make_tile_list(int n_y, unsigned int n_tiles) {
    tile_t* tiles = (tile_t*)calloc(n_tiles, sizeof(tile_t));
    if (!tiles) {
        fprintf(stderr, "Failed to allocate tile list.\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
    }
    int base   = n_y / n_tiles;
    int extra  = n_y % n_tiles;
    int r      = 0;

    for (int t = 0; t < n_tiles; ++t) {
        int rows = base + (t < extra ? 1 : 0);
        tiles[t].start_row = r;
        tiles[t].end_row   = r + rows;
        r += rows;
    }
    return tiles;
}

/**
 * Function responsible for the starting initialization of the pixels array.
 */
static void init_pixels(pixel* restrict pixels, const unsigned int n_x, const unsigned int n_y) {
    #pragma omp parallel for \
        collapse(2) \
        schedule(static)
    for (unsigned int row = 0; row < n_y; row++) {
        for (unsigned int column = 0; column < n_x; column++) {
            pixel* current_pixel = &pixels[row * n_x + column];
            current_pixel->x = column;
            current_pixel->y = row;
        }
    }
}

/**
 * Function responsible for computing the Mandelbrot set values.
 */
bool compute_mandelbrot(
    pixel* restrict pixels, 
    const complex double btm_left,
    const complex double top_right,
    const int n_x,
    const int n_y,
    const unsigned int i_max,
    tile_t tile
    ) {
    const double dx = (creal(top_right) - creal(btm_left)) / (n_x - 1);
    const double dy = (cimag(top_right) - cimag(btm_left)) / (n_y - 1);
    #ifdef _OPENMP
    #pragma omp parallel for \
        collapse(2) \
        schedule(dynamic, 64) \
        default(none) \
        shared(pixels, btm_left, top_right, dy, dx, i_max, n_x, n_y, tile) \
        proc_bind(spread)
    #endif
    for (int line = tile.start_row; line<tile.end_row; line++) {
        const double c_im = cimag(btm_left) + line * dy;
        for(int column = 0; column<n_x; column++) {
            int idx = line * n_x + column;
            const double c_re = creal(btm_left) + column * dx;
            pixel* current_pixel = &pixels[idx];
            double z_re = 0.0;
            double z_im = 0.0;
            unsigned int iter = 0;
            for (iter = 0; iter < i_max; ++iter) {
                double new_re = z_re*z_re - z_im*z_im + c_re;
                double new_im = 2.0*z_re*z_im + c_im;
                z_re = new_re;
                z_im = new_im;

                // The check condition has been changed from |z| > 2 to |z|^2 > 4 
                // in order to avoid a meaningless sqrt operation.
                if (z_re*z_re + z_im*z_im > 4.0) break;
            }
            current_pixel->data = (iter == i_max) ? 0 : (unsigned short)iter;
        }
    }
    return true;
}

/**
 * Write the Mandelbrot image to a PGM file.
 */
static void save_image(pixel* pixels,
                const int n_x,
                const int n_y,
                const unsigned int i_max,
                const char* filename)
{
    const size_t total = (size_t)n_x * (size_t)n_y;
    if (i_max <= 255) {
        unsigned char* img8 = (unsigned char*)calloc(total, sizeof(unsigned char));
        if (!img8) {
            fprintf(stderr, "Failed to allocate memory for 8‑bit PGM image.\n");
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
        }
        for (size_t idx = 0; idx < total; ++idx) {
            img8[idx] = (unsigned char)pixels[idx].data;
        }
        write_pgm_image(img8, (int)i_max, n_x, n_y, filename);
        free(img8);
    } else {
        unsigned short* img = (unsigned short*)calloc(total, sizeof(unsigned short));
        if (!img) {
            fprintf(stderr, "Failed to allocate memory for 16‑bit PGM image.\n");
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
        }
        for (size_t idx = 0; idx < total; ++idx) {
            unsigned int scaled = (unsigned int)pixels[idx].data * 65535U / i_max;
            img[idx] = (unsigned short)scaled;
        }
        if (i_max > 255) {
            swap_image(img, n_x, n_y, 65535);
        }
        write_pgm_image(img, 65535, n_x, n_y, filename);
        free(img);
    }
}

/**
 * Master workload function. It is only ever executed by rank 0 of the MPI stack.
 */
static pixel* master_workload(
    const complex double btm_left,
    const complex double top_right,
    const int n_x, 
    const int n_y,
    const unsigned int i_max
    ) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    unsigned int number_pixels = n_x * n_y;
    pixel *pixels = (pixel*)calloc(number_pixels, sizeof(pixel));
    if (!pixels) {
        fprintf(stderr, "Failed to allocate memory for pixels in master workload.\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
    }
    init_pixels(pixels, n_x, n_y);

    if (world_size == 1) {
        compute_mandelbrot(pixels, btm_left, top_right, n_x, n_y, i_max, (tile_t){0, n_y});
        return pixels;
    }

    unsigned int workers = world_size - 1;
    unsigned int n_tiles = 4 * workers;
    unsigned int tile_index = 0;
    tile_t *tiles = make_tile_list(n_y, n_tiles);

    // Initially send one tile to each worker up to number of tiles.
    // WARNING: This always assumes n_tiles > workers!
    for (int w = 0; w < workers; w++, tile_index++) {
        MPI_Send(&tiles[tile_index], 2, MPI_UNSIGNED, w + 1, MPI_MASTER_TO_WORKER, MPI_COMM_WORLD);
    }

    MPI_Status status;
    tile_t tile;
    // The while contains the + workers in order to stop them after all the tiles are completed.
    while (tile_index < n_tiles + workers) {

        MPI_Recv(&tile, 2, MPI_UNSIGNED, MPI_ANY_SOURCE, MPI_WORKER_TO_MASTER_TILE, MPI_COMM_WORLD, &status);
        unsigned int n_rows = tile.end_row - tile.start_row;

        // Receive pixel data for this tile
        pixel *buf = (pixel*)malloc(n_rows * n_x * sizeof(pixel));
        MPI_Recv(buf, n_rows * n_x, pixel_type, MPI_ANY_SOURCE, MPI_WORKER_TO_MASTER_DATA, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE;
        memcpy(&pixels[tile.start_row * n_x], buf, n_rows * n_x * sizeof(pixel));
        free(buf);

        if (tile_index < n_tiles) {
            // Send next tile if available.
            MPI_Send(&tiles[tile_index], 2, MPI_UNSIGNED, src, MPI_MASTER_TO_WORKER, MPI_COMM_WORLD);
        } else {
            // No more tiles available. Stop worker.
            MPI_Send(NULL, 0, MPI_UNSIGNED, src, MPI_WORK_TERM_TAG, MPI_COMM_WORLD);
        }
        tile_index++;
    }
    free(tiles);
    return pixels;
}

static void worker_workload(
    const complex double btm_left,
    const complex double top_right,
    const int n_x, 
    const int n_y,
    const unsigned int i_max
) {
    MPI_Status status;
    tile_t tile;
    // Do-while loop because it has to enter at least once.
    do {
        MPI_Recv(&tile, 2, MPI_UNSIGNED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == MPI_WORK_TERM_TAG) {
            break;
        }
        int n_rows = tile.end_row - tile.start_row;
        pixel *pixels = (pixel*)calloc(n_x * n_y, sizeof(pixel));
        if (!pixels) {
            fprintf(stderr, "Failed to allocate pixel buffer in worker.\n");
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
        }
        init_pixels(pixels, n_x, n_rows);
        compute_mandelbrot(pixels, btm_left, top_right, n_x, n_y, i_max, tile);
        MPI_Send(&tile, 2, MPI_UNSIGNED, 0, MPI_WORKER_TO_MASTER_TILE, MPI_COMM_WORLD);
        MPI_Send(&pixels[tile.start_row * n_x], n_rows * n_x, pixel_type, 0, MPI_WORKER_TO_MASTER_DATA, MPI_COMM_WORLD);
        free(pixels);
    } while (status.MPI_TAG != MPI_WORK_TERM_TAG);
}

int main(int argc, char* argv[]) {
    int mpi_provided_thread_level; 
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided_thread_level);
    if (mpi_provided_thread_level < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "A problem arose when asking for MPI_THREAD_FUNNELED level.\n"); 
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    MPI_Type_contiguous(sizeof(pixel), MPI_BYTE, &pixel_type);
    MPI_Type_commit(&pixel_type);
    if (argc < 8) {
        fprintf(stderr, "Please enter all of the required arguments (specifically, in order: n_x, n_y, x_L, y_L, x_R, y_R, I_max).");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    const long n_x_long = strtol(argv[1], NULL, 10);
    const long n_y_long = strtol(argv[2], NULL, 10);
    if (n_x_long > UINT_MAX || n_y_long > UINT_MAX || n_x_long < 0 || n_y_long < 0) {
        fprintf(stderr, "Invalid image density provided: (n_x, n_y) = (%ld, %ld)", n_x_long, n_y_long);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    const unsigned int n_x = (const unsigned int)n_x_long;
    const unsigned int n_y = (const unsigned int)n_y_long;

    long double x_l, y_l, x_r, y_r;
    if (!parse_ld(argv[3], &x_l)) {
        fprintf(stderr, "Invalid long double for x_l: %s\n", argv[3]);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    if (!parse_ld(argv[4], &y_l)) {
        fprintf(stderr, "Invalid long double for y_l: %s\n", argv[4]);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    if (!parse_ld(argv[5], &x_r)) {
        fprintf(stderr, "Invalid long double for x_r: %s\n", argv[5]);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    if (!parse_ld(argv[6], &y_r)) {
        fprintf(stderr, "Invalid long double for y_r: %s\n", argv[6]);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    const long i_max_ = strtol(argv[7], NULL, 10);
    if (i_max_ > USHRT_MAX || i_max_ < 0) {
        fprintf(stderr, "The provided iteration is invalid (either negative or too large): %ld", i_max_);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1;
    }
    const unsigned int i_max = (const unsigned int)i_max_;

    const complex double btm_left = x_l + y_l*I;
    const complex double top_right = x_r + y_r*I;

    const unsigned int number_pixels = n_x * n_y;
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // There is some potential for threads affinity here.
    if (world_rank == 0) {
        // Master workload handles allocation, initialization, and computation.
        pixel *master_pixels = master_workload(btm_left, top_right, n_x, n_y, i_max);
        const char *outfilename = "mandelbrot.pgm";
        save_image(master_pixels, n_x, n_y, i_max, outfilename);
        free(master_pixels);
    } else {
        worker_workload(btm_left, top_right, n_x, n_y, i_max);
    }
    MPI_Type_free(&pixel_type);
    MPI_Finalize(); 
    return 0;
}