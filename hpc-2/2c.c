#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>

/*
 *  Forward declaration of the PGM helper functions defined in
 *  read_write_pgm_image.c.  The actual definitions are compiled
 *  alongside this file, so we only need the prototype here.
 */
void write_pgm_image(void *image, int maxval, int xsize, int ysize, const char *image_name);
void swap_image(void *image, int xsize, int ysize, int maxval);

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

static tile_t* make_tile_list(int n_y, int n_tiles) {
    tile_t* tiles = (tile_t*)calloc(n_tiles, sizeof(tile_t));
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
        shared(pixels, btm_left, top_right, dy, dx, i_max, n_x, n_y) \
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

static pixel* master_workload(
    const complex double btm_left,
    const complex double top_right,
    const int n_x, 
    const int n_y,
    const unsigned int i_max
    ) {
    /* Allocate the pixel buffer and initialize coordinates */
    unsigned int number_pixels = n_x * n_y;
    pixel *pixels = (pixel*)calloc(number_pixels, sizeof(pixel));
    if (!pixels) {
        fprintf(stderr, "Failed to allocate memory for pixels in master workload.\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
    }
    init_pixels(pixels, n_x, n_y);

    /* Determine number of tiles based on MPI world size */
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size == 1) {
        // If only one MPI process is detected then the master is responsible for computing the entire Mandelbrot.
        // Additional if statement to check removes the overhead of the master-worker structure.
        compute_mandelbrot(pixels, btm_left, top_right, n_x, n_y, i_max, (tile_t){0, n_y});
        return pixels;
    }
    int n_tiles = world_size;
    tile_t *tiles = make_tile_list(n_y, n_tiles);
    if (!tiles) {
        fprintf(stderr, "Failed to allocate tile list.\n");
        free(pixels);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
    }

    // TODO: Implement MPI broadcast/scatter.

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

}

int main(int argc, char* argv[]) {
    int mpi_provided_thread_level; 
    MPI_Init_threads(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided_thread_level); 
    if (mpi_provided_thread_level < MPI_THREAD_FUNNELED) { 
        printf("A problem arose when asking for MPI_THREAD_FUNNELED level.\n"); 
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return 1; 
    } 
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
    // There is some potential for threads affinity here.
    // Master workload handles allocation, initialization, and computation.
    pixel *master_pixels = master_workload(btm_left, top_right, n_x, n_y, i_max);
    const char *outfilename = "mandelbrot.pgm";
    save_image(master_pixels, n_x, n_y, i_max, outfilename);
    free(master_pixels);
    MPI_Finalize(); 
    return 0;
}