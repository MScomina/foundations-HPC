#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/**
 * Helper function for parsing arguments to long double.
 */
bool parse_ld(const char* s, long double* out) {
    char* endptr;
    errno = 0;
    long double val = strtold(s, &endptr);

    if (errno != 0)               return false;
    if (endptr == s)              return false;
    if (*endptr != '\0')          return false;

    *out = val;
    return true;
}

/**
 * Function responsible for the starting initialization of the pixels array.
 */
void init_pixels(pixel* restrict pixels, const unsigned int n_x, const unsigned int n_y) {
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
    const unsigned int i_max
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
    for (int line = 0; line<n_y; line++) {
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
void save_image(pixel* pixels,
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
            return;
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
            return;
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

int main(int argc, char* argv[]) {
    /* ---------- Timing setup ---------- */
    double t0 = omp_get_wtime();
    double init_start, init_end;
    double compute_start, compute_end;
    double save_start, save_end;
    double total_time;
    /* --------------------------------- */
    if (argc < 8) {
        fprintf(stderr, "Please enter all of the required arguments (specifically, in order: n_x, n_y, x_L, y_L, x_R, y_R, I_max).");
        return 1;
    }
    const long n_x_long = strtol(argv[1], NULL, 10);
    const long n_y_long = strtol(argv[2], NULL, 10);
    if (n_x_long > UINT_MAX || n_y_long > UINT_MAX || n_x_long < 0 || n_y_long < 0) {
        fprintf(stderr, "Invalid image density provided: (n_x, n_y) = (%ld, %ld)", n_x_long, n_y_long);
        return 1;
    }
    const unsigned int n_x = (const unsigned int)n_x_long;
    const unsigned int n_y = (const unsigned int)n_y_long;

    long double x_l, y_l, x_r, y_r;
    if (!parse_ld(argv[3], &x_l)) {
        fprintf(stderr, "Invalid long double for x_l: %s\n", argv[3]);
        return 1;
    }
    if (!parse_ld(argv[4], &y_l)) {
        fprintf(stderr, "Invalid long double for y_l: %s\n", argv[4]);
        return 1;
    }
    if (!parse_ld(argv[5], &x_r)) {
        fprintf(stderr, "Invalid long double for x_r: %s\n", argv[5]);
        return 1;
    }
    if (!parse_ld(argv[6], &y_r)) {
        fprintf(stderr, "Invalid long double for y_r: %s\n", argv[6]);
        return 1;
    }
    const long i_max_ = strtol(argv[7], NULL, 10);
    if (i_max_ > USHRT_MAX || i_max_ < 0) {
        fprintf(stderr, "The provided iteration is invalid (either negative or too large): %ld", i_max_);
        return 1;
    }
    const unsigned int i_max = (const unsigned int)i_max_;

    const complex double btm_left = x_l + y_l*I;
    const complex double top_right = x_r + y_r*I;

    const unsigned int number_pixels = n_x * n_y;
    // There is some potential for threads affinity here.
    pixel* pixels = (pixel*)calloc(number_pixels, sizeof(pixel));
    init_start = omp_get_wtime();
    init_pixels(pixels, n_x, n_y);
    init_end = omp_get_wtime();
    compute_start = omp_get_wtime();
    compute_mandelbrot(pixels, btm_left, top_right, n_x, n_y, i_max);
    compute_end = omp_get_wtime();
    const char *outfilename = "mandelbrot.pgm";
    save_start = omp_get_wtime();
    save_image(pixels, n_x, n_y, i_max, outfilename);
    save_end = omp_get_wtime();
    free(pixels);
    total_time = omp_get_wtime() - t0;
    fprintf(stderr, "\nTiming summary:\n");
    fprintf(stderr, "  init   : %.6f s\n", init_end - init_start);
    fprintf(stderr, "  compute: %.6f s\n", compute_end - compute_start);
    fprintf(stderr, "  save   : %.6f s\n", save_end - save_start);
    fprintf(stderr, "  total  : %.6f s\n", total_time);
    return 0;
}