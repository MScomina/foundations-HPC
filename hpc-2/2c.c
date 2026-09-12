#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

typedef struct {
    int x;
    int y;
    char data;
} pixel;

typedef struct {
    pixel* pixel;
    unsigned int iterations;
    bool completed;
    complex long double c_data;
    complex long double z_data;
} unfinished_pixel;

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
void init_pixels(pixel* pixels, const unsigned int n_x, const unsigned int n_y) {
    for (unsigned int row = 0; row < n_y; ++row) {
        for (unsigned int column = 0; column < n_x; ++column) {
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
    pixel* pixels, 
    const complex double* btm_left,
    const complex double* top_right,
    const int n_x, 
    const int n_y,
    const int number_pixels,
    const unsigned int* T
    ) {
    return true;
}

int main(int argc, char* argv[]) {
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
    pixel* pixels = (pixel*)calloc(number_pixels, sizeof(pixel));
    init_pixels(pixels, n_x, n_y);
    compute_mandelbrot(pixels, &btm_left, &top_right, n_x, n_y, number_pixels, &i_max);
    free(pixels);
    return 0;
}