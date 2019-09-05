/*
 * Helper utilities for C libraries
 * --------------------------------
 * Mostly, index mappers into flattened multi-dimensional arrays
 */

#ifndef C_HELPERS
#define C_HELPERS

#include <stdlib.h>

#define PI    3.14159265358979311599796346854419
#define TWOPI 6.28318530717958623199592693708837

// Map the index (i,j,k) to an index l, the "flat" index
// into the row-major 3D array of dimensions, (m,n,p).
int inline mi3d(int i, int j, int k, int m, int n, int p) {
    #if MI_CHECK_INDEX
    int do_abort = 0;
    if (i < 0 || i > m-1) {
        printf("Bad index: i = %d (valid = 0-%d).\n", i, m-1);
        do_abort = 1;
    }
    if (j < 0 || j > n-1) {
        printf("Bad index: j = %d (valid = 0-%d).\n", j, n-1);
        do_abort = 1;
    }
    if (k < 0 || k > p-1) {
        printf("Bad index: k = %d (valid = 0-%d).\n", k, p-1);
        do_abort = 1;
    }
    if (do_abort == 1) abort();
    #endif
    return n*p*i + p*j + k;
}

// Map the index (i,j) to an index l, the "flat" index
// into the row-major 2D array of dimensions, (m,n).
int inline mi2d(int i, int j, int m, int n) {
    #if MI_CHECK_INDEX
    int do_abort = 0;
    if (i < 0 || i > m-1) {
        printf("Bad index: i = %d (valid = 0-%d).\n", i, m-1);
        do_abort = 1;
    }
    if (j < 0 || j > n-1) {
        printf("Bad index: j = %d (valid = 0-%d).\n", j, n-1);
        do_abort = 1;
    }
    if (do_abort == 1) abort();
    #endif
    return n*i + j;
}


// Simple math functions.
double inline max(double a, double b) { return a < b ? b : a; }
double inline min(double a, double b) { return a > b ? b : a; }
double inline sqr(double a) { return a*a; }

bool inline check_bounds(int i, int j, int k, int m, int n, int p) {
    if (i < 0) return false; 
    if (j < 0) return false;
    if (k < 0) return false;
    if (i > m-1) return false;
    if (j > n-1) return false;
    if (k > p-1) return false;
    return true;
}


#endif
