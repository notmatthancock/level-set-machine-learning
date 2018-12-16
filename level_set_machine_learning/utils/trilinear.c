#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "utils.h"

/* A big ugly function for performing trilinear interpolation on
 * a 3d scalar field.
 */

double interpolate_point(double i, double j, double k, double * img,
                         double di, double dj, double dk, // delta terms
                         int m, int n, int p) {
    int ilow, ihigh, jlow, jhigh, klow, khigh;

    // Return zero for out-of-bounds values.
    if (i < 0 || i > (m-1)*di ||
        j < 0 || j > (n-1)*dj ||
        k < 0 || k > (p-1)*dk) {
        return 0.0;
    }

    i /= di;
    j /= dj;
    k /= dk;

    ilow = (int) i;
    jlow = (int) j;
    klow = (int) k;

    ihigh = ilow + 1;
    jhigh = jlow + 1;
    khigh = klow + 1;

    if (i < m-1 && j < n-1 && k < p-1) {
        //printf("%.16f %.16f %.16f 1\n", i, j, k);
        // Full 3D interpolation.
        return (ihigh-i)*(jhigh-j)*(khigh-k)*img[mi3d(ilow , jlow , klow , m, n, p)] +
               (i-ilow )*(jhigh-j)*(khigh-k)*img[mi3d(ihigh, jlow , klow , m, n, p)] +
               (ihigh-i)*(j-jlow )*(khigh-k)*img[mi3d(ilow , jhigh, klow , m, n, p)] +
               (ihigh-i)*(jhigh-j)*(k-klow )*img[mi3d(ilow , jlow , khigh, m, n, p)] +
               (i-ilow )*(j-jlow )*(khigh-k)*img[mi3d(ihigh, jhigh, klow , m, n, p)] +
               (i-ilow )*(jhigh-j)*(k-klow )*img[mi3d(ihigh, jlow , khigh, m, n, p)] +
               (ihigh-i)*(j-jlow )*(k-klow )*img[mi3d(ilow , jhigh, khigh, m, n, p)] +
               (i-ilow )*(j-jlow )*(k-klow )*img[mi3d(ihigh, jhigh, khigh, m, n, p)];
    }
    else if (i == m-1 && j < n-1 && k < p-1) {
        //printf("2\n");
        // 2D interpolation on bottom face.
        return (jhigh-j)*(khigh-k)*img[mi3d(m-1, jlow , klow , m, n, p)] +
               (j-jlow )*(khigh-k)*img[mi3d(m-1, jhigh, klow , m, n, p)] +
               (jhigh-j)*(k-klow )*img[mi3d(m-1, jlow , khigh, m, n, p)] +
               (j-jlow )*(k-klow )*img[mi3d(m-1, jhigh, khigh, m, n, p)];
    }
    else if (i < m-1 && j == n-1 && k < p-1) {
        //printf("3\n");
        // 2D interpolation on right face.
        return (ihigh-i)*(khigh-k)*img[mi3d(ilow , n-1, klow , m, n, p)] +
               (i-ilow )*(khigh-k)*img[mi3d(ihigh, n-1, klow , m, n, p)] +
               (ihigh-i)*(k-klow )*img[mi3d(ilow , n-1, khigh, m, n, p)] +
               (i-ilow )*(k-klow )*img[mi3d(ihigh, n-1, khigh, m, n, p)];
    }
    else if (i < m-1 && j < n-1 && k == p-1) {
        //printf("4\n");
        // 2D interpolation on back face.
        return (ihigh-i)*(jhigh-j)*img[mi3d(ilow , jlow , p-1, m, n, p)] +
               (i-ilow )*(jhigh-j)*img[mi3d(ihigh, jlow , p-1, m, n, p)] +
               (ihigh-i)*(j-jlow )*img[mi3d(ilow , jhigh, p-1, m, n, p)] +
               (i-ilow )*(j-jlow )*img[mi3d(ihigh, jhigh, p-1, m, n, p)];
    }
    else if (i == m-1 && j == n-1 && k < p-1) {
        //printf("5\n");
        // 1D interpolation along bottom right edge.
        return (khigh-k)*img[mi3d(m-1, n-1, klow , m, n, p)] +
               (k-klow )*img[mi3d(m-1, n-1, khigh, m, n, p)];
    }
    else if (i == m-1 && j < n-1 && k == p-1) {
        //printf("6\n");
        // 1D interpolation along back bottom edge.
        return (jhigh-j)*img[mi3d(m-1, jlow , p-1, m, n, p)] +
               (j-jlow )*img[mi3d(m-1, jhigh, p-1, m, n, p)];
    }
    else if (i < m-1 && j == n-1 && k == p-1) {
        //printf("7\n");
        // 1D interpolation along back, right edge.
        //  _
        // /_/|<-
        // |_|/
        return (ihigh-i)*img[mi3d(ilow , n-1, p-1, m, n, p)] +
               (i-ilow )*img[mi3d(ihigh, n-1, p-1, m, n, p)];
    }
    else {
        //printf("8\n");
        return img[mi3d(m-1, n-1, p-1, m, n, p)];
    }
}

//void interpolate(double * igrid, double * jgrid, double * kgrid,
//                 int q, int r, int s, double * img, int m, int n, int p,
//                 double * irp) {
//    int l;
//
//    for (int i=0; i < q; i++) {
//        for (int j=0; j < r; j++) {
//            for (int k=0; k < s; k++) {
//                l = mi3d(i,j,k,q,r,s);
//                irp[l] = interpolate_point(igrid[l], jgrid[l], kgrid[l],
//                                           img, m, n, p);
//            }
//        }
//    }
//}
