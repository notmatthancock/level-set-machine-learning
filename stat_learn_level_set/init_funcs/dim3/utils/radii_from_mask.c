#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "../../../utils/utils.h"
#include "../../../utils/trilinear.c"

/* Compute the radial distance from provided 3d `seed`
 * in direction given by theta and phi angles until a 
 * value of `false` is encountered in `mask`.
 */

void radii_from_mask(int ntpr, double * thetas, double * phis,
                     double * radii, double * seed, int m, int n, int p, 
                     double di, double dj, double dk,
                     bool * mask) {
    double ii,jj,kk;
    double a,b,c;
    double mval;
    double dt = 0.1;

    // Convert boolean mask to float to be used in interpolation.
    double * dmask = malloc(m*n*p * sizeof(double));
    for (int i=0; i < m*n*p; i++) {
        dmask[i] = mask[i] ? 1.0 : 0.0;
    }
    
    for (int i=0; i < ntpr; i++) {
        radii[i] = 0.0;
        
        ii = seed[0];
        jj = seed[1];
        kk = seed[2];

        a = dt*cos(thetas[i]) * sin(phis[i]);
        b = dt*sin(thetas[i]) * sin(phis[i]);
        c = dt*cos(phis[i]);

        while (true) {
            // Advance one step along ray.
            ii += a;
            jj += b;
            kk += c;
            radii[i] += dt;

            mval = interpolate_point(ii, jj, kk, dmask, di, dj, dk, m, n, p);

            if (mval < 0.5) break;
        }
    }

    // Free the memory used by the float mask.
    free(dmask);
}
