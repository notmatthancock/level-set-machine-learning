#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "utils.h"
#include "trilinear.c"

// di, dj, dk should be normalized to be unit length.
// samples is (n,n,n, 2*nsamples)
void get_samples(int n, double * img, bool * mask, double * com,
                 int nsamples, double * samples) {
    int l,ll;
    double a, b, c;
    double ii_i, jj_i, kk_i;
    double ii_o, jj_o, kk_o;
    bool in_bounds_i, in_bounds_o, is_zero;
    double dt,cdist;

    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            for (int k=0; k < n; k++) {
                l = map_index(i,j,k,n,n,n);
                if (!mask[l]) continue;

                cdist = sqrt(sqr(i-com[0])+sqr(j-com[1])+sqr(k-com[2]));
                dt = cdist / (nsamples+1.0);

                a = dt * (com[0]-i) / cdist;
                b = dt * (com[1]-j) / cdist;
                c = dt * (com[2]-k) / cdist;

                // The gradient vector is zero, so we can't compute
                // the feature for this coordinate, (i,j,k).
                is_zero = false;
                if (a == 0 && b == 0 && c == 0) is_zero = true;

                ii_i = (double) i;
                jj_i = (double) j;
                kk_i = (double) k;

                ii_o = (double) i;
                jj_o = (double) j;
                kk_o = (double) k;

                for (int q=1; q <= nsamples; q++) {
                    in_bounds_i = check_bounds((int) round(ii_i),
                                               (int) round(jj_i),
                                               (int) round(kk_i), n, n, n);
                    in_bounds_o = check_bounds((int) round(ii_o),
                                               (int) round(jj_o),
                                               (int) round(kk_o), n, n, n);

                    ll = nsamples*l + q-1;

                    if (!in_bounds_i || is_zero)
                        samples[2*ll+0] = 0.0;
                    else
                        samples[2*ll+0] = interpolate_point(ii_i, jj_i, kk_i,
                                                            img, n, n, n);
                    if (!in_bounds_o || is_zero)
                        samples[2*ll+1] = 0.0;
                    else
                        samples[2*ll+1] = interpolate_point(ii_o, jj_o, kk_o,
                                                            img, n, n, n);

                    // Advance one step along ray.
                    ii_i += a;
                    jj_i += b;
                    kk_i += c;

                    ii_o -= a;
                    jj_o -= b;
                    kk_o -= c;
                }

            } // End loop k.
        } // End loop j.
    } // End loop k.
}
