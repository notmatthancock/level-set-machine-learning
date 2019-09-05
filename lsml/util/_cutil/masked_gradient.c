/*
 * Masked gradient
 * ---------------
 * Routines for computing gradients over masked regions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "helpers.c"


void gradient_centered3d(int m, int n, int p, double * A, bool * mask,
                         double * di, double * dj, double * dk, double * gmag,
                         double deli, double delj, double delk,
                         int normalize) {
    int l;

    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            for(int k=0; k < p; k++) {
                l = mi3d(i,j,k,m,n,p);

                if (!mask[l]) continue;

                if (i == 0) {
                    di[l] = A[mi3d(i+1,j,k,m,n,p)] - A[l];
                }
                else if (i == m-1) {
                    di[l] = A[l] - A[mi3d(i-1,j,k,m,n,p)];
                }
                else {
                    di[l] = 0.5*(A[mi3d(i+1,j,k,m,n,p)] -\
                                 A[mi3d(i-1,j,k,m,n,p)]);
                }

                // Gradient along j axes.
                if (j == 0) {
                    dj[l] = A[mi3d(i,j+1,k,m,n,p)] - A[l];
                }
                else if (j == n-1) {
                    dj[l] = A[l] - A[mi3d(i,j-1,k,m,n,p)];
                }
                else {
                    dj[l] = 0.5*(A[mi3d(i,j+1,k,m,n,p)] -\
                                 A[mi3d(i,j-1,k,m,n,p)]);
                }

                // Gradient along k axes.
                if (k == 0) {
                    dk[l] = A[mi3d(i,j,k+1,m,n,p)] - A[l];
                }
                else if (k == p-1) {
                    dk[l] = A[l] - A[mi3d(i,j,k-1,m,n,p)];
                }
                else {
                    dk[l] = 0.5*(A[mi3d(i,j,k+1,m,n,p)] -\
                                 A[mi3d(i,j,k-1,m,n,p)]);
                }

                di[l] = di[l] / deli;
                dj[l] = dj[l] / delj;
                dk[l] = dk[l] / delk;

                gmag[l] = sqrt(sqr(di[l]) + sqr(dj[l]) + sqr(dk[l]));

                if (normalize == 1 && gmag[l] > 0) {
                    di[l] /= gmag[l];
                    dj[l] /= gmag[l];
                    dk[l] /= gmag[l];
                }
            } // End k loop.
        } // End j loop.
    } // End i loop.
}

void gradient_centered2d(int m, int n, double * A, bool * mask,
                         double * di, double * dj, double * gmag,
                         double deli, double delj,
                         int normalize) {
    int l;

    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            l = mi2d(i,j,m,n);

            if (!mask[l]) continue;

            if (i == 0) {
                di[l] = A[mi2d(i+1,j,m,n)] - A[l];
            }
            else if (i == m-1) {
                di[l] = A[l] - A[mi2d(i-1,j,m,n)];
            }
            else {
                di[l] = 0.5*(A[mi2d(i+1,j,m,n)] -\
                             A[mi2d(i-1,j,m,n)]);
            }

            // Gradient along j axes.
            if (j == 0) {
                dj[l] = A[mi2d(i,j+1,m,n)] - A[l];
            }
            else if (j == n-1) {
                dj[l] = A[l] - A[mi2d(i,j-1,m,n)];
            }
            else {
                dj[l] = 0.5*(A[mi2d(i,j+1,m,n)] -\
                             A[mi2d(i,j-1,m,n)]);
            }

            di[l] = di[l] / deli;
            dj[l] = dj[l] / delj;

            gmag[l] = sqrt(sqr(di[l]) + sqr(dj[l]));

            if (normalize == 1 && gmag[l] > 0) {
                di[l] /= gmag[l];
                dj[l] /= gmag[l];
            }
        } // End j loop.
    } // End i loop.
}

void gradient_centered1d(int m, double * A, bool * mask,
                         double * di, double * gmag,
                         double deli,
                         int normalize) {
    for(int i=0; i < m; i++) {
        if (!mask[i]) continue;

        if (i == 0) {
            di[i] = A[i+1] - A[i];
        }
        else if (i == m-1) {
            di[i] = A[i] - A[i-1];
        }
        else {
            di[i] = 0.5*(A[i+1] - A[i-1]);
        }

        di[i] = di[i] / deli;

        gmag[i] = (di[i] > 0) ? di[i] : -di[i];

        if (normalize == 1 && gmag[i] > 0) {
            di[i] /= gmag[i];
        }
    } // End i loop.
}

void gmag_os3d(int m, int n, int p, double * A, bool * mask,
               double * nu, double * gmag,
               double deli, double delj, double delk) {
    int l;
    double fi,fj,fk,bi,bj,bk;

    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            for(int k=0; k < p; k++) {

                l = mi3d(i,j,k,m,n,p);
                if (!mask[l]) continue;

                if (i == 0) {
                    fi = A[mi3d(i+1,j,k,m,n,p)] - A[l];
                    bi = fi;
                }
                else if (i == m-1) {
                    bi = A[l] - A[mi3d(i-1,j,k,m,n,p)];
                    fi = bi;
                }
                else {
                    fi = A[mi3d(i+1,j,k,m,n,p)] - A[l];
                    bi = A[l] - A[mi3d(i-1,j,k,m,n,p)];
                }

                // Gradient along j axes.
                if (j == 0) {
                    fj = A[mi3d(i,j+1,k,m,n,p)] - A[l];
                    bj = fj;
                }
                else if (j == n-1) {
                    bj = A[l] - A[mi3d(i,j-1,k,m,n,p)];
                    fj = bj;
                }
                else {
                    fj = A[mi3d(i,j+1,k,m,n,p)] - A[l];
                    bj = A[l] - A[mi3d(i,j-1,k,m,n,p)];
                }

                // Gradient along k axes.
                if (k == 0) {
                    fk = A[mi3d(i,j,k+1,m,n,p)] - A[l];
                    bk = fk;
                }
                else if (k == p-1) {
                    bk = A[l] - A[mi3d(i,j,k-1,m,n,p)];
                    fk = bk;
                }
                else {
                    fk = A[mi3d(i,j,k+1,m,n,p)] - A[l];
                    bk = A[l] - A[mi3d(i,j,k-1,m,n,p)];
                }

                fi = fi/deli;
                bi = bi/deli;
                fj = fj/delj;
                bj = bj/delj;
                fk = fk/delk;
                bk = bk/delk;

                if (nu[l] < 0) {
                    gmag[l] = sqrt(sqr(max(bi,0)) + sqr(min(fi,0)) + \
                                   sqr(max(bj,0)) + sqr(min(fj,0)) + \
                                   sqr(max(bk,0)) + sqr(min(fk,0)));
                }
                else {
                    gmag[l] = sqrt(sqr(min(bi,0)) + sqr(max(fi,0)) + \
                                   sqr(min(bj,0)) + sqr(max(fj,0)) + \
                                   sqr(min(bk,0)) + sqr(max(fk,0)));
                } // End if speed.
            } // End k loop.
        } // End j loop.
    } // End i loop.
}

void gmag_os2d(int m, int n, double * A, bool * mask,
               double * nu, double * gmag,
               double deli, double delj) {
    int l;
    double fi,fj,bi,bj;

    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            l = mi2d(i,j,m,n);

            if (!mask[l]) continue;

            if (i == 0) {
                fi = A[mi2d(i+1,j,m,n)] - A[l];
                bi = fi;
            }
            else if (i == m-1) {
                bi = A[l] - A[mi2d(i-1,j,m,n)];
                fi = bi;
            }
            else {
                fi = A[mi2d(i+1,j,m,n)] - A[l];
                bi = A[l] - A[mi2d(i-1,j,m,n)];
            }

            // Gradient along j axes.
            if (j == 0) {
                fj = A[mi2d(i,j+1,m,n)] - A[l];
                bj = fj;
            }
            else if (j == n-1) {
                bj = A[l] - A[mi2d(i,j-1,m,n)];
                fj = bj;
            }
            else {
                fj = A[mi2d(i,j+1,m,n)] - A[l];
                bj = A[l] - A[mi2d(i,j-1,m,n)];
            }

            fi = fi/deli;
            bi = bi/deli;
            fj = fj/delj;
            bj = bj/delj;

            if (nu[l] < 0) {
                gmag[l] = sqrt(sqr(max(bi,0)) + sqr(min(fi,0)) + \
                               sqr(max(bj,0)) + sqr(min(fj,0)));
            }
            else {
                gmag[l] = sqrt(sqr(min(bi,0)) + sqr(max(fi,0)) + \
                               sqr(min(bj,0)) + sqr(max(fj,0)));
            } // End if speed.
        } // End j loop.
    } // End i loop.
}

void gmag_os1d(int m, double * A, bool * mask,
               double * nu, double * gmag,
               double deli) {
    double fi,bi;

    for(int i=0; i < m; i++) {
        if (!mask[i]) continue;

        if (i == 0) {
            fi = A[i+1] - A[i];
            bi = fi;
        }
        else if (i == m-1) {
            bi = A[i] - A[i-1];
            fi = bi;
        }
        else {
            fi = A[i+1] - A[i];
            bi = A[i] - A[i-1];
        }

        fi = fi/deli;
        bi = bi/deli;

        if (nu[i] < 0) {
            gmag[i] = sqrt(sqr(max(bi,0)) + sqr(min(fi,0)));
        }
        else {
            gmag[i] = sqrt(sqr(min(bi,0)) + sqr(max(fi,0)));
        } // End if speed.
    } // End i loop.
}
