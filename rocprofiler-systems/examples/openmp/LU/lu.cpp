/*
MIT License

Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
    GMAP website: https://gmap.pucrs.br

    Pontifical Catholic University of Rio Grande do Sul (PUCRS)
    Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

The original NPB 3.4.1 version was written in Fortran and belongs to:
    http://www.nas.nasa.gov/Software/NPB/

Authors of the Fortran code:
    S. Weeratunga
    V. Venkatakrishnan
    E. Barszcz
    M. Yarrow
    H. Jin

------------------------------------------------------------------------------

The serial C++ version is a translation of the original NPB 3.4.1
Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER

Authors of the C++ code:
    Dalvan Griebler <dalvangriebler@gmail.com>
    Gabriell Araujo <hexenoften@gmail.com>
    Júnior Löff <loffjh@gmail.com>

------------------------------------------------------------------------------

The OpenMP version is a parallel implementation of the serial C++ version
OpenMP version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-OMP

Authors of the OpenMP code:
    Júnior Löff <loffjh@gmail.com>

*/

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * driver for the performance evaluation of the solver for
 * five coupled parabolic/elliptic partial differential equations
 * ---------------------------------------------------------------------
 * parameters which can be overridden in runtime config file
 * isiz1,isiz2,isiz3 give the maximum size
 * ipr = 1 to print out verbose information
 * omega = 2.0 is correct for all classes
 * tolrsd is tolerance levels for steady state residuals
 * ---------------------------------------------------------------------
 * field variables and residuals
 * to improve cache performance, second two dimensions padded by 1
 * for even number sizes only.
 * note: corresponding array (called "v") in routines blts, buts,
 * and l2norm are similarly padded
 * ---------------------------------------------------------------------
 */
#define IPR_DEFAULT   1
#define OMEGA_DEFAULT 1.2
#define TOLRSD1_DEF   1.0e-08
#define TOLRSD2_DEF   1.0e-08
#define TOLRSD3_DEF   1.0e-08
#define TOLRSD4_DEF   1.0e-08
#define TOLRSD5_DEF   1.0e-08
#define C1            1.40e+00
#define C2            0.40e+00
#define C3            1.00e-01
#define C4            1.00e+00
#define C5            1.40e+00
#define T_TOTAL       1
#define T_RHSX        2
#define T_RHSY        3
#define T_RHSZ        4
#define T_RHS         5
#define T_JACLD       6
#define T_BLTS        7
#define T_JACU        8
#define T_BUTS        9
#define T_ADD         10
#define T_L2NORM      11
#define T_LAST        11

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static double u[ISIZ3][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5];
static double rsd[ISIZ3][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5];
static double frct[ISIZ3][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5];
static double flux[ISIZ1][5];
static double qs[ISIZ3][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1];
static double rho_i[ISIZ3][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1];
static double a[ISIZ2][ISIZ1 / 2 * 2 + 1][5][5];
static double b[ISIZ2][ISIZ1 / 2 * 2 + 1][5][5];
static double c[ISIZ2][ISIZ1 / 2 * 2 + 1][5][5];
static double d[ISIZ2][ISIZ1 / 2 * 2 + 1][5][5];
static double ce[13][5];
#else
static double (*u)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5] =
    (double (*)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5])
        malloc(sizeof(double) *
               ((ISIZ3) * (ISIZ2 / 2 * 2 + 1) * (ISIZ1 / 2 * 2 + 1) * (5)));
static double (*rsd)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5] =
    (double (*)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5])
        malloc(sizeof(double) *
               ((ISIZ3) * (ISIZ2 / 2 * 2 + 1) * (ISIZ1 / 2 * 2 + 1) * (5)));
static double (*frct)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5] =
    (double (*)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5])
        malloc(sizeof(double) *
               ((ISIZ3) * (ISIZ2 / 2 * 2 + 1) * (ISIZ1 / 2 * 2 + 1) * (5)));
static double (*flux)[5] = (double (*)[5]) malloc(sizeof(double) * ((ISIZ1) * (5)));
static double (*qs)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1] =
    (double (*)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1])
        malloc(sizeof(double) * ((ISIZ3) * (ISIZ2 / 2 * 2 + 1) * (ISIZ1 / 2 * 2 + 1)));
static double (*rho_i)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1] =
    (double (*)[ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1])
        malloc(sizeof(double) * ((ISIZ3) * (ISIZ2 / 2 * 2 + 1) * (ISIZ1 / 2 * 2 + 1)));
static double (*a)[ISIZ1 / 2 * 2 + 1][5][5] = (double (*)[ISIZ1 / 2 * 2 + 1][5][5])
    malloc(sizeof(double) * ((ISIZ2) * (ISIZ1 / 2 * 2 + 1) * (5) * (5)));
static double (*b)[ISIZ1 / 2 * 2 + 1][5][5] = (double (*)[ISIZ1 / 2 * 2 + 1][5][5])
    malloc(sizeof(double) * ((ISIZ2) * (ISIZ1 / 2 * 2 + 1) * (5) * (5)));
static double (*c)[ISIZ1 / 2 * 2 + 1][5][5] = (double (*)[ISIZ1 / 2 * 2 + 1][5][5])
    malloc(sizeof(double) * ((ISIZ2) * (ISIZ1 / 2 * 2 + 1) * (5) * (5)));
static double (*d)[ISIZ1 / 2 * 2 + 1][5][5] = (double (*)[ISIZ1 / 2 * 2 + 1][5][5])
    malloc(sizeof(double) * ((ISIZ2) * (ISIZ1 / 2 * 2 + 1) * (5) * (5)));
static double (*ce)[5] = (double (*)[5]) malloc(sizeof(double) * ((13) * (5)));
#endif
/* grid */
static double dxi, deta, dzeta;
static double tx1, tx2, tx3;
static double ty1, ty2, ty3;
static double tz1, tz2, tz3;
static int    nx, ny, nz;
static int    nx0, ny0, nz0;
static int    ist, iend;
static int    jst, jend;
static int    ii1, ii2;
static int    ji1, ji2;
static int    ki1, ki2;
/* dissipation */
static double dx1, dx2, dx3, dx4, dx5;
static double dy1, dy2, dy3, dy4, dy5;
static double dz1, dz2, dz3, dz4, dz5;
static double dssp;
/* output control parameters */
static int ipr, inorm;
/* newton-raphson iteration control parameters */
static double dt, omega, tolrsd[5], rsdnm[5], errnm[5], frc;
static int    itmax;
/* timer */
static double  maxtime;
static boolean timeron;

/* function prototypes */
void
blts(int nx, int ny, int nz, int k, double omega,
     double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5],
     double ldz[][ISIZ1 / 2 * 2 + 1][5][5], double ldy[][ISIZ1 / 2 * 2 + 1][5][5],
     double ldx[][ISIZ1 / 2 * 2 + 1][5][5], double d[][ISIZ1 / 2 * 2 + 1][5][5], int ist,
     int iend, int jst, int jend, int nx0, int ny0);
void
buts(int nx, int ny, int nz, int k, double omega,
     double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5], void* pointer_tv,
     double d[][ISIZ1 / 2 * 2 + 1][5][5], double udx[][ISIZ1 / 2 * 2 + 1][5][5],
     double udy[][ISIZ1 / 2 * 2 + 1][5][5], double udz[][ISIZ1 / 2 * 2 + 1][5][5],
     int ist, int iend, int jst, int jend, int nx0, int ny0);
void
domain();
void
erhs();
void
error();
void
exact(int i, int j, int k, double u000ijk[]);
void
jacld(int k);
void
jacu(int k);
void
l2norm(int nx0, int ny0, int nz0, int ist, int iend, int jst, int jend,
       double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5], double sum[5]);
void
pintgr();
void
read_input();
void
rhs();
void
setbv();
void
setcoeff();
void
setiv();
void
ssor(int niter);
void
verify(double xcr[], double xce[], double xci, char* class_npb, boolean* verified);

static boolean flag[ISIZ1 / 2 * 2 + 1];
static boolean flag2[ISIZ1 / 2 * 2 + 1];

/* lu */
int
main(int, char*[])
{
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
    printf(
        " DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
    char    class_npb;
    boolean verified;
    double  mflops;
    double  t, tmax, trecs[T_LAST + 1];
    int     i;
    char*   t_names[T_LAST + 1];
    /*
     * ---------------------------------------------------------------------
     * setup info for timers
     * ---------------------------------------------------------------------
     */
    FILE* fp;
    if((fp = fopen("timer.flag", "r")) != NULL)
    {
        timeron           = TRUE;
        t_names[T_TOTAL]  = (char*) "total";
        t_names[T_RHSX]   = (char*) "rhsx";
        t_names[T_RHSY]   = (char*) "rhsy";
        t_names[T_RHSZ]   = (char*) "rhsz";
        t_names[T_RHS]    = (char*) "rhs";
        t_names[T_JACLD]  = (char*) "jacld";
        t_names[T_BLTS]   = (char*) "blts";
        t_names[T_JACU]   = (char*) "jacu";
        t_names[T_BUTS]   = (char*) "buts";
        t_names[T_ADD]    = (char*) "add";
        t_names[T_L2NORM] = (char*) "l2norm";
        fclose(fp);
    }
    else
    {
        timeron = FALSE;
    }
    /*
     * ---------------------------------------------------------------------
     * read input data
     * ---------------------------------------------------------------------
     */
    read_input();
    /*
     * ---------------------------------------------------------------------
     * set up domain sizes
     * ---------------------------------------------------------------------
     */
    domain();
    /*
     * ---------------------------------------------------------------------
     * set up coefficients
     * ---------------------------------------------------------------------
     */
    setcoeff();

#pragma omp parallel
    {
        /*
         * ---------------------------------------------------------------------
         * set the boundary values for dependent variables
         * ---------------------------------------------------------------------
         */
        setbv();
        /*
         * ---------------------------------------------------------------------
         * set the initial values for dependent variables
         * ---------------------------------------------------------------------
         */
        setiv();
        /*
         * ---------------------------------------------------------------------
         * compute the forcing term based on prescribed exact solution
         * ---------------------------------------------------------------------
         */
        erhs();
    } /* end parallel */

    /*
     * ---------------------------------------------------------------------
     * perform one SSOR iteration to touch all pages
     * ---------------------------------------------------------------------
     */
    ssor(1);
#pragma omp parallel
    {
        /*
         * ---------------------------------------------------------------------
         * reset the boundary and initial values
         * ---------------------------------------------------------------------
         */
        setbv();
        setiv();
    }

    /*
     * ---------------------------------------------------------------------
     * perform the SSOR iterations
     * ---------------------------------------------------------------------
     */
    ssor(itmax);
    /*
     * ---------------------------------------------------------------------
     * compute the solution error
     * ---------------------------------------------------------------------
     */
    error();
    /*
     * ---------------------------------------------------------------------
     * compute the surface integral
     * ---------------------------------------------------------------------
     */
    pintgr();
    /*
     * ---------------------------------------------------------------------
     * verification test
     * ---------------------------------------------------------------------
     */
    verify(rsdnm, errnm, frc, &class_npb, &verified);
    mflops = (double) itmax *
             (1984.77 * (double) nx0 * (double) ny0 * (double) nz0 -
              10923.3 * pow(((double) (nx0 + ny0 + nz0) / 3.0), 2.0) +
              27770.9 * (double) (nx0 + ny0 + nz0) / 3.0 - 144010.0) /
             (maxtime * 1000000.0);
    setenv("OMP_NUM_THREADS", "1", 0);
    c_print_results((char*) "LU", class_npb, nx0, ny0, nz0, itmax, maxtime, mflops,
                    (char*) "          floating point", verified, (char*) NPBVERSION,
                    (char*) COMPILETIME, (char*) COMPILERVERSION, (char*) LIBVERSION,
                    std::getenv("OMP_NUM_THREADS"), (char*) CS1, (char*) CS2, (char*) CS3,
                    (char*) CS4, (char*) CS5, (char*) CS6, (char*) "(none)");
    /*
     * ---------------------------------------------------------------------
     * more timers
     * ---------------------------------------------------------------------
     */
    if(timeron)
    {
        for(i = 1; i <= T_LAST; i++)
        {
            trecs[i] = timer_read(i);
        }
        tmax = maxtime;
        if(tmax == 0.0)
        {
            tmax = 1.0;
        }
        printf("  SECTION     Time (secs)\n");
        for(i = 1; i <= T_LAST; i++)
        {
            printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[i], trecs[i],
                   trecs[i] * 100. / tmax);
            if(i == T_RHS)
            {
                t = trecs[T_RHSX] + trecs[T_RHSY] + trecs[T_RHSZ];
                printf("     --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t * 100. / tmax);
                t = trecs[i] - t;
                printf("     --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t * 100. / tmax);
            }
        }
    }
    return 0;
}

/*
 * ---------------------------------------------------------------------
 * compute the regular-sparse, block lower triangular solution:
 * v <-- ( L-inv ) * v
 * ---------------------------------------------------------------------
 * to improve cache performance, second two dimensions padded by 1
 * for even number sizes only. only needed in v.
 * ---------------------------------------------------------------------
 */
void
blts(int /*nx*/, int /*ny*/, int /*nz*/, int k, double omega,
     double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5],
     double ldz[][ISIZ1 / 2 * 2 + 1][5][5], double ldy[][ISIZ1 / 2 * 2 + 1][5][5],
     double ldx[][ISIZ1 / 2 * 2 + 1][5][5], double d[][ISIZ1 / 2 * 2 + 1][5][5], int ist,
     int iend, int jst, int jend, int /*nx0*/, int /*ny0*/)
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, m;
    double tmp, tmp1;
    double tmat[5][5], tv[5];

#pragma omp for nowait schedule(static)
    for(j = jst; j < jend; j++)
    {
        for(i = ist; i < iend; i++)
        {
            for(m = 0; m < 5; m++)
            {
                v[k][j][i][m] =
                    v[k][j][i][m] - omega * (ldz[j][i][0][m] * v[k - 1][j][i][0] +
                                             ldz[j][i][1][m] * v[k - 1][j][i][1] +
                                             ldz[j][i][2][m] * v[k - 1][j][i][2] +
                                             ldz[j][i][3][m] * v[k - 1][j][i][3] +
                                             ldz[j][i][4][m] * v[k - 1][j][i][4]);
            }
        }
    }

#pragma omp for nowait schedule(static)
    for(j = jst; j < jend; j++)
    {
        if(j != jst)
        {
            while(flag[j - 1] == 0)
            {
#pragma omp flush
                ;
            }
        }
        if(j != jend - 1)
        {
            while(flag[j] == 1)
            {
#pragma omp flush
                ;
            }
        }

        for(i = ist; i < iend; i++)
        {
            for(m = 0; m < 5; m++)
            {
                tv[m] = v[k][j][i][m] - omega * (ldy[j][i][0][m] * v[k][j - 1][i][0] +
                                                 ldx[j][i][0][m] * v[k][j][i - 1][0] +
                                                 ldy[j][i][1][m] * v[k][j - 1][i][1] +
                                                 ldx[j][i][1][m] * v[k][j][i - 1][1] +
                                                 ldy[j][i][2][m] * v[k][j - 1][i][2] +
                                                 ldx[j][i][2][m] * v[k][j][i - 1][2] +
                                                 ldy[j][i][3][m] * v[k][j - 1][i][3] +
                                                 ldx[j][i][3][m] * v[k][j][i - 1][3] +
                                                 ldy[j][i][4][m] * v[k][j - 1][i][4] +
                                                 ldx[j][i][4][m] * v[k][j][i - 1][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * diagonal block inversion
             *
             * forward elimination
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                tmat[0][m] = d[j][i][0][m];
                tmat[1][m] = d[j][i][1][m];
                tmat[2][m] = d[j][i][2][m];
                tmat[3][m] = d[j][i][3][m];
                tmat[4][m] = d[j][i][4][m];
            }
            /* */
            tmp1       = 1.0 / tmat[0][0];
            tmp        = tmp1 * tmat[0][1];
            tmat[1][1] = tmat[1][1] - tmp * tmat[1][0];
            tmat[2][1] = tmat[2][1] - tmp * tmat[2][0];
            tmat[3][1] = tmat[3][1] - tmp * tmat[3][0];
            tmat[4][1] = tmat[4][1] - tmp * tmat[4][0];
            tv[1]      = tv[1] - tv[0] * tmp;
            /* */
            tmp        = tmp1 * tmat[0][2];
            tmat[1][2] = tmat[1][2] - tmp * tmat[1][0];
            tmat[2][2] = tmat[2][2] - tmp * tmat[2][0];
            tmat[3][2] = tmat[3][2] - tmp * tmat[3][0];
            tmat[4][2] = tmat[4][2] - tmp * tmat[4][0];
            tv[2]      = tv[2] - tv[0] * tmp;
            /* */
            tmp        = tmp1 * tmat[0][3];
            tmat[1][3] = tmat[1][3] - tmp * tmat[1][0];
            tmat[2][3] = tmat[2][3] - tmp * tmat[2][0];
            tmat[3][3] = tmat[3][3] - tmp * tmat[3][0];
            tmat[4][3] = tmat[4][3] - tmp * tmat[4][0];
            tv[3]      = tv[3] - tv[0] * tmp;
            /* */
            tmp        = tmp1 * tmat[0][4];
            tmat[1][4] = tmat[1][4] - tmp * tmat[1][0];
            tmat[2][4] = tmat[2][4] - tmp * tmat[2][0];
            tmat[3][4] = tmat[3][4] - tmp * tmat[3][0];
            tmat[4][4] = tmat[4][4] - tmp * tmat[4][0];
            tv[4]      = tv[4] - tv[0] * tmp;
            /* */
            tmp1       = 1.0 / tmat[1][1];
            tmp        = tmp1 * tmat[1][2];
            tmat[2][2] = tmat[2][2] - tmp * tmat[2][1];
            tmat[3][2] = tmat[3][2] - tmp * tmat[3][1];
            tmat[4][2] = tmat[4][2] - tmp * tmat[4][1];
            tv[2]      = tv[2] - tv[1] * tmp;
            /* */
            tmp        = tmp1 * tmat[1][3];
            tmat[2][3] = tmat[2][3] - tmp * tmat[2][1];
            tmat[3][3] = tmat[3][3] - tmp * tmat[3][1];
            tmat[4][3] = tmat[4][3] - tmp * tmat[4][1];
            tv[3]      = tv[3] - tv[1] * tmp;
            /* */
            tmp        = tmp1 * tmat[1][4];
            tmat[2][4] = tmat[2][4] - tmp * tmat[2][1];
            tmat[3][4] = tmat[3][4] - tmp * tmat[3][1];
            tmat[4][4] = tmat[4][4] - tmp * tmat[4][1];
            tv[4]      = tv[4] - tv[1] * tmp;
            /* */
            tmp1       = 1.0 / tmat[2][2];
            tmp        = tmp1 * tmat[2][3];
            tmat[3][3] = tmat[3][3] - tmp * tmat[3][2];
            tmat[4][3] = tmat[4][3] - tmp * tmat[4][2];
            tv[3]      = tv[3] - tv[2] * tmp;
            /* */
            tmp        = tmp1 * tmat[2][4];
            tmat[3][4] = tmat[3][4] - tmp * tmat[3][2];
            tmat[4][4] = tmat[4][4] - tmp * tmat[4][2];
            tv[4]      = tv[4] - tv[2] * tmp;
            /* */
            tmp1       = 1.0 / tmat[3][3];
            tmp        = tmp1 * tmat[3][4];
            tmat[4][4] = tmat[4][4] - tmp * tmat[4][3];
            tv[4]      = tv[4] - tv[3] * tmp;
            /*
             * ---------------------------------------------------------------------
             * back substitution
             * ---------------------------------------------------------------------
             */
            v[k][j][i][4] = tv[4] / tmat[4][4];
            tv[3]         = tv[3] - tmat[4][3] * v[k][j][i][4];
            v[k][j][i][3] = tv[3] / tmat[3][3];
            tv[2] = tv[2] - tmat[3][2] * v[k][j][i][3] - tmat[4][2] * v[k][j][i][4];
            v[k][j][i][2] = tv[2] / tmat[2][2];
            tv[1] = tv[1] - tmat[2][1] * v[k][j][i][2] - tmat[3][1] * v[k][j][i][3] -
                    tmat[4][1] * v[k][j][i][4];
            v[k][j][i][1] = tv[1] / tmat[1][1];
            tv[0] = tv[0] - tmat[1][0] * v[k][j][i][1] - tmat[2][0] * v[k][j][i][2] -
                    tmat[3][0] * v[k][j][i][3] - tmat[4][0] * v[k][j][i][4];
            v[k][j][i][0] = tv[0] / tmat[0][0];
        }

        if(j != jend - 1) flag[j] = 1;
        if(j != jst) flag[j - 1] = 0;
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the regular-sparse, block upper triangular solution:
 * v <-- ( U-inv ) * v
 * ---------------------------------------------------------------------
 * to improve cache performance, second two dimensions padded by 1
 * for even number sizes only. only needed in v.
 * ---------------------------------------------------------------------
 */
void
buts(int /*nx*/, int /*ny*/, int /*nz*/, int k, double omega,
     double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5], void* pointer_tv,
     double d[][ISIZ1 / 2 * 2 + 1][5][5], double udx[][ISIZ1 / 2 * 2 + 1][5][5],
     double udy[][ISIZ1 / 2 * 2 + 1][5][5], double udz[][ISIZ1 / 2 * 2 + 1][5][5],
     int ist, int iend, int jst, int jend, int /*nx0*/, int /*ny0*/)
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    double(*tv)[ISIZ1 / 2 * 2 + 1][5] = (double(*)[ISIZ1 / 2 * 2 + 1][5]) pointer_tv;
    int    i, j, m;
    double tmp, tmp1;
    double tmat[5][5];

#pragma omp for nowait schedule(static)
    for(j = jend - 1; j >= jst; j--)
    {
        for(i = iend - 1; i >= ist; i--)
        {
            for(m = 0; m < 5; m++)
            {
                tv[j][i][m] = omega * (udz[j][i][0][m] * v[k + 1][j][i][0] +
                                       udz[j][i][1][m] * v[k + 1][j][i][1] +
                                       udz[j][i][2][m] * v[k + 1][j][i][2] +
                                       udz[j][i][3][m] * v[k + 1][j][i][3] +
                                       udz[j][i][4][m] * v[k + 1][j][i][4]);
            }
        }
    }

#pragma omp for nowait schedule(static)
    for(j = jend - 1; j >= jst; j--)
    {
        if(j != jend - 1)
        {
            while(flag2[j + 1] == 0)
            {
#pragma omp flush
                ;
            }
        }
        if(j != jst)
        {
            while(flag2[j] == 1)
            {
#pragma omp flush
                ;
            }
        }

        for(i = iend - 1; i >= ist; i--)
        {
            for(m = 0; m < 5; m++)
            {
                tv[j][i][m] = tv[j][i][m] + omega * (udy[j][i][0][m] * v[k][j + 1][i][0] +
                                                     udx[j][i][0][m] * v[k][j][i + 1][0] +
                                                     udy[j][i][1][m] * v[k][j + 1][i][1] +
                                                     udx[j][i][1][m] * v[k][j][i + 1][1] +
                                                     udy[j][i][2][m] * v[k][j + 1][i][2] +
                                                     udx[j][i][2][m] * v[k][j][i + 1][2] +
                                                     udy[j][i][3][m] * v[k][j + 1][i][3] +
                                                     udx[j][i][3][m] * v[k][j][i + 1][3] +
                                                     udy[j][i][4][m] * v[k][j + 1][i][4] +
                                                     udx[j][i][4][m] * v[k][j][i + 1][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * diagonal block inversion
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                tmat[0][m] = d[j][i][0][m];
                tmat[1][m] = d[j][i][1][m];
                tmat[2][m] = d[j][i][2][m];
                tmat[3][m] = d[j][i][3][m];
                tmat[4][m] = d[j][i][4][m];
            }
            /* */
            tmp1        = 1.0 / tmat[0][0];
            tmp         = tmp1 * tmat[0][1];
            tmat[1][1]  = tmat[1][1] - tmp * tmat[1][0];
            tmat[2][1]  = tmat[2][1] - tmp * tmat[2][0];
            tmat[3][1]  = tmat[3][1] - tmp * tmat[3][0];
            tmat[4][1]  = tmat[4][1] - tmp * tmat[4][0];
            tv[j][i][1] = tv[j][i][1] - tv[j][i][0] * tmp;
            /* */
            tmp         = tmp1 * tmat[0][2];
            tmat[1][2]  = tmat[1][2] - tmp * tmat[1][0];
            tmat[2][2]  = tmat[2][2] - tmp * tmat[2][0];
            tmat[3][2]  = tmat[3][2] - tmp * tmat[3][0];
            tmat[4][2]  = tmat[4][2] - tmp * tmat[4][0];
            tv[j][i][2] = tv[j][i][2] - tv[j][i][0] * tmp;
            /* */
            tmp         = tmp1 * tmat[0][3];
            tmat[1][3]  = tmat[1][3] - tmp * tmat[1][0];
            tmat[2][3]  = tmat[2][3] - tmp * tmat[2][0];
            tmat[3][3]  = tmat[3][3] - tmp * tmat[3][0];
            tmat[4][3]  = tmat[4][3] - tmp * tmat[4][0];
            tv[j][i][3] = tv[j][i][3] - tv[j][i][0] * tmp;
            /* */
            tmp         = tmp1 * tmat[0][4];
            tmat[1][4]  = tmat[1][4] - tmp * tmat[1][0];
            tmat[2][4]  = tmat[2][4] - tmp * tmat[2][0];
            tmat[3][4]  = tmat[3][4] - tmp * tmat[3][0];
            tmat[4][4]  = tmat[4][4] - tmp * tmat[4][0];
            tv[j][i][4] = tv[j][i][4] - tv[j][i][0] * tmp;
            /* */
            tmp1        = 1.0 / tmat[1][1];
            tmp         = tmp1 * tmat[1][2];
            tmat[2][2]  = tmat[2][2] - tmp * tmat[2][1];
            tmat[3][2]  = tmat[3][2] - tmp * tmat[3][1];
            tmat[4][2]  = tmat[4][2] - tmp * tmat[4][1];
            tv[j][i][2] = tv[j][i][2] - tv[j][i][1] * tmp;
            /* */
            tmp         = tmp1 * tmat[1][3];
            tmat[2][3]  = tmat[2][3] - tmp * tmat[2][1];
            tmat[3][3]  = tmat[3][3] - tmp * tmat[3][1];
            tmat[4][3]  = tmat[4][3] - tmp * tmat[4][1];
            tv[j][i][3] = tv[j][i][3] - tv[j][i][1] * tmp;
            /* */
            tmp         = tmp1 * tmat[1][4];
            tmat[2][4]  = tmat[2][4] - tmp * tmat[2][1];
            tmat[3][4]  = tmat[3][4] - tmp * tmat[3][1];
            tmat[4][4]  = tmat[4][4] - tmp * tmat[4][1];
            tv[j][i][4] = tv[j][i][4] - tv[j][i][1] * tmp;
            /* */
            tmp1        = 1.0 / tmat[2][2];
            tmp         = tmp1 * tmat[2][3];
            tmat[3][3]  = tmat[3][3] - tmp * tmat[3][2];
            tmat[4][3]  = tmat[4][3] - tmp * tmat[4][2];
            tv[j][i][3] = tv[j][i][3] - tv[j][i][2] * tmp;
            /* */
            tmp         = tmp1 * tmat[2][4];
            tmat[3][4]  = tmat[3][4] - tmp * tmat[3][2];
            tmat[4][4]  = tmat[4][4] - tmp * tmat[4][2];
            tv[j][i][4] = tv[j][i][4] - tv[j][i][2] * tmp;
            /* */
            tmp1        = 1.0 / tmat[3][3];
            tmp         = tmp1 * tmat[3][4];
            tmat[4][4]  = tmat[4][4] - tmp * tmat[4][3];
            tv[j][i][4] = tv[j][i][4] - tv[j][i][3] * tmp;
            /*
             * ---------------------------------------------------------------------
             * back substitution
             * ---------------------------------------------------------------------
             */
            tv[j][i][4] = tv[j][i][4] / tmat[4][4];
            tv[j][i][3] = tv[j][i][3] - tmat[4][3] * tv[j][i][4];
            tv[j][i][3] = tv[j][i][3] / tmat[3][3];
            tv[j][i][2] =
                tv[j][i][2] - tmat[3][2] * tv[j][i][3] - tmat[4][2] * tv[j][i][4];
            tv[j][i][2] = tv[j][i][2] / tmat[2][2];
            tv[j][i][1] = tv[j][i][1] - tmat[2][1] * tv[j][i][2] -
                          tmat[3][1] * tv[j][i][3] - tmat[4][1] * tv[j][i][4];
            tv[j][i][1] = tv[j][i][1] / tmat[1][1];
            tv[j][i][0] = tv[j][i][0] - tmat[1][0] * tv[j][i][1] -
                          tmat[2][0] * tv[j][i][2] - tmat[3][0] * tv[j][i][3] -
                          tmat[4][0] * tv[j][i][4];
            tv[j][i][0]   = tv[j][i][0] / tmat[0][0];
            v[k][j][i][0] = v[k][j][i][0] - tv[j][i][0];
            v[k][j][i][1] = v[k][j][i][1] - tv[j][i][1];
            v[k][j][i][2] = v[k][j][i][2] - tv[j][i][2];
            v[k][j][i][3] = v[k][j][i][3] - tv[j][i][3];
            v[k][j][i][4] = v[k][j][i][4] - tv[j][i][4];
        }

        if(j != jend - 1) flag2[j + 1] = 0;
        if(j != jst) flag2[j] = 1;
    }
}

void
domain()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    nx = nx0;
    ny = ny0;
    nz = nz0;
    /*
     * ---------------------------------------------------------------------
     * check the sub-domain size
     * ---------------------------------------------------------------------
     */
    if((nx < 4) || (ny < 4) || (nz < 4))
    {
        printf("     SUBDOMAIN SIZE IS TOO SMALL - \n"
               "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
               "     SO THAT NX, NY AND NZ ARE GREATER THAN OR EQUAL\n"
               "     TO 4 THEY ARE CURRENTLY%3d%3d%3d\n",
               nx, ny, nz);
        exit(EXIT_FAILURE);
    }
    if((nx > ISIZ1) || (ny > ISIZ2) || (nz > ISIZ3))
    {
        printf("     SUBDOMAIN SIZE IS TOO LARGE - \n"
               "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
               "     SO THAT NX, NY AND NZ ARE LESS THAN OR EQUAL TO \n"
               "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY.  THEY ARE\n"
               "     CURRENTLYi%4d%4d%4d\n",
               nx, ny, nz);
        exit(EXIT_FAILURE);
    }
    /*
     * ---------------------------------------------------------------------
     * set up the start and end in i and j extents for all processors
     * ---------------------------------------------------------------------
     */
    ist  = 1;
    iend = nx - 1;
    jst  = 1;
    jend = ny - 1;
    ii1  = 1;
    ii2  = nx0 - 1;
    ji1  = 1;
    ji2  = ny0 - 2;
    ki1  = 2;
    ki2  = nz0 - 1;
}

/*
 * ---------------------------------------------------------------------
 * compute the right hand side based on exact solution
 * ---------------------------------------------------------------------
 */
void
erhs()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double xi, eta, zeta;
    double q;
    double u21, u31, u41;
    double tmp;
    double u21i, u31i, u41i, u51i;
    double u21j, u31j, u41j, u51j;
    double u21k, u31k, u41k, u51k;
    double u21im1, u31im1, u41im1, u51im1;
    double u21jm1, u31jm1, u41jm1, u51jm1;
    double u21km1, u31km1, u41km1, u51km1;
    double flux[ISIZ1][5];

#pragma omp for
    for(k = 0; k < nz; k++)
    {
        for(j = 0; j < ny; j++)
        {
            for(i = 0; i < nx; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] = 0.0;
                }
            }
        }
    }

#pragma omp for
    for(k = 0; k < nz; k++)
    {
        zeta = ((double) k) / (nz - 1);
        for(j = 0; j < ny; j++)
        {
            eta = ((double) j) / (ny0 - 1);
            for(i = 0; i < nx; i++)
            {
                xi = ((double) i) / (nx0 - 1);
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        ce[0][m] +
                        (ce[1][m] + (ce[4][m] + (ce[7][m] + ce[10][m] * xi) * xi) * xi) *
                            xi +
                        (ce[2][m] +
                         (ce[5][m] + (ce[8][m] + ce[11][m] * eta) * eta) * eta) *
                            eta +
                        (ce[3][m] +
                         (ce[6][m] + (ce[9][m] + ce[12][m] * zeta) * zeta) * zeta) *
                            zeta;
                }
            }
        }
    }
/*
 * ---------------------------------------------------------------------
 * xi-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 1; k < nz - 1; k++)
    {
        for(j = jst; j < jend; j++)
        {
            for(i = 0; i < nx; i++)
            {
                flux[i][0] = rsd[k][j][i][1];
                u21        = rsd[k][j][i][1] / rsd[k][j][i][0];
                q          = 0.50 *
                    (rsd[k][j][i][1] * rsd[k][j][i][1] +
                     rsd[k][j][i][2] * rsd[k][j][i][2] +
                     rsd[k][j][i][3] * rsd[k][j][i][3]) /
                    rsd[k][j][i][0];
                flux[i][1] = rsd[k][j][i][1] * u21 + C2 * (rsd[k][j][i][4] - q);
                flux[i][2] = rsd[k][j][i][2] * u21;
                flux[i][3] = rsd[k][j][i][3] * u21;
                flux[i][4] = (C1 * rsd[k][j][i][4] - C2 * q) * u21;
            }
            for(i = ist; i < iend; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] - tx2 * (flux[i + 1][m] - flux[i - 1][m]);
                }
            }
            for(i = ist; i < nx; i++)
            {
                tmp        = 1.0 / rsd[k][j][i][0];
                u21i       = tmp * rsd[k][j][i][1];
                u31i       = tmp * rsd[k][j][i][2];
                u41i       = tmp * rsd[k][j][i][3];
                u51i       = tmp * rsd[k][j][i][4];
                tmp        = 1.0 / rsd[k][j][i - 1][0];
                u21im1     = tmp * rsd[k][j][i - 1][1];
                u31im1     = tmp * rsd[k][j][i - 1][2];
                u41im1     = tmp * rsd[k][j][i - 1][3];
                u51im1     = tmp * rsd[k][j][i - 1][4];
                flux[i][1] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
                flux[i][2] = tx3 * (u31i - u31im1);
                flux[i][3] = tx3 * (u41i - u41im1);
                flux[i][4] = 0.50 * (1.0 - C1 * C5) * tx3 *
                                 ((u21i * u21i + u31i * u31i + u41i * u41i) -
                                  (u21im1 * u21im1 + u31im1 * u31im1 + u41im1 * u41im1)) +
                             (1.0 / 6.0) * tx3 * (u21i * u21i - u21im1 * u21im1) +
                             C1 * C5 * tx3 * (u51i - u51im1);
            }
            for(i = ist; i < iend; i++)
            {
                frct[k][j][i][0] =
                    frct[k][j][i][0] + dx1 * tx1 *
                                           (rsd[k][j][i - 1][0] - 2.0 * rsd[k][j][i][0] +
                                            rsd[k][j][i + 1][0]);
                frct[k][j][i][1] = frct[k][j][i][1] +
                                   tx3 * C3 * C4 * (flux[i + 1][1] - flux[i][1]) +
                                   dx2 * tx1 *
                                       (rsd[k][j][i - 1][1] - 2.0 * rsd[k][j][i][1] +
                                        rsd[k][j][i + 1][1]);
                frct[k][j][i][2] = frct[k][j][i][2] +
                                   tx3 * C3 * C4 * (flux[i + 1][2] - flux[i][2]) +
                                   dx3 * tx1 *
                                       (rsd[k][j][i - 1][2] - 2.0 * rsd[k][j][i][2] +
                                        rsd[k][j][i + 1][2]);
                frct[k][j][i][3] = frct[k][j][i][3] +
                                   tx3 * C3 * C4 * (flux[i + 1][3] - flux[i][3]) +
                                   dx4 * tx1 *
                                       (rsd[k][j][i - 1][3] - 2.0 * rsd[k][j][i][3] +
                                        rsd[k][j][i + 1][3]);
                frct[k][j][i][4] = frct[k][j][i][4] +
                                   tx3 * C3 * C4 * (flux[i + 1][4] - flux[i][4]) +
                                   dx5 * tx1 *
                                       (rsd[k][j][i - 1][4] - 2.0 * rsd[k][j][i][4] +
                                        rsd[k][j][i + 1][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                frct[k][j][1][m] =
                    frct[k][j][1][m] - dssp * (+5.0 * rsd[k][j][1][m] -
                                               4.0 * rsd[k][j][2][m] + rsd[k][j][3][m]);
                frct[k][j][2][m] =
                    frct[k][j][2][m] -
                    dssp * (-4.0 * rsd[k][j][1][m] + 6.0 * rsd[k][j][2][m] -
                            4.0 * rsd[k][j][3][m] + rsd[k][j][4][m]);
            }
            for(i = 3; i < nx - 3; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] -
                        dssp * (rsd[k][j][i - 2][m] - 4.0 * rsd[k][j][i - 1][m] +
                                6.0 * rsd[k][j][i][m] - 4.0 * rsd[k][j][i + 1][m] +
                                rsd[k][j][i + 2][m]);
                }
            }
            for(m = 0; m < 5; m++)
            {
                frct[k][j][nx - 3][m] =
                    frct[k][j][nx - 3][m] -
                    dssp * (rsd[k][j][nx - 5][m] - 4.0 * rsd[k][j][nx - 4][m] +
                            6.0 * rsd[k][j][nx - 3][m] - 4.0 * rsd[k][j][nx - 2][m]);
                frct[k][j][nx - 2][m] =
                    frct[k][j][nx - 2][m] -
                    dssp * (rsd[k][j][nx - 4][m] - 4.0 * rsd[k][j][nx - 3][m] +
                            5.0 * rsd[k][j][nx - 2][m]);
            }
        }
    }
/*
 * ---------------------------------------------------------------------
 * eta-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 1; k < nz - 1; k++)
    {
        for(i = ist; i < iend; i++)
        {
            for(j = 0; j < ny; j++)
            {
                flux[j][0] = rsd[k][j][i][2];
                u31        = rsd[k][j][i][2] / rsd[k][j][i][0];
                q          = 0.50 *
                    (rsd[k][j][i][1] * rsd[k][j][i][1] +
                     rsd[k][j][i][2] * rsd[k][j][i][2] +
                     rsd[k][j][i][3] * rsd[k][j][i][3]) /
                    rsd[k][j][i][0];
                flux[j][1] = rsd[k][j][i][1] * u31;
                flux[j][2] = rsd[k][j][i][2] * u31 + C2 * (rsd[k][j][i][4] - q);
                flux[j][3] = rsd[k][j][i][3] * u31;
                flux[j][4] = (C1 * rsd[k][j][i][4] - C2 * q) * u31;
            }
            for(j = jst; j < jend; j++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] - ty2 * (flux[j + 1][m] - flux[j - 1][m]);
                }
            }
            for(j = jst; j < ny; j++)
            {
                tmp        = 1.0 / rsd[k][j][i][0];
                u21j       = tmp * rsd[k][j][i][1];
                u31j       = tmp * rsd[k][j][i][2];
                u41j       = tmp * rsd[k][j][i][3];
                u51j       = tmp * rsd[k][j][i][4];
                tmp        = 1.0 / rsd[k][j - 1][i][0];
                u21jm1     = tmp * rsd[k][j - 1][i][1];
                u31jm1     = tmp * rsd[k][j - 1][i][2];
                u41jm1     = tmp * rsd[k][j - 1][i][3];
                u51jm1     = tmp * rsd[k][j - 1][i][4];
                flux[j][1] = ty3 * (u21j - u21jm1);
                flux[j][2] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
                flux[j][3] = ty3 * (u41j - u41jm1);
                flux[j][4] = 0.50 * (1.0 - C1 * C5) * ty3 *
                                 ((u21j * u21j + u31j * u31j + u41j * u41j) -
                                  (u21jm1 * u21jm1 + u31jm1 * u31jm1 + u41jm1 * u41jm1)) +
                             (1.0 / 6.0) * ty3 * (u31j * u31j - u31jm1 * u31jm1) +
                             C1 * C5 * ty3 * (u51j - u51jm1);
            }
            for(j = jst; j < jend; j++)
            {
                frct[k][j][i][0] =
                    frct[k][j][i][0] + dy1 * ty1 *
                                           (rsd[k][j - 1][i][0] - 2.0 * rsd[k][j][i][0] +
                                            rsd[k][j + 1][i][0]);
                frct[k][j][i][1] = frct[k][j][i][1] +
                                   ty3 * C3 * C4 * (flux[j + 1][1] - flux[j][1]) +
                                   dy2 * ty1 *
                                       (rsd[k][j - 1][i][1] - 2.0 * rsd[k][j][i][1] +
                                        rsd[k][j + 1][i][1]);
                frct[k][j][i][2] = frct[k][j][i][2] +
                                   ty3 * C3 * C4 * (flux[j + 1][2] - flux[j][2]) +
                                   dy3 * ty1 *
                                       (rsd[k][j - 1][i][2] - 2.0 * rsd[k][j][i][2] +
                                        rsd[k][j + 1][i][2]);
                frct[k][j][i][3] = frct[k][j][i][3] +
                                   ty3 * C3 * C4 * (flux[j + 1][3] - flux[j][3]) +
                                   dy4 * ty1 *
                                       (rsd[k][j - 1][i][3] - 2.0 * rsd[k][j][i][3] +
                                        rsd[k][j + 1][i][3]);
                frct[k][j][i][4] = frct[k][j][i][4] +
                                   ty3 * C3 * C4 * (flux[j + 1][4] - flux[j][4]) +
                                   dy5 * ty1 *
                                       (rsd[k][j - 1][i][4] - 2.0 * rsd[k][j][i][4] +
                                        rsd[k][j + 1][i][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                frct[k][1][i][m] =
                    frct[k][1][i][m] - dssp * (+5.0 * rsd[k][1][i][m] -
                                               4.0 * rsd[k][2][i][m] + rsd[k][3][i][m]);
                frct[k][2][i][m] =
                    frct[k][2][i][m] -
                    dssp * (-4.0 * rsd[k][1][i][m] + 6.0 * rsd[k][2][i][m] -
                            4.0 * rsd[k][3][i][m] + rsd[k][4][i][m]);
            }
            for(j = 3; j < ny - 3; j++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] -
                        dssp * (rsd[k][j - 2][i][m] - 4.0 * rsd[k][j - 1][i][m] +
                                6.0 * rsd[k][j][i][m] - 4.0 * rsd[k][j + 1][i][m] +
                                rsd[k][j + 2][i][m]);
                }
            }
            for(m = 0; m < 5; m++)
            {
                frct[k][ny - 3][i][m] =
                    frct[k][ny - 3][i][m] -
                    dssp * (rsd[k][ny - 5][i][m] - 4.0 * rsd[k][ny - 4][i][m] +
                            6.0 * rsd[k][ny - 3][i][m] - 4.0 * rsd[k][ny - 2][i][m]);
                frct[k][ny - 2][i][m] =
                    frct[k][ny - 2][i][m] -
                    dssp * (rsd[k][ny - 4][i][m] - 4.0 * rsd[k][ny - 3][i][m] +
                            5.0 * rsd[k][ny - 2][i][m]);
            }
        }
    }
/*
 * ---------------------------------------------------------------------
 * zeta-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(j = jst; j < jend; j++)
    {
        for(i = ist; i < iend; i++)
        {
            for(k = 0; k < nz; k++)
            {
                flux[k][0] = rsd[k][j][i][3];
                u41        = rsd[k][j][i][3] / rsd[k][j][i][0];
                q          = 0.50 *
                    (rsd[k][j][i][1] * rsd[k][j][i][1] +
                     rsd[k][j][i][2] * rsd[k][j][i][2] +
                     rsd[k][j][i][3] * rsd[k][j][i][3]) /
                    rsd[k][j][i][0];
                flux[k][1] = rsd[k][j][i][1] * u41;
                flux[k][2] = rsd[k][j][i][2] * u41;
                flux[k][3] = rsd[k][j][i][3] * u41 + C2 * (rsd[k][j][i][4] - q);
                flux[k][4] = (C1 * rsd[k][j][i][4] - C2 * q) * u41;
            }
            for(k = 1; k < nz - 1; k++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] - tz2 * (flux[k + 1][m] - flux[k - 1][m]);
                }
            }
            for(k = 1; k < nz; k++)
            {
                tmp        = 1.0 / rsd[k][j][i][0];
                u21k       = tmp * rsd[k][j][i][1];
                u31k       = tmp * rsd[k][j][i][2];
                u41k       = tmp * rsd[k][j][i][3];
                u51k       = tmp * rsd[k][j][i][4];
                tmp        = 1.0 / rsd[k - 1][j][i][0];
                u21km1     = tmp * rsd[k - 1][j][i][1];
                u31km1     = tmp * rsd[k - 1][j][i][2];
                u41km1     = tmp * rsd[k - 1][j][i][3];
                u51km1     = tmp * rsd[k - 1][j][i][4];
                flux[k][1] = tz3 * (u21k - u21km1);
                flux[k][2] = tz3 * (u31k - u31km1);
                flux[k][3] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
                flux[k][4] = 0.50 * (1.0 - C1 * C5) * tz3 *
                                 ((u21k * u21k + u31k * u31k + u41k * u41k) -
                                  (u21km1 * u21km1 + u31km1 * u31km1 + u41km1 * u41km1)) +
                             (1.0 / 6.0) * tz3 * (u41k * u41k - u41km1 * u41km1) +
                             C1 * C5 * tz3 * (u51k - u51km1);
            }
            for(k = 1; k < nz - 1; k++)
            {
                frct[k][j][i][0] =
                    frct[k][j][i][0] + dz1 * tz1 *
                                           (rsd[k + 1][j][i][0] - 2.0 * rsd[k][j][i][0] +
                                            rsd[k - 1][j][i][0]);
                frct[k][j][i][1] = frct[k][j][i][1] +
                                   tz3 * C3 * C4 * (flux[k + 1][1] - flux[k][1]) +
                                   dz2 * tz1 *
                                       (rsd[k + 1][j][i][1] - 2.0 * rsd[k][j][i][1] +
                                        rsd[k - 1][j][i][1]);
                frct[k][j][i][2] = frct[k][j][i][2] +
                                   tz3 * C3 * C4 * (flux[k + 1][2] - flux[k][2]) +
                                   dz3 * tz1 *
                                       (rsd[k + 1][j][i][2] - 2.0 * rsd[k][j][i][2] +
                                        rsd[k - 1][j][i][2]);
                frct[k][j][i][3] = frct[k][j][i][3] +
                                   tz3 * C3 * C4 * (flux[k + 1][3] - flux[k][3]) +
                                   dz4 * tz1 *
                                       (rsd[k + 1][j][i][3] - 2.0 * rsd[k][j][i][3] +
                                        rsd[k - 1][j][i][3]);
                frct[k][j][i][4] = frct[k][j][i][4] +
                                   tz3 * C3 * C4 * (flux[k + 1][4] - flux[k][4]) +
                                   dz5 * tz1 *
                                       (rsd[k + 1][j][i][4] - 2.0 * rsd[k][j][i][4] +
                                        rsd[k - 1][j][i][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                frct[1][j][i][m] =
                    frct[1][j][i][m] - dssp * (+5.0 * rsd[1][j][i][m] -
                                               4.0 * rsd[2][j][i][m] + rsd[3][j][i][m]);
                frct[2][j][i][m] =
                    frct[2][j][i][m] -
                    dssp * (-4.0 * rsd[1][j][i][m] + 6.0 * rsd[2][j][i][m] -
                            4.0 * rsd[3][j][i][m] + rsd[4][j][i][m]);
            }
            for(k = 3; k < nz - 3; k++)
            {
                for(m = 0; m < 5; m++)
                {
                    frct[k][j][i][m] =
                        frct[k][j][i][m] -
                        dssp * (rsd[k - 2][j][i][m] - 4.0 * rsd[k - 1][j][i][m] +
                                6.0 * rsd[k][j][i][m] - 4.0 * rsd[k + 1][j][i][m] +
                                rsd[k + 2][j][i][m]);
                }
            }
            for(m = 0; m < 5; m++)
            {
                frct[nz - 3][j][i][m] =
                    frct[nz - 3][j][i][m] -
                    dssp * (rsd[nz - 5][j][i][m] - 4.0 * rsd[nz - 4][j][i][m] +
                            6.0 * rsd[nz - 3][j][i][m] - 4.0 * rsd[nz - 2][j][i][m]);
                frct[nz - 2][j][i][m] =
                    frct[nz - 2][j][i][m] -
                    dssp * (rsd[nz - 4][j][i][m] - 4.0 * rsd[nz - 3][j][i][m] +
                            5.0 * rsd[nz - 2][j][i][m]);
            }
        }
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the solution error
 * ---------------------------------------------------------------------
 */
void
error()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double tmp;
    double u000ijk[5];
    for(m = 0; m < 5; m++)
    {
        errnm[m] = 0.0;
    }
    for(k = 1; k < nz - 1; k++)
    {
        for(j = jst; j < jend; j++)
        {
            for(i = ist; i < iend; i++)
            {
                exact(i, j, k, u000ijk);
                for(m = 0; m < 5; m++)
                {
                    tmp      = (u000ijk[m] - u[k][j][i][m]);
                    errnm[m] = errnm[m] + tmp * tmp;
                }
            }
        }
    }
    for(m = 0; m < 5; m++)
    {
        errnm[m] = sqrt(errnm[m] / ((nx0 - 2) * (ny0 - 2) * (nz0 - 2)));
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the exact solution at (i,j,k)
 * ---------------------------------------------------------------------
 */
void
exact(int i, int j, int k, double u000ijk[])
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    m;
    double xi, eta, zeta;
    xi   = ((double) i) / (nx0 - 1);
    eta  = ((double) j) / (ny0 - 1);
    zeta = ((double) k) / (nz - 1);
    for(m = 0; m < 5; m++)
    {
        u000ijk[m] =
            ce[0][m] +
            (ce[1][m] + (ce[4][m] + (ce[7][m] + ce[10][m] * xi) * xi) * xi) * xi +
            (ce[2][m] + (ce[5][m] + (ce[8][m] + ce[11][m] * eta) * eta) * eta) * eta +
            (ce[3][m] + (ce[6][m] + (ce[9][m] + ce[12][m] * zeta) * zeta) * zeta) * zeta;
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the lower triangular part of the jacobian matrix
 * ---------------------------------------------------------------------
 */
void
jacld(int k)
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j;
    double r43;
    double c1345;
    double c34;
    double tmp1, tmp2, tmp3;
    r43   = (4.0 / 3.0);
    c1345 = C1 * C3 * C4 * C5;
    c34   = C3 * C4;

#pragma omp for nowait schedule(static)
    for(j = jst; j < jend; j++)
    {
        for(i = ist; i < iend; i++)
        {
            /*
             * ---------------------------------------------------------------------
             * form the block daigonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            d[j][i][0][0] = 1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
            d[j][i][1][0] = 0.0;
            d[j][i][2][0] = 0.0;
            d[j][i][3][0] = 0.0;
            d[j][i][4][0] = 0.0;
            d[j][i][0][1] =
                -dt * 2.0 * (tx1 * r43 + ty1 + tz1) * c34 * tmp2 * u[k][j][i][1];
            d[j][i][1][1] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 * r43 + ty1 + tz1) +
                            dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
            d[j][i][2][1] = 0.0;
            d[j][i][3][1] = 0.0;
            d[j][i][4][1] = 0.0;
            d[j][i][0][2] =
                -dt * 2.0 * (tx1 + ty1 * r43 + tz1) * c34 * tmp2 * u[k][j][i][2];
            d[j][i][1][2] = 0.0;
            d[j][i][2][2] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 * r43 + tz1) +
                            dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
            d[j][i][3][2] = 0.0;
            d[j][i][4][2] = 0.0;
            d[j][i][0][3] =
                -dt * 2.0 * (tx1 + ty1 + tz1 * r43) * c34 * tmp2 * u[k][j][i][3];
            d[j][i][1][3] = 0.0;
            d[j][i][2][3] = 0.0;
            d[j][i][3][3] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 + tz1 * r43) +
                            dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
            d[j][i][4][3] = 0.0;
            d[j][i][0][4] = -dt * 2.0 *
                            (((tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) +
                               tz1 * (c34 - c1345)) *
                                  (u[k][j][i][1] * u[k][j][i][1]) +
                              (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) +
                               tz1 * (c34 - c1345)) *
                                  (u[k][j][i][2] * u[k][j][i][2]) +
                              (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) +
                               tz1 * (r43 * c34 - c1345)) *
                                  (u[k][j][i][3] * u[k][j][i][3])) *
                                 tmp3 +
                             (tx1 + ty1 + tz1) * c1345 * tmp2 * u[k][j][i][4]);
            d[j][i][1][4] =
                dt * 2.0 * tmp2 * u[k][j][i][1] *
                (tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (c34 - c1345));
            d[j][i][2][4] =
                dt * 2.0 * tmp2 * u[k][j][i][2] *
                (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) + tz1 * (c34 - c1345));
            d[j][i][3][4] =
                dt * 2.0 * tmp2 * u[k][j][i][3] *
                (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (r43 * c34 - c1345));
            d[j][i][4][4] = 1.0 + dt * 2.0 * (tx1 + ty1 + tz1) * c1345 * tmp1 +
                            dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
            /*
             * ---------------------------------------------------------------------
             * form the first block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k - 1][j][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            a[j][i][0][0] = -dt * tz1 * dz1;
            a[j][i][1][0] = 0.0;
            a[j][i][2][0] = 0.0;
            a[j][i][3][0] = -dt * tz2;
            a[j][i][4][0] = 0.0;
            a[j][i][0][1] =
                -dt * tz2 * (-(u[k - 1][j][i][1] * u[k - 1][j][i][3]) * tmp2) -
                dt * tz1 * (-c34 * tmp2 * u[k - 1][j][i][1]);
            a[j][i][1][1] = -dt * tz2 * (u[k - 1][j][i][3] * tmp1) -
                            dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
            a[j][i][2][1] = 0.0;
            a[j][i][3][1] = -dt * tz2 * (u[k - 1][j][i][1] * tmp1);
            a[j][i][4][1] = 0.0;
            a[j][i][0][2] =
                -dt * tz2 * (-(u[k - 1][j][i][2] * u[k - 1][j][i][3]) * tmp2) -
                dt * tz1 * (-c34 * tmp2 * u[k - 1][j][i][2]);
            a[j][i][1][2] = 0.0;
            a[j][i][2][2] = -dt * tz2 * (u[k - 1][j][i][3] * tmp1) -
                            dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
            a[j][i][3][2] = -dt * tz2 * (u[k - 1][j][i][2] * tmp1);
            a[j][i][4][2] = 0.0;
            a[j][i][0][3] =
                -dt * tz2 *
                    (-(u[k - 1][j][i][3] * tmp1) * (u[k - 1][j][i][3] * tmp1) +
                     C2 * qs[k - 1][j][i] * tmp1) -
                dt * tz1 * (-r43 * c34 * tmp2 * u[k - 1][j][i][3]);
            a[j][i][1][3] = -dt * tz2 * (-C2 * (u[k - 1][j][i][1] * tmp1));
            a[j][i][2][3] = -dt * tz2 * (-C2 * (u[k - 1][j][i][2] * tmp1));
            a[j][i][3][3] = -dt * tz2 * (2.0 - C2) * (u[k - 1][j][i][3] * tmp1) -
                            dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
            a[j][i][4][3] = -dt * tz2 * C2;
            a[j][i][0][4] =
                -dt * tz2 *
                    ((C2 * 2.0 * qs[k - 1][j][i] - C1 * u[k - 1][j][i][4]) *
                     u[k - 1][j][i][3] * tmp2) -
                dt * tz1 *
                    (-(c34 - c1345) * tmp3 * (u[k - 1][j][i][1] * u[k - 1][j][i][1]) -
                     (c34 - c1345) * tmp3 * (u[k - 1][j][i][2] * u[k - 1][j][i][2]) -
                     (r43 * c34 - c1345) * tmp3 *
                         (u[k - 1][j][i][3] * u[k - 1][j][i][3]) -
                     c1345 * tmp2 * u[k - 1][j][i][4]);
            a[j][i][1][4] =
                -dt * tz2 * (-C2 * (u[k - 1][j][i][1] * u[k - 1][j][i][3]) * tmp2) -
                dt * tz1 * (c34 - c1345) * tmp2 * u[k - 1][j][i][1];
            a[j][i][2][4] =
                -dt * tz2 * (-C2 * (u[k - 1][j][i][2] * u[k - 1][j][i][3]) * tmp2) -
                dt * tz1 * (c34 - c1345) * tmp2 * u[k - 1][j][i][2];
            a[j][i][3][4] = -dt * tz2 *
                                (C1 * (u[k - 1][j][i][4] * tmp1) -
                                 C2 * (qs[k - 1][j][i] * tmp1 +
                                       u[k - 1][j][i][3] * u[k - 1][j][i][3] * tmp2)) -
                            dt * tz1 * (r43 * c34 - c1345) * tmp2 * u[k - 1][j][i][3];
            a[j][i][4][4] = -dt * tz2 * (C1 * (u[k - 1][j][i][3] * tmp1)) -
                            dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
            /*
             * ---------------------------------------------------------------------
             * form the second block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j - 1][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            b[j][i][0][0] = -dt * ty1 * dy1;
            b[j][i][1][0] = 0.0;
            b[j][i][2][0] = -dt * ty2;
            b[j][i][3][0] = 0.0;
            b[j][i][4][0] = 0.0;
            b[j][i][0][1] =
                -dt * ty2 * (-(u[k][j - 1][i][1] * u[k][j - 1][i][2]) * tmp2) -
                dt * ty1 * (-c34 * tmp2 * u[k][j - 1][i][1]);
            b[j][i][1][1] = -dt * ty2 * (u[k][j - 1][i][2] * tmp1) -
                            dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
            b[j][i][2][1] = -dt * ty2 * (u[k][j - 1][i][1] * tmp1);
            b[j][i][3][1] = 0.0;
            b[j][i][4][1] = 0.0;
            b[j][i][0][2] =
                -dt * ty2 *
                    (-(u[k][j - 1][i][2] * tmp1) * (u[k][j - 1][i][2] * tmp1) +
                     C2 * (qs[k][j - 1][i] * tmp1)) -
                dt * ty1 * (-r43 * c34 * tmp2 * u[k][j - 1][i][2]);
            b[j][i][1][2] = -dt * ty2 * (-C2 * (u[k][j - 1][i][1] * tmp1));
            b[j][i][2][2] = -dt * ty2 * ((2.0 - C2) * (u[k][j - 1][i][2] * tmp1)) -
                            dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
            b[j][i][3][2] = -dt * ty2 * (-C2 * (u[k][j - 1][i][3] * tmp1));
            b[j][i][4][2] = -dt * ty2 * C2;
            b[j][i][0][3] =
                -dt * ty2 * (-(u[k][j - 1][i][2] * u[k][j - 1][i][3]) * tmp2) -
                dt * ty1 * (-c34 * tmp2 * u[k][j - 1][i][3]);
            b[j][i][1][3] = 0.0;
            b[j][i][2][3] = -dt * ty2 * (u[k][j - 1][i][3] * tmp1);
            b[j][i][3][3] = -dt * ty2 * (u[k][j - 1][i][2] * tmp1) -
                            dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
            b[j][i][4][3] = 0.0;
            b[j][i][0][4] =
                -dt * ty2 *
                    ((C2 * 2.0 * qs[k][j - 1][i] - C1 * u[k][j - 1][i][4]) *
                     (u[k][j - 1][i][2] * tmp2)) -
                dt * ty1 *
                    (-(c34 - c1345) * tmp3 * (u[k][j - 1][i][1] * u[k][j - 1][i][1]) -
                     (r43 * c34 - c1345) * tmp3 *
                         (u[k][j - 1][i][2] * u[k][j - 1][i][2]) -
                     (c34 - c1345) * tmp3 * (u[k][j - 1][i][3] * u[k][j - 1][i][3]) -
                     c1345 * tmp2 * u[k][j - 1][i][4]);
            b[j][i][1][4] =
                -dt * ty2 * (-C2 * (u[k][j - 1][i][1] * u[k][j - 1][i][2]) * tmp2) -
                dt * ty1 * (c34 - c1345) * tmp2 * u[k][j - 1][i][1];
            b[j][i][2][4] = -dt * ty2 *
                                (C1 * (u[k][j - 1][i][4] * tmp1) -
                                 C2 * (qs[k][j - 1][i] * tmp1 +
                                       u[k][j - 1][i][2] * u[k][j - 1][i][2] * tmp2)) -
                            dt * ty1 * (r43 * c34 - c1345) * tmp2 * u[k][j - 1][i][2];
            b[j][i][3][4] =
                -dt * ty2 * (-C2 * (u[k][j - 1][i][2] * u[k][j - 1][i][3]) * tmp2) -
                dt * ty1 * (c34 - c1345) * tmp2 * u[k][j - 1][i][3];
            b[j][i][4][4] = -dt * ty2 * (C1 * (u[k][j - 1][i][2] * tmp1)) -
                            dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
            /*
             * ---------------------------------------------------------------------
             * form the third block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j][i - 1];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            c[j][i][0][0] = -dt * tx1 * dx1;
            c[j][i][1][0] = -dt * tx2;
            c[j][i][2][0] = 0.0;
            c[j][i][3][0] = 0.0;
            c[j][i][4][0] = 0.0;
            c[j][i][0][1] =
                -dt * tx2 *
                    (-(u[k][j][i - 1][1] * tmp1) * (u[k][j][i - 1][1] * tmp1) +
                     C2 * qs[k][j][i - 1] * tmp1) -
                dt * tx1 * (-r43 * c34 * tmp2 * u[k][j][i - 1][1]);
            c[j][i][1][1] = -dt * tx2 * ((2.0 - C2) * (u[k][j][i - 1][1] * tmp1)) -
                            dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
            c[j][i][2][1] = -dt * tx2 * (-C2 * (u[k][j][i - 1][2] * tmp1));
            c[j][i][3][1] = -dt * tx2 * (-C2 * (u[k][j][i - 1][3] * tmp1));
            c[j][i][4][1] = -dt * tx2 * C2;
            c[j][i][0][2] =
                -dt * tx2 * (-(u[k][j][i - 1][1] * u[k][j][i - 1][2]) * tmp2) -
                dt * tx1 * (-c34 * tmp2 * u[k][j][i - 1][2]);
            c[j][i][1][2] = -dt * tx2 * (u[k][j][i - 1][2] * tmp1);
            c[j][i][2][2] = -dt * tx2 * (u[k][j][i - 1][1] * tmp1) -
                            dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
            c[j][i][3][2] = 0.0;
            c[j][i][4][2] = 0.0;
            c[j][i][0][3] =
                -dt * tx2 * (-(u[k][j][i - 1][1] * u[k][j][i - 1][3]) * tmp2) -
                dt * tx1 * (-c34 * tmp2 * u[k][j][i - 1][3]);
            c[j][i][1][3] = -dt * tx2 * (u[k][j][i - 1][3] * tmp1);
            c[j][i][2][3] = 0.0;
            c[j][i][3][3] = -dt * tx2 * (u[k][j][i - 1][1] * tmp1) -
                            dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
            c[j][i][4][3] = 0.0;
            c[j][i][0][4] =
                -dt * tx2 *
                    ((C2 * 2.0 * qs[k][j][i - 1] - C1 * u[k][j][i - 1][4]) *
                     u[k][j][i - 1][1] * tmp2) -
                dt * tx1 *
                    (-(r43 * c34 - c1345) * tmp3 *
                         (u[k][j][i - 1][1] * u[k][j][i - 1][1]) -
                     (c34 - c1345) * tmp3 * (u[k][j][i - 1][2] * u[k][j][i - 1][2]) -
                     (c34 - c1345) * tmp3 * (u[k][j][i - 1][3] * u[k][j][i - 1][3]) -
                     c1345 * tmp2 * u[k][j][i - 1][4]);
            c[j][i][1][4] = -dt * tx2 *
                                (C1 * (u[k][j][i - 1][4] * tmp1) -
                                 C2 * (u[k][j][i - 1][1] * u[k][j][i - 1][1] * tmp2 +
                                       qs[k][j][i - 1] * tmp1)) -
                            dt * tx1 * (r43 * c34 - c1345) * tmp2 * u[k][j][i - 1][1];
            c[j][i][2][4] =
                -dt * tx2 * (-C2 * (u[k][j][i - 1][2] * u[k][j][i - 1][1]) * tmp2) -
                dt * tx1 * (c34 - c1345) * tmp2 * u[k][j][i - 1][2];
            c[j][i][3][4] =
                -dt * tx2 * (-C2 * (u[k][j][i - 1][3] * u[k][j][i - 1][1]) * tmp2) -
                dt * tx1 * (c34 - c1345) * tmp2 * u[k][j][i - 1][3];
            c[j][i][4][4] = -dt * tx2 * (C1 * (u[k][j][i - 1][1] * tmp1)) -
                            dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
        }
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the upper triangular part of the jacobian matrix
 * ---------------------------------------------------------------------
 */
void
jacu(int k)
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j;
    double r43;
    double c1345;
    double c34;
    double tmp1, tmp2, tmp3;
    r43   = (4.0 / 3.0);
    c1345 = C1 * C3 * C4 * C5;
    c34   = C3 * C4;

#pragma omp for nowait schedule(static)
    for(j = jend - 1; j >= jst; j--)
    {
        for(i = iend - 1; i >= ist; i--)
        {
            /*
             * ---------------------------------------------------------------------
             * form the block daigonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            d[j][i][0][0] = 1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
            d[j][i][1][0] = 0.0;
            d[j][i][2][0] = 0.0;
            d[j][i][3][0] = 0.0;
            d[j][i][4][0] = 0.0;
            d[j][i][0][1] =
                dt * 2.0 * (-tx1 * r43 - ty1 - tz1) * (c34 * tmp2 * u[k][j][i][1]);
            d[j][i][1][1] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 * r43 + ty1 + tz1) +
                            dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
            d[j][i][2][1] = 0.0;
            d[j][i][3][1] = 0.0;
            d[j][i][4][1] = 0.0;
            d[j][i][0][2] =
                dt * 2.0 * (-tx1 - ty1 * r43 - tz1) * (c34 * tmp2 * u[k][j][i][2]);
            d[j][i][1][2] = 0.0;
            d[j][i][2][2] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 * r43 + tz1) +
                            dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
            d[j][i][3][2] = 0.0;
            d[j][i][4][2] = 0.0;
            d[j][i][0][3] =
                dt * 2.0 * (-tx1 - ty1 - tz1 * r43) * (c34 * tmp2 * u[k][j][i][3]);
            d[j][i][1][3] = 0.0;
            d[j][i][2][3] = 0.0;
            d[j][i][3][3] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 + tz1 * r43) +
                            dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
            d[j][i][4][3] = 0.0;
            d[j][i][0][4] = -dt * 2.0 *
                            (((tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) +
                               tz1 * (c34 - c1345)) *
                                  (u[k][j][i][1] * u[k][j][i][1]) +
                              (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) +
                               tz1 * (c34 - c1345)) *
                                  (u[k][j][i][2] * u[k][j][i][2]) +
                              (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) +
                               tz1 * (r43 * c34 - c1345)) *
                                  (u[k][j][i][3] * u[k][j][i][3])) *
                                 tmp3 +
                             (tx1 + ty1 + tz1) * c1345 * tmp2 * u[k][j][i][4]);
            d[j][i][1][4] =
                dt * 2.0 *
                (tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (c34 - c1345)) *
                tmp2 * u[k][j][i][1];
            d[j][i][2][4] =
                dt * 2.0 *
                (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) + tz1 * (c34 - c1345)) *
                tmp2 * u[k][j][i][2];
            d[j][i][3][4] =
                dt * 2.0 *
                (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (r43 * c34 - c1345)) *
                tmp2 * u[k][j][i][3];
            d[j][i][4][4] = 1.0 + dt * 2.0 * (tx1 + ty1 + tz1) * c1345 * tmp1 +
                            dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
            /*
             * ---------------------------------------------------------------------
             * form the first block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j][i + 1];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            a[j][i][0][0] = -dt * tx1 * dx1;
            a[j][i][1][0] = dt * tx2;
            a[j][i][2][0] = 0.0;
            a[j][i][3][0] = 0.0;
            a[j][i][4][0] = 0.0;
            a[j][i][0][1] =
                dt * tx2 *
                    (-(u[k][j][i + 1][1] * tmp1) * (u[k][j][i + 1][1] * tmp1) +
                     C2 * qs[k][j][i + 1] * tmp1) -
                dt * tx1 * (-r43 * c34 * tmp2 * u[k][j][i + 1][1]);
            a[j][i][1][1] = dt * tx2 * ((2.0 - C2) * (u[k][j][i + 1][1] * tmp1)) -
                            dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
            a[j][i][2][1] = dt * tx2 * (-C2 * (u[k][j][i + 1][2] * tmp1));
            a[j][i][3][1] = dt * tx2 * (-C2 * (u[k][j][i + 1][3] * tmp1));
            a[j][i][4][1] = dt * tx2 * C2;
            a[j][i][0][2] = dt * tx2 * (-(u[k][j][i + 1][1] * u[k][j][i + 1][2]) * tmp2) -
                            dt * tx1 * (-c34 * tmp2 * u[k][j][i + 1][2]);
            a[j][i][1][2] = dt * tx2 * (u[k][j][i + 1][2] * tmp1);
            a[j][i][2][2] = dt * tx2 * (u[k][j][i + 1][1] * tmp1) -
                            dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
            a[j][i][3][2] = 0.0;
            a[j][i][4][2] = 0.0;
            a[j][i][0][3] = dt * tx2 * (-(u[k][j][i + 1][1] * u[k][j][i + 1][3]) * tmp2) -
                            dt * tx1 * (-c34 * tmp2 * u[k][j][i + 1][3]);
            a[j][i][1][3] = dt * tx2 * (u[k][j][i + 1][3] * tmp1);
            a[j][i][2][3] = 0.0;
            a[j][i][3][3] = dt * tx2 * (u[k][j][i + 1][1] * tmp1) -
                            dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
            a[j][i][4][3] = 0.0;
            a[j][i][0][4] =
                dt * tx2 *
                    ((C2 * 2.0 * qs[k][j][i + 1] - C1 * u[k][j][i + 1][4]) *
                     (u[k][j][i + 1][1] * tmp2)) -
                dt * tx1 *
                    (-(r43 * c34 - c1345) * tmp3 *
                         (u[k][j][i + 1][1] * u[k][j][i + 1][1]) -
                     (c34 - c1345) * tmp3 * (u[k][j][i + 1][2] * u[k][j][i + 1][2]) -
                     (c34 - c1345) * tmp3 * (u[k][j][i + 1][3] * u[k][j][i + 1][3]) -
                     c1345 * tmp2 * u[k][j][i + 1][4]);
            a[j][i][1][4] = dt * tx2 *
                                (C1 * (u[k][j][i + 1][4] * tmp1) -
                                 C2 * (u[k][j][i + 1][1] * u[k][j][i + 1][1] * tmp2 +
                                       qs[k][j][i + 1] * tmp1)) -
                            dt * tx1 * (r43 * c34 - c1345) * tmp2 * u[k][j][i + 1][1];
            a[j][i][2][4] =
                dt * tx2 * (-C2 * (u[k][j][i + 1][2] * u[k][j][i + 1][1]) * tmp2) -
                dt * tx1 * (c34 - c1345) * tmp2 * u[k][j][i + 1][2];
            a[j][i][3][4] =
                dt * tx2 * (-C2 * (u[k][j][i + 1][3] * u[k][j][i + 1][1]) * tmp2) -
                dt * tx1 * (c34 - c1345) * tmp2 * u[k][j][i + 1][3];
            a[j][i][4][4] = dt * tx2 * (C1 * (u[k][j][i + 1][1] * tmp1)) -
                            dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
            /*
             * ---------------------------------------------------------------------
             * form the second block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k][j + 1][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            b[j][i][0][0] = -dt * ty1 * dy1;
            b[j][i][1][0] = 0.0;
            b[j][i][2][0] = dt * ty2;
            b[j][i][3][0] = 0.0;
            b[j][i][4][0] = 0.0;
            b[j][i][0][1] = dt * ty2 * (-(u[k][j + 1][i][1] * u[k][j + 1][i][2]) * tmp2) -
                            dt * ty1 * (-c34 * tmp2 * u[k][j + 1][i][1]);
            b[j][i][1][1] = dt * ty2 * (u[k][j + 1][i][2] * tmp1) -
                            dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
            b[j][i][2][1] = dt * ty2 * (u[k][j + 1][i][1] * tmp1);
            b[j][i][3][1] = 0.0;
            b[j][i][4][1] = 0.0;
            b[j][i][0][2] =
                dt * ty2 *
                    (-(u[k][j + 1][i][2] * tmp1) * (u[k][j + 1][i][2] * tmp1) +
                     C2 * (qs[k][j + 1][i] * tmp1)) -
                dt * ty1 * (-r43 * c34 * tmp2 * u[k][j + 1][i][2]);
            b[j][i][1][2] = dt * ty2 * (-C2 * (u[k][j + 1][i][1] * tmp1));
            b[j][i][2][2] = dt * ty2 * ((2.0 - C2) * (u[k][j + 1][i][2] * tmp1)) -
                            dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
            b[j][i][3][2] = dt * ty2 * (-C2 * (u[k][j + 1][i][3] * tmp1));
            b[j][i][4][2] = dt * ty2 * C2;
            b[j][i][0][3] = dt * ty2 * (-(u[k][j + 1][i][2] * u[k][j + 1][i][3]) * tmp2) -
                            dt * ty1 * (-c34 * tmp2 * u[k][j + 1][i][3]);
            b[j][i][1][3] = 0.0;
            b[j][i][2][3] = dt * ty2 * (u[k][j + 1][i][3] * tmp1);
            b[j][i][3][3] = dt * ty2 * (u[k][j + 1][i][2] * tmp1) -
                            dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
            b[j][i][4][3] = 0.0;
            b[j][i][0][4] =
                dt * ty2 *
                    ((C2 * 2.0 * qs[k][j + 1][i] - C1 * u[k][j + 1][i][4]) *
                     (u[k][j + 1][i][2] * tmp2)) -
                dt * ty1 *
                    (-(c34 - c1345) * tmp3 * (u[k][j + 1][i][1] * u[k][j + 1][i][1]) -
                     (r43 * c34 - c1345) * tmp3 *
                         (u[k][j + 1][i][2] * u[k][j + 1][i][2]) -
                     (c34 - c1345) * tmp3 * (u[k][j + 1][i][3] * u[k][j + 1][i][3]) -
                     c1345 * tmp2 * u[k][j + 1][i][4]);
            b[j][i][1][4] =
                dt * ty2 * (-C2 * (u[k][j + 1][i][1] * u[k][j + 1][i][2]) * tmp2) -
                dt * ty1 * (c34 - c1345) * tmp2 * u[k][j + 1][i][1];
            b[j][i][2][4] = dt * ty2 *
                                (C1 * (u[k][j + 1][i][4] * tmp1) -
                                 C2 * (qs[k][j + 1][i] * tmp1 +
                                       u[k][j + 1][i][2] * u[k][j + 1][i][2] * tmp2)) -
                            dt * ty1 * (r43 * c34 - c1345) * tmp2 * u[k][j + 1][i][2];
            b[j][i][3][4] =
                dt * ty2 * (-C2 * (u[k][j + 1][i][2] * u[k][j + 1][i][3]) * tmp2) -
                dt * ty1 * (c34 - c1345) * tmp2 * u[k][j + 1][i][3];
            b[j][i][4][4] = dt * ty2 * (C1 * (u[k][j + 1][i][2] * tmp1)) -
                            dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
            /*
             * ---------------------------------------------------------------------
             * form the third block sub-diagonal
             * ---------------------------------------------------------------------
             */
            tmp1          = rho_i[k + 1][j][i];
            tmp2          = tmp1 * tmp1;
            tmp3          = tmp1 * tmp2;
            c[j][i][0][0] = -dt * tz1 * dz1;
            c[j][i][1][0] = 0.0;
            c[j][i][2][0] = 0.0;
            c[j][i][3][0] = dt * tz2;
            c[j][i][4][0] = 0.0;
            c[j][i][0][1] = dt * tz2 * (-(u[k + 1][j][i][1] * u[k + 1][j][i][3]) * tmp2) -
                            dt * tz1 * (-c34 * tmp2 * u[k + 1][j][i][1]);
            c[j][i][1][1] = dt * tz2 * (u[k + 1][j][i][3] * tmp1) -
                            dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
            c[j][i][2][1] = 0.0;
            c[j][i][3][1] = dt * tz2 * (u[k + 1][j][i][1] * tmp1);
            c[j][i][4][1] = 0.0;
            c[j][i][0][2] = dt * tz2 * (-(u[k + 1][j][i][2] * u[k + 1][j][i][3]) * tmp2) -
                            dt * tz1 * (-c34 * tmp2 * u[k + 1][j][i][2]);
            c[j][i][1][2] = 0.0;
            c[j][i][2][2] = dt * tz2 * (u[k + 1][j][i][3] * tmp1) -
                            dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
            c[j][i][3][2] = dt * tz2 * (u[k + 1][j][i][2] * tmp1);
            c[j][i][4][2] = 0.0;
            c[j][i][0][3] =
                dt * tz2 *
                    (-(u[k + 1][j][i][3] * tmp1) * (u[k + 1][j][i][3] * tmp1) +
                     C2 * (qs[k + 1][j][i] * tmp1)) -
                dt * tz1 * (-r43 * c34 * tmp2 * u[k + 1][j][i][3]);
            c[j][i][1][3] = dt * tz2 * (-C2 * (u[k + 1][j][i][1] * tmp1));
            c[j][i][2][3] = dt * tz2 * (-C2 * (u[k + 1][j][i][2] * tmp1));
            c[j][i][3][3] = dt * tz2 * (2.0 - C2) * (u[k + 1][j][i][3] * tmp1) -
                            dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
            c[j][i][4][3] = dt * tz2 * C2;
            c[j][i][0][4] =
                dt * tz2 *
                    ((C2 * 2.0 * qs[k + 1][j][i] - C1 * u[k + 1][j][i][4]) *
                     (u[k + 1][j][i][3] * tmp2)) -
                dt * tz1 *
                    (-(c34 - c1345) * tmp3 * (u[k + 1][j][i][1] * u[k + 1][j][i][1]) -
                     (c34 - c1345) * tmp3 * (u[k + 1][j][i][2] * u[k + 1][j][i][2]) -
                     (r43 * c34 - c1345) * tmp3 *
                         (u[k + 1][j][i][3] * u[k + 1][j][i][3]) -
                     c1345 * tmp2 * u[k + 1][j][i][4]);
            c[j][i][1][4] =
                dt * tz2 * (-C2 * (u[k + 1][j][i][1] * u[k + 1][j][i][3]) * tmp2) -
                dt * tz1 * (c34 - c1345) * tmp2 * u[k + 1][j][i][1];
            c[j][i][2][4] =
                dt * tz2 * (-C2 * (u[k + 1][j][i][2] * u[k + 1][j][i][3]) * tmp2) -
                dt * tz1 * (c34 - c1345) * tmp2 * u[k + 1][j][i][2];
            c[j][i][3][4] = dt * tz2 *
                                (C1 * (u[k + 1][j][i][4] * tmp1) -
                                 C2 * (qs[k + 1][j][i] * tmp1 +
                                       u[k + 1][j][i][3] * u[k + 1][j][i][3] * tmp2)) -
                            dt * tz1 * (r43 * c34 - c1345) * tmp2 * u[k + 1][j][i][3];
            c[j][i][4][4] = dt * tz2 * (C1 * (u[k + 1][j][i][3] * tmp1)) -
                            dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
        }
    }
}

/*
 * ---------------------------------------------------------------------
 * to compute the l2-norm of vector v.
 * ---------------------------------------------------------------------
 * to improve cache performance, second two dimensions padded by 1
 * for even number sizes only.  Only needed in v.
 * ---------------------------------------------------------------------
 */
void
l2norm(int nx0, int ny0, int nz0, int ist, int iend, int jst, int jend,
       double v[][ISIZ2 / 2 * 2 + 1][ISIZ1 / 2 * 2 + 1][5], double sum[5])
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

#pragma omp single
    for(m = 0; m < 5; m++)
    {
        sum[m] = 0.0;
    }

#pragma omp for nowait
    for(k = 1; k < nz0 - 1; k++)
    {
        for(j = jst; j < jend; j++)
        {
            for(i = ist; i < iend; i++)
            {
                sum0 = sum0 + v[i][j][k][0] * v[i][j][k][0];
                sum1 = sum1 + v[i][j][k][1] * v[i][j][k][1];
                sum2 = sum2 + v[i][j][k][2] * v[i][j][k][2];
                sum3 = sum3 + v[i][j][k][3] * v[i][j][k][3];
                sum4 = sum4 + v[i][j][k][4] * v[i][j][k][4];
            }
        }
    }

#pragma omp critical
    {
        sum[0] += sum0;
        sum[1] += sum1;
        sum[2] += sum2;
        sum[3] += sum3;
        sum[4] += sum4;
    }
#pragma omp barrier

#pragma omp single
    for(m = 0; m < 5; m++)
    {
        sum[m] = sqrt(sum[m] / ((nx0 - 2) * (ny0 - 2) * (nz0 - 2)));
    }
}

void
pintgr()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k;
    int    ibeg, ifin, ifin1;
    int    jbeg, jfin, jfin1;
    double phi1[ISIZ3 + 2][ISIZ2 + 2];
    double phi2[ISIZ3 + 2][ISIZ2 + 2];
    double frc1, frc2, frc3;
    /*
     * ---------------------------------------------------------------------
     * set up the sub-domains for integeration in each processor
     * ---------------------------------------------------------------------
     */
    ibeg  = ii1;
    ifin  = ii2;
    jbeg  = ji1;
    jfin  = ji2;
    ifin1 = ifin - 1;
    jfin1 = jfin - 1;
    /*
     * ---------------------------------------------------------------------
     * initialize
     * ---------------------------------------------------------------------
     */
    for(i = 0; i <= ISIZ2 + 1; i++)
    {
        for(k = 0; k <= ISIZ3 + 1; k++)
        {
            phi1[k][i] = 0.0;
            phi2[k][i] = 0.0;
        }
    }
    for(j = jbeg; j < jfin; j++)
    {
        for(i = ibeg; i < ifin; i++)
        {
            k          = ki1;
            phi1[j][i] = C2 * (u[k][j][i][4] - 0.50 *
                                                   (u[k][j][i][1] * u[k][j][i][1] +
                                                    u[k][j][i][2] * u[k][j][i][2] +
                                                    u[k][j][i][3] * u[k][j][i][3]) /
                                                   u[k][j][i][0]);
            k          = ki2 - 1;
            phi2[j][i] = C2 * (u[k][j][i][4] - 0.50 *
                                                   (u[k][j][i][1] * u[k][j][i][1] +
                                                    u[k][j][i][2] * u[k][j][i][2] +
                                                    u[k][j][i][3] * u[k][j][i][3]) /
                                                   u[k][j][i][0]);
        }
    }
    frc1 = 0.0;
    for(j = jbeg; j < jfin1; j++)
    {
        for(i = ibeg; i < ifin1; i++)
        {
            frc1 = frc1 +
                   (phi1[j][i] + phi1[j][i + 1] + phi1[j + 1][i] + phi1[j + 1][i + 1] +
                    phi2[j][i] + phi2[j][i + 1] + phi2[j + 1][i] + phi2[j + 1][i + 1]);
        }
    }
    frc1 = dxi * deta * frc1;
    /*
     * ---------------------------------------------------------------------
     * initialize
     * ---------------------------------------------------------------------
     */
    for(i = 0; i <= ISIZ2 + 1; i++)
    {
        for(k = 0; k <= ISIZ3 + 1; k++)
        {
            phi1[k][i] = 0.0;
            phi2[k][i] = 0.0;
        }
    }
    if(jbeg == ji1)
    {
        for(k = ki1; k < ki2; k++)
        {
            for(i = ibeg; i < ifin; i++)
            {
                phi1[k][i] =
                    C2 * (u[k][jbeg][i][4] - 0.50 *
                                                 (u[k][jbeg][i][1] * u[k][jbeg][i][1] +
                                                  u[k][jbeg][i][2] * u[k][jbeg][i][2] +
                                                  u[k][jbeg][i][3] * u[k][jbeg][i][3]) /
                                                 u[k][jbeg][i][0]);
            }
        }
    }
    if(jfin == ji2)
    {
        for(k = ki1; k < ki2; k++)
        {
            for(i = ibeg; i < ifin; i++)
            {
                phi2[k][i] = C2 * (u[k][jfin - 1][i][4] -
                                   0.50 *
                                       (u[k][jfin - 1][i][1] * u[k][jfin - 1][i][1] +
                                        u[k][jfin - 1][i][2] * u[k][jfin - 1][i][2] +
                                        u[k][jfin - 1][i][3] * u[k][jfin - 1][i][3]) /
                                       u[k][jfin - 1][i][0]);
            }
        }
    }
    frc2 = 0.0;
    for(k = ki1; k < ki2 - 1; k++)
    {
        for(i = ibeg; i < ifin1; i++)
        {
            frc2 = frc2 +
                   (phi1[k][i] + phi1[k][i + 1] + phi1[k + 1][i] + phi1[k + 1][i + 1] +
                    phi2[k][i] + phi2[k][i + 1] + phi2[k + 1][i] + phi2[k + 1][i + 1]);
        }
    }
    frc2 = dxi * dzeta * frc2;
    /*
     * ---------------------------------------------------------------------
     * initialize
     * ---------------------------------------------------------------------
     */
    for(i = 0; i <= ISIZ2 + 1; i++)
    {
        for(k = 0; k <= ISIZ3 + 1; k++)
        {
            phi1[k][i] = 0.0;
            phi2[k][i] = 0.0;
        }
    }
    if(ibeg == ii1)
    {
        for(k = ki1; k < ki2; k++)
        {
            for(j = jbeg; j < jfin; j++)
            {
                phi1[k][j] =
                    C2 * (u[k][j][ibeg][4] - 0.50 *
                                                 (u[k][j][ibeg][1] * u[k][j][ibeg][1] +
                                                  u[k][j][ibeg][2] * u[k][j][ibeg][2] +
                                                  u[k][j][ibeg][3] * u[k][j][ibeg][3]) /
                                                 u[k][j][ibeg][0]);
            }
        }
    }
    if(ifin == ii2)
    {
        for(k = ki1; k < ki2; k++)
        {
            for(j = jbeg; j < jfin; j++)
            {
                phi2[k][j] = C2 * (u[k][j][ifin - 1][4] -
                                   0.50 *
                                       (u[k][j][ifin - 1][1] * u[k][j][ifin - 1][1] +
                                        u[k][j][ifin - 1][2] * u[k][j][ifin - 1][2] +
                                        u[k][j][ifin - 1][3] * u[k][j][ifin - 1][3]) /
                                       u[k][j][ifin - 1][0]);
            }
        }
    }
    frc3 = 0.0;
    for(k = ki1; k < ki2 - 1; k++)
    {
        for(j = jbeg; j < jfin1; j++)
        {
            frc3 = frc3 +
                   (phi1[k][j] + phi1[k][j + 1] + phi1[k + 1][j] + phi1[k + 1][j + 1] +
                    phi2[k][j] + phi2[k][j + 1] + phi2[k + 1][j] + phi2[k + 1][j + 1]);
        }
    }
    frc3 = deta * dzeta * frc3;
    frc  = 0.25 * (frc1 + frc2 + frc3);
}

void
read_input()
{
    /*
     * ---------------------------------------------------------------------
     * if input file does not exist, it uses defaults
     * ipr = 1 for detailed progress output
     * inorm = how often the norm is printed (once every inorm iterations)
     * itmax = number of pseudo time steps
     * dt = time step
     * omega 1 over-relaxation factor for SSOR
     * tolrsd = steady state residual tolerance levels
     * nx, ny, nz = number of grid points in x, y, z directions
     * ---------------------------------------------------------------------
     */
    FILE* fp;
    int   avoid_warning = 0;
    if((fp = fopen("inputlu.data", "r")) != nullptr)
    {
        printf("Reading from input file inputlu.data\n");
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%d%d", &ipr, &inorm);
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%d", &itmax);
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%lf", &dt);
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%lf", &omega);
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%lf%lf%lf%lf%lf", &tolrsd[0], &tolrsd[1], &tolrsd[2],
                               &tolrsd[3], &tolrsd[4]);
        while(fgetc(fp) != '\n')
            ;
        while(fgetc(fp) != '\n')
            ;
        avoid_warning = fscanf(fp, "%d%d%d", &nx0, &ny0, &nz0);
        fclose(fp);
    }
    else
    {
        ipr       = IPR_DEFAULT;
        inorm     = INORM_DEFAULT;
        itmax     = ITMAX_DEFAULT;
        dt        = DT_DEFAULT;
        omega     = OMEGA_DEFAULT;
        tolrsd[0] = TOLRSD1_DEF;
        tolrsd[1] = TOLRSD2_DEF;
        tolrsd[2] = TOLRSD3_DEF;
        tolrsd[3] = TOLRSD4_DEF;
        tolrsd[4] = TOLRSD5_DEF;
        nx0       = ISIZ1;
        ny0       = ISIZ2;
        nz0       = ISIZ3;
    }
    (void) avoid_warning;
    /*
     * ---------------------------------------------------------------------
     * check problem size
     * ---------------------------------------------------------------------
     */
    if((nx0 < 4) || (ny0 < 4) || (nz0 < 4))
    {
        printf("     PROBLEM SIZE IS TOO SMALL - \n"
               "     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5\n");
        exit(EXIT_FAILURE);
    }
    if((nx0 > ISIZ1) || (ny0 > ISIZ2) || (nz0 > ISIZ3))
    {
        printf("     PROBLEM SIZE IS TOO LARGE - \n"
               "     NX, NY AND NZ SHOULD BE EQUAL TO \n"
               "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY\n");
        exit(EXIT_FAILURE);
    }
    printf("\n\n NAS Parallel Benchmarks 4.1 Parallel C++ version with OpenMP - LU "
           "Benchmark\n\n");
    printf(" Size: %4dx%4dx%4d\n", nx0, ny0, nz0);
    printf(" Iterations: %4d\n", itmax);
    printf("\n");
}

/*
 * ---------------------------------------------------------------------
 * compute the right hand sides
 * ---------------------------------------------------------------------
 */
void
rhs()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double q;
    double tmp, utmp[ISIZ3][6], rtmp[ISIZ3][5];
    double u21, u31, u41;
    double u21i, u31i, u41i, u51i;
    double u21j, u31j, u41j, u51j;
    double u21k, u31k, u41k, u51k;
    double u21im1, u31im1, u41im1, u51im1;
    double u21jm1, u31jm1, u41jm1, u51jm1;
    double u21km1, u31km1, u41km1, u51km1;
    double flux[ISIZ1][5];

    if(timeron)
    {
        timer_start(T_RHS);
    }
#pragma omp for
    for(k = 0; k < nz; k++)
    {
        for(j = 0; j < ny; j++)
        {
            for(i = 0; i < nx; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] = -frct[k][j][i][m];
                }
                tmp            = 1.0 / u[k][j][i][0];
                rho_i[k][j][i] = tmp;
                qs[k][j][i] =
                    0.50 *
                    (u[k][j][i][1] * u[k][j][i][1] + u[k][j][i][2] * u[k][j][i][2] +
                     u[k][j][i][3] * u[k][j][i][3]) *
                    tmp;
            }
        }
    }
    if(timeron)
    {
        timer_start(T_RHSX);
    }
/*
 * ---------------------------------------------------------------------
 * xi-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 1; k < nz - 1; k++)
    {
        for(j = jst; j < jend; j++)
        {
            for(i = 0; i < nx; i++)
            {
                flux[i][0] = u[k][j][i][1];
                u21        = u[k][j][i][1] * rho_i[k][j][i];
                q          = qs[k][j][i];
                flux[i][1] = u[k][j][i][1] * u21 + C2 * (u[k][j][i][4] - q);
                flux[i][2] = u[k][j][i][2] * u21;
                flux[i][3] = u[k][j][i][3] * u21;
                flux[i][4] = (C1 * u[k][j][i][4] - C2 * q) * u21;
            }
            for(i = ist; i < iend; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        rsd[k][j][i][m] - tx2 * (flux[i + 1][m] - flux[i - 1][m]);
                }
            }
            for(i = ist; i < nx; i++)
            {
                tmp        = rho_i[k][j][i];
                u21i       = tmp * u[k][j][i][1];
                u31i       = tmp * u[k][j][i][2];
                u41i       = tmp * u[k][j][i][3];
                u51i       = tmp * u[k][j][i][4];
                tmp        = rho_i[k][j][i - 1];
                u21im1     = tmp * u[k][j][i - 1][1];
                u31im1     = tmp * u[k][j][i - 1][2];
                u41im1     = tmp * u[k][j][i - 1][3];
                u51im1     = tmp * u[k][j][i - 1][4];
                flux[i][1] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
                flux[i][2] = tx3 * (u31i - u31im1);
                flux[i][3] = tx3 * (u41i - u41im1);
                flux[i][4] = 0.50 * (1.0 - C1 * C5) * tx3 *
                                 ((u21i * u21i + u31i * u31i + u41i * u41i) -
                                  (u21im1 * u21im1 + u31im1 * u31im1 + u41im1 * u41im1)) +
                             (1.0 / 6.0) * tx3 * (u21i * u21i - u21im1 * u21im1) +
                             C1 * C5 * tx3 * (u51i - u51im1);
            }
            for(i = ist; i < iend; i++)
            {
                rsd[k][j][i][0] =
                    rsd[k][j][i][0] +
                    dx1 * tx1 *
                        (u[k][j][i - 1][0] - 2.0 * u[k][j][i][0] + u[k][j][i + 1][0]);
                rsd[k][j][i][1] =
                    rsd[k][j][i][1] + tx3 * C3 * C4 * (flux[i + 1][1] - flux[i][1]) +
                    dx2 * tx1 *
                        (u[k][j][i - 1][1] - 2.0 * u[k][j][i][1] + u[k][j][i + 1][1]);
                rsd[k][j][i][2] =
                    rsd[k][j][i][2] + tx3 * C3 * C4 * (flux[i + 1][2] - flux[i][2]) +
                    dx3 * tx1 *
                        (u[k][j][i - 1][2] - 2.0 * u[k][j][i][2] + u[k][j][i + 1][2]);
                rsd[k][j][i][3] =
                    rsd[k][j][i][3] + tx3 * C3 * C4 * (flux[i + 1][3] - flux[i][3]) +
                    dx4 * tx1 *
                        (u[k][j][i - 1][3] - 2.0 * u[k][j][i][3] + u[k][j][i + 1][3]);
                rsd[k][j][i][4] =
                    rsd[k][j][i][4] + tx3 * C3 * C4 * (flux[i + 1][4] - flux[i][4]) +
                    dx5 * tx1 *
                        (u[k][j][i - 1][4] - 2.0 * u[k][j][i][4] + u[k][j][i + 1][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                rsd[k][j][1][m] =
                    rsd[k][j][1][m] -
                    dssp * (+5.0 * u[k][j][1][m] - 4.0 * u[k][j][2][m] + u[k][j][3][m]);
                rsd[k][j][2][m] =
                    rsd[k][j][2][m] - dssp * (-4.0 * u[k][j][1][m] + 6.0 * u[k][j][2][m] -
                                              4.0 * u[k][j][3][m] + u[k][j][4][m]);
            }
            for(i = 3; i < nx - 3; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        rsd[k][j][i][m] -
                        dssp * (u[k][j][i - 2][m] - 4.0 * u[k][j][i - 1][m] +
                                6.0 * u[k][j][i][m] - 4.0 * u[k][j][i + 1][m] +
                                u[k][j][i + 2][m]);
                }
            }
            for(m = 0; m < 5; m++)
            {
                rsd[k][j][nx - 3][m] =
                    rsd[k][j][nx - 3][m] -
                    dssp * (u[k][j][nx - 5][m] - 4.0 * u[k][j][nx - 4][m] +
                            6.0 * u[k][j][nx - 3][m] - 4.0 * u[k][j][nx - 2][m]);
                rsd[k][j][nx - 2][m] =
                    rsd[k][j][nx - 2][m] -
                    dssp * (u[k][j][nx - 4][m] - 4.0 * u[k][j][nx - 3][m] +
                            5.0 * u[k][j][nx - 2][m]);
            }
        }
    }
    if(timeron)
    {
        timer_stop(T_RHSX);
    }
    if(timeron)
    {
        timer_start(T_RHSY);
    }
/*
 * ---------------------------------------------------------------------
 * eta-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 1; k < nz - 1; k++)
    {
        for(i = ist; i < iend; i++)
        {
            for(j = 0; j < ny; j++)
            {
                flux[j][0] = u[k][j][i][2];
                u31        = u[k][j][i][2] * rho_i[k][j][i];
                q          = qs[k][j][i];
                flux[j][1] = u[k][j][i][1] * u31;
                flux[j][2] = u[k][j][i][2] * u31 + C2 * (u[k][j][i][4] - q);
                flux[j][3] = u[k][j][i][3] * u31;
                flux[j][4] = (C1 * u[k][j][i][4] - C2 * q) * u31;
            }
            for(j = jst; j < jend; j++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        rsd[k][j][i][m] - ty2 * (flux[j + 1][m] - flux[j - 1][m]);
                }
            }
            for(j = jst; j < ny; j++)
            {
                tmp        = rho_i[k][j][i];
                u21j       = tmp * u[k][j][i][1];
                u31j       = tmp * u[k][j][i][2];
                u41j       = tmp * u[k][j][i][3];
                u51j       = tmp * u[k][j][i][4];
                tmp        = rho_i[k][j - 1][i];
                u21jm1     = tmp * u[k][j - 1][i][1];
                u31jm1     = tmp * u[k][j - 1][i][2];
                u41jm1     = tmp * u[k][j - 1][i][3];
                u51jm1     = tmp * u[k][j - 1][i][4];
                flux[j][1] = ty3 * (u21j - u21jm1);
                flux[j][2] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
                flux[j][3] = ty3 * (u41j - u41jm1);
                flux[j][4] = 0.50 * (1.0 - C1 * C5) * ty3 *
                                 ((u21j * u21j + u31j * u31j + u41j * u41j) -
                                  (u21jm1 * u21jm1 + u31jm1 * u31jm1 + u41jm1 * u41jm1)) +
                             (1.0 / 6.0) * ty3 * (u31j * u31j - u31jm1 * u31jm1) +
                             C1 * C5 * ty3 * (u51j - u51jm1);
            }
            for(j = jst; j < jend; j++)
            {
                rsd[k][j][i][0] =
                    rsd[k][j][i][0] +
                    dy1 * ty1 *
                        (u[k][j - 1][i][0] - 2.0 * u[k][j][i][0] + u[k][j + 1][i][0]);
                rsd[k][j][i][1] =
                    rsd[k][j][i][1] + ty3 * C3 * C4 * (flux[j + 1][1] - flux[j][1]) +
                    dy2 * ty1 *
                        (u[k][j - 1][i][1] - 2.0 * u[k][j][i][1] + u[k][j + 1][i][1]);
                rsd[k][j][i][2] =
                    rsd[k][j][i][2] + ty3 * C3 * C4 * (flux[j + 1][2] - flux[j][2]) +
                    dy3 * ty1 *
                        (u[k][j - 1][i][2] - 2.0 * u[k][j][i][2] + u[k][j + 1][i][2]);
                rsd[k][j][i][3] =
                    rsd[k][j][i][3] + ty3 * C3 * C4 * (flux[j + 1][3] - flux[j][3]) +
                    dy4 * ty1 *
                        (u[k][j - 1][i][3] - 2.0 * u[k][j][i][3] + u[k][j + 1][i][3]);
                rsd[k][j][i][4] =
                    rsd[k][j][i][4] + ty3 * C3 * C4 * (flux[j + 1][4] - flux[j][4]) +
                    dy5 * ty1 *
                        (u[k][j - 1][i][4] - 2.0 * u[k][j][i][4] + u[k][j + 1][i][4]);
            }
        }
        /*
         * ---------------------------------------------------------------------
         * fourth-order dissipation
         * ---------------------------------------------------------------------
         */
        for(i = ist; i < iend; i++)
        {
            for(m = 0; m < 5; m++)
            {
                rsd[k][1][i][m] =
                    rsd[k][1][i][m] -
                    dssp * (+5.0 * u[k][1][i][m] - 4.0 * u[k][2][i][m] + u[k][3][i][m]);
                rsd[k][2][i][m] =
                    rsd[k][2][i][m] - dssp * (-4.0 * u[k][1][i][m] + 6.0 * u[k][2][i][m] -
                                              4.0 * u[k][3][i][m] + u[k][4][i][m]);
            }
        }
        for(j = 3; j < ny - 3; j++)
        {
            for(i = ist; i < iend; i++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        rsd[k][j][i][m] -
                        dssp * (u[k][j - 2][i][m] - 4.0 * u[k][j - 1][i][m] +
                                6.0 * u[k][j][i][m] - 4.0 * u[k][j + 1][i][m] +
                                u[k][j + 2][i][m]);
                }
            }
        }
        for(i = ist; i < iend; i++)
        {
            for(m = 0; m < 5; m++)
            {
                rsd[k][ny - 3][i][m] =
                    rsd[k][ny - 3][i][m] -
                    dssp * (u[k][ny - 5][i][m] - 4.0 * u[k][ny - 4][i][m] +
                            6.0 * u[k][ny - 3][i][m] - 4.0 * u[k][ny - 2][i][m]);
                rsd[k][ny - 2][i][m] =
                    rsd[k][ny - 2][i][m] -
                    dssp * (u[k][ny - 4][i][m] - 4.0 * u[k][ny - 3][i][m] +
                            5.0 * u[k][ny - 2][i][m]);
            }
        }
    }
    if(timeron)
    {
        timer_stop(T_RHSY);
    }
    if(timeron)
    {
        timer_start(T_RHSZ);
    }
/*
 * ---------------------------------------------------------------------
 * zeta-direction flux differences
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(j = jst; j < jend; j++)
    {
        for(i = ist; i < iend; i++)
        {
            for(k = 0; k < nz; k++)
            {
                utmp[k][0] = u[k][j][i][0];
                utmp[k][1] = u[k][j][i][1];
                utmp[k][2] = u[k][j][i][2];
                utmp[k][3] = u[k][j][i][3];
                utmp[k][4] = u[k][j][i][4];
                utmp[k][5] = rho_i[k][j][i];
            }
            for(k = 0; k < nz; k++)
            {
                flux[k][0] = utmp[k][3];
                u41        = utmp[k][3] * utmp[k][5];
                q          = qs[k][j][i];
                flux[k][1] = utmp[k][1] * u41;
                flux[k][2] = utmp[k][2] * u41;
                flux[k][3] = utmp[k][3] * u41 + C2 * (utmp[k][4] - q);
                flux[k][4] = (C1 * utmp[k][4] - C2 * q) * u41;
            }
            for(k = 1; k < nz - 1; k++)
            {
                for(m = 0; m < 5; m++)
                {
                    rtmp[k][m] =
                        rsd[k][j][i][m] - tz2 * (flux[k + 1][m] - flux[k - 1][m]);
                }
            }
            for(k = 1; k < nz; k++)
            {
                tmp        = utmp[k][5];
                u21k       = tmp * utmp[k][1];
                u31k       = tmp * utmp[k][2];
                u41k       = tmp * utmp[k][3];
                u51k       = tmp * utmp[k][4];
                tmp        = utmp[k - 1][5];
                u21km1     = tmp * utmp[k - 1][1];
                u31km1     = tmp * utmp[k - 1][2];
                u41km1     = tmp * utmp[k - 1][3];
                u51km1     = tmp * utmp[k - 1][4];
                flux[k][1] = tz3 * (u21k - u21km1);
                flux[k][2] = tz3 * (u31k - u31km1);
                flux[k][3] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
                flux[k][4] = 0.50 * (1.0 - C1 * C5) * tz3 *
                                 ((u21k * u21k + u31k * u31k + u41k * u41k) -
                                  (u21km1 * u21km1 + u31km1 * u31km1 + u41km1 * u41km1)) +
                             (1.0 / 6.0) * tz3 * (u41k * u41k - u41km1 * u41km1) +
                             C1 * C5 * tz3 * (u51k - u51km1);
            }
            for(k = 1; k < nz - 1; k++)
            {
                rtmp[k][0] =
                    rtmp[k][0] +
                    dz1 * tz1 * (utmp[k - 1][0] - 2.0 * utmp[k][0] + utmp[k + 1][0]);
                rtmp[k][1] =
                    rtmp[k][1] + tz3 * C3 * C4 * (flux[k + 1][1] - flux[k][1]) +
                    dz2 * tz1 * (utmp[k - 1][1] - 2.0 * utmp[k][1] + utmp[k + 1][1]);
                rtmp[k][2] =
                    rtmp[k][2] + tz3 * C3 * C4 * (flux[k + 1][2] - flux[k][2]) +
                    dz3 * tz1 * (utmp[k - 1][2] - 2.0 * utmp[k][2] + utmp[k + 1][2]);
                rtmp[k][3] =
                    rtmp[k][3] + tz3 * C3 * C4 * (flux[k + 1][3] - flux[k][3]) +
                    dz4 * tz1 * (utmp[k - 1][3] - 2.0 * utmp[k][3] + utmp[k + 1][3]);
                rtmp[k][4] =
                    rtmp[k][4] + tz3 * C3 * C4 * (flux[k + 1][4] - flux[k][4]) +
                    dz5 * tz1 * (utmp[k - 1][4] - 2.0 * utmp[k][4] + utmp[k + 1][4]);
            }
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for(m = 0; m < 5; m++)
            {
                rsd[1][j][i][m] = rtmp[1][m] - dssp * (+5.0 * utmp[1][m] -
                                                       4.0 * utmp[2][m] + utmp[3][m]);
                rsd[2][j][i][m] =
                    rtmp[2][m] - dssp * (-4.0 * utmp[1][m] + 6.0 * utmp[2][m] -
                                         4.0 * utmp[3][m] + utmp[4][m]);
            }
            for(k = 3; k < nz - 3; k++)
            {
                for(m = 0; m < 5; m++)
                {
                    rsd[k][j][i][m] =
                        rtmp[k][m] -
                        dssp * (utmp[k - 2][m] - 4.0 * utmp[k - 1][m] + 6.0 * utmp[k][m] -
                                4.0 * utmp[k + 1][m] + utmp[k + 2][m]);
                }
            }
            for(m = 0; m < 5; m++)
            {
                rsd[nz - 3][j][i][m] =
                    rtmp[nz - 3][m] -
                    dssp * (utmp[nz - 5][m] - 4.0 * utmp[nz - 4][m] +
                            6.0 * utmp[nz - 3][m] - 4.0 * utmp[nz - 2][m]);
                rsd[nz - 2][j][i][m] =
                    rtmp[nz - 2][m] - dssp * (utmp[nz - 4][m] - 4.0 * utmp[nz - 3][m] +
                                              5.0 * utmp[nz - 2][m]);
            }
        }
    }
    if(timeron)
    {
        timer_stop(T_RHSZ);
    }
    if(timeron)
    {
        timer_stop(T_RHS);
    }
}

/*
 * ---------------------------------------------------------------------
 * set the boundary values of dependent variables
 * ---------------------------------------------------------------------
 */
void
setbv()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double temp1[5], temp2[5];
/*
 * ---------------------------------------------------------------------
 * set the dependent variable values along the top and bottom faces
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(j = 0; j < ny; j++)
    {
        for(i = 0; i < nx; i++)
        {
            exact(i, j, 0, temp1);
            exact(i, j, nz - 1, temp2);
            for(m = 0; m < 5; m++)
            {
                u[0][j][i][m]      = temp1[m];
                u[nz - 1][j][i][m] = temp2[m];
            }
        }
    }
/*
 * ---------------------------------------------------------------------
 * set the dependent variable values along north and south faces
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 0; k < nz; k++)
    {
        for(i = 0; i < nx; i++)
        {
            exact(i, 0, k, temp1);
            exact(i, ny - 1, k, temp2);
            for(m = 0; m < 5; m++)
            {
                u[k][0][i][m]      = temp1[m];
                u[k][ny - 1][i][m] = temp2[m];
            }
        }
    }
/*
 * ---------------------------------------------------------------------
 * set the dependent variable values along east and west faces
 * ---------------------------------------------------------------------
 */
#pragma omp for
    for(k = 0; k < nz; k++)
    {
        for(j = 0; j < ny; j++)
        {
            exact(0, j, k, temp1);
            exact(nx - 1, j, k, temp2);
            for(m = 0; m < 5; m++)
            {
                u[k][j][0][m]      = temp1[m];
                u[k][j][nx - 1][m] = temp2[m];
            }
        }
    }
}

void
setcoeff()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     * set up coefficients
     * ---------------------------------------------------------------------
     */
    dxi   = 1.0 / (nx0 - 1);
    deta  = 1.0 / (ny0 - 1);
    dzeta = 1.0 / (nz0 - 1);
    tx1   = 1.0 / (dxi * dxi);
    tx2   = 1.0 / (2.0 * dxi);
    tx3   = 1.0 / dxi;
    ty1   = 1.0 / (deta * deta);
    ty2   = 1.0 / (2.0 * deta);
    ty3   = 1.0 / deta;
    tz1   = 1.0 / (dzeta * dzeta);
    tz2   = 1.0 / (2.0 * dzeta);
    tz3   = 1.0 / dzeta;
    /*
     * ---------------------------------------------------------------------
     * diffusion coefficients
     * ---------------------------------------------------------------------
     */
    dx1 = 0.75;
    dx2 = dx1;
    dx3 = dx1;
    dx4 = dx1;
    dx5 = dx1;
    dy1 = 0.75;
    dy2 = dy1;
    dy3 = dy1;
    dy4 = dy1;
    dy5 = dy1;
    dz1 = 1.00;
    dz2 = dz1;
    dz3 = dz1;
    dz4 = dz1;
    dz5 = dz1;
    /*
     * ---------------------------------------------------------------------
     * fourth difference dissipation
     * ---------------------------------------------------------------------
     */
    dssp = (max(max(dx1, dy1), dz1)) / 4.0;
    /*
     * ---------------------------------------------------------------------
     * coefficients of the exact solution to the first pde
     * ---------------------------------------------------------------------
     */
    ce[0][0]  = 2.0;
    ce[1][0]  = 0.0;
    ce[2][0]  = 0.0;
    ce[3][0]  = 4.0;
    ce[4][0]  = 5.0;
    ce[5][0]  = 3.0;
    ce[6][0]  = 5.0e-01;
    ce[7][0]  = 2.0e-02;
    ce[8][0]  = 1.0e-02;
    ce[9][0]  = 3.0e-02;
    ce[10][0] = 5.0e-01;
    ce[11][0] = 4.0e-01;
    ce[12][0] = 3.0e-01;
    /*
     * ---------------------------------------------------------------------
     * coefficients of the exact solution to the second pde
     * ---------------------------------------------------------------------
     */
    ce[0][1]  = 1.0;
    ce[1][1]  = 0.0;
    ce[2][1]  = 0.0;
    ce[3][1]  = 0.0;
    ce[4][1]  = 1.0;
    ce[5][1]  = 2.0;
    ce[6][1]  = 3.0;
    ce[7][1]  = 1.0e-02;
    ce[8][1]  = 3.0e-02;
    ce[9][1]  = 2.0e-02;
    ce[10][1] = 4.0e-01;
    ce[11][1] = 3.0e-01;
    ce[12][1] = 5.0e-01;
    /*
     * ---------------------------------------------------------------------
     * coefficients of the exact solution to the third pde
     * ---------------------------------------------------------------------
     */
    ce[0][2]  = 2.0;
    ce[1][2]  = 2.0;
    ce[2][2]  = 0.0;
    ce[3][2]  = 0.0;
    ce[4][2]  = 0.0;
    ce[5][2]  = 2.0;
    ce[6][2]  = 3.0;
    ce[7][2]  = 4.0e-02;
    ce[8][2]  = 3.0e-02;
    ce[9][2]  = 5.0e-02;
    ce[10][2] = 3.0e-01;
    ce[11][2] = 5.0e-01;
    ce[12][2] = 4.0e-01;
    /*
     * ---------------------------------------------------------------------
     * coefficients of the exact solution to the fourth pde
     * ---------------------------------------------------------------------
     */
    ce[0][3]  = 2.0;
    ce[1][3]  = 2.0;
    ce[2][3]  = 0.0;
    ce[3][3]  = 0.0;
    ce[4][3]  = 0.0;
    ce[5][3]  = 2.0;
    ce[6][3]  = 3.0;
    ce[7][3]  = 3.0e-02;
    ce[8][3]  = 5.0e-02;
    ce[9][3]  = 4.0e-02;
    ce[10][3] = 2.0e-01;
    ce[11][3] = 1.0e-01;
    ce[12][3] = 3.0e-01;
    /*
     * ---------------------------------------------------------------------
     * coefficients of the exact solution to the fifth pde
     * ---------------------------------------------------------------------
     */
    ce[0][4]  = 5.0;
    ce[1][4]  = 4.0;
    ce[2][4]  = 3.0;
    ce[3][4]  = 2.0;
    ce[4][4]  = 1.0e-01;
    ce[5][4]  = 4.0e-01;
    ce[6][4]  = 3.0e-01;
    ce[7][4]  = 5.0e-02;
    ce[8][4]  = 4.0e-02;
    ce[9][4]  = 3.0e-02;
    ce[10][4] = 1.0e-01;
    ce[11][4] = 3.0e-01;
    ce[12][4] = 2.0e-01;
}

/*
 * ---------------------------------------------------------------------
 * set the initial values of independent variables based on tri-linear
 * interpolation of boundary values in the computational space.
 * ---------------------------------------------------------------------
 */
void
setiv()
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m;
    double xi, eta, zeta;
    double pxi, peta, pzeta;
    double ue_1jk[5], ue_nx0jk[5], ue_i1k[5];
    double ue_iny0k[5], ue_ij1[5], ue_ijnz[5];

#pragma omp for
    for(k = 1; k < nz - 1; k++)
    {
        zeta = ((double) k) / (nz - 1);
        for(j = 1; j < ny - 1; j++)
        {
            eta = ((double) j) / (ny0 - 1);
            for(i = 1; i < nx - 1; i++)
            {
                xi = ((double) i) / (nx0 - 1);
                exact(0, j, k, ue_1jk);
                exact(nx0 - 1, j, k, ue_nx0jk);
                exact(i, 0, k, ue_i1k);
                exact(i, ny0 - 1, k, ue_iny0k);
                exact(i, j, 0, ue_ij1);
                exact(i, j, nz - 1, ue_ijnz);
                for(m = 0; m < 5; m++)
                {
                    pxi           = (1.0 - xi) * ue_1jk[m] + xi * ue_nx0jk[m];
                    peta          = (1.0 - eta) * ue_i1k[m] + eta * ue_iny0k[m];
                    pzeta         = (1.0 - zeta) * ue_ij1[m] + zeta * ue_ijnz[m];
                    u[k][j][i][m] = pxi + peta + pzeta - pxi * peta - peta * pzeta -
                                    pzeta * pxi + pxi * peta * pzeta;
                }
            }
        }
    }
}

/*
 * ---------------------------------------------------------------------
 * to perform pseudo-time stepping SSOR iterations
 * for five nonlinear pde's.
 * ---------------------------------------------------------------------
 */
void
ssor(int niter)
{
    /*
     * ---------------------------------------------------------------------
     * local variables
     * ---------------------------------------------------------------------
     */
    int    i, j, k, m, n;
    int    istep;
    double tmp, tv[ISIZ2 * (ISIZ1 / 2 * 2 + 1) * 5];
    double delunm[5];
    /*
     * ---------------------------------------------------------------------
     * begin pseudo-time stepping iterations
     * ---------------------------------------------------------------------
     */
    tmp = 1.0 / (omega * (2.0 - omega));

/*
 * ---------------------------------------------------------------------
 * initialize a,b,c,d to zero (guarantees that page tables have been
 * formed, if applicable on given architecture, before timestepping).
 * ---------------------------------------------------------------------
 */
#pragma omp parallel for private(i, j, n, m)
    for(j = 0; j < ISIZ2; j++)
    {
        for(i = 0; i < ISIZ1; i++)
        {
            for(n = 0; n < 5; n++)
            {
                for(m = 0; m < 5; m++)
                {
                    a[j][i][n][m] = 0.0;
                    b[j][i][n][m] = 0.0;
                    c[j][i][n][m] = 0.0;
                    d[j][i][n][m] = 0.0;
                }
            }
        }
    }
    for(i = 1; i <= T_LAST; i++)
    {
        timer_clear(i);
    }

#pragma omp parallel
    {
        /*
         * ---------------------------------------------------------------------
         * compute the steady-state residuals
         * ---------------------------------------------------------------------
         */
        rhs();

        /*
         * ---------------------------------------------------------------------
         * compute the L2 norms of newton iteration residuals
         * ---------------------------------------------------------------------
         */
        l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd, rsdnm);
    } /* end parallel */

    for(i = 1; i <= T_LAST; i++)
    {
        timer_clear(i);
    }
    timer_start(1);

#pragma omp parallel private(istep, i, j, k, m)
    {
        /*
         * ---------------------------------------------------------------------
         * the timestep loop
         * ---------------------------------------------------------------------
         */
        for(istep = 1; istep <= niter; istep++)
        {
            if((istep % 20) == 0 || istep == itmax || istep == 1)
            {
#pragma omp master
                if(niter > 1)
                {
                    printf(" Time step %4d\n", istep);
                }
            }
            /*
             * ---------------------------------------------------------------------
             * perform SSOR iteration
             * ---------------------------------------------------------------------
             */
            if(timeron)
            {
#pragma omp master
                timer_start(T_RHS);
            }
#pragma omp for
            for(k = 1; k < nz - 1; k++)
            {
                for(j = jst; j < jend; j++)
                {
                    for(i = ist; i < iend; i++)
                    {
                        for(m = 0; m < 5; m++)
                        {
                            rsd[k][j][i][m] = dt * rsd[k][j][i][m];
                        }
                    }
                }
            }
            if(timeron)
            {
#pragma omp master
                timer_stop(T_RHS);
            }

            for(k = 1; k < nz - 1; k++)
            {
                /*
                 * ---------------------------------------------------------------------
                 * form the lower triangular part of the jacobian matrix
                 * ---------------------------------------------------------------------
                 */
                if(timeron)
                {
#pragma omp master
                    timer_start(T_JACLD);
                }
                jacld(k);
                if(timeron)
                {
#pragma omp master
                    timer_stop(T_JACLD);
                }

                /*
                 * ---------------------------------------------------------------------
                 * perform the lower triangular solution
                 * ---------------------------------------------------------------------
                 */
                if(timeron)
                {
#pragma omp master
                    timer_start(T_BLTS);
                }

                blts(nx, ny, nz, k, omega, rsd, a, b, c, d, ist, iend, jst, jend, nx0,
                     ny0);

                if(timeron)
                {
#pragma omp master
                    timer_stop(T_BLTS);
                }
            }

#pragma omp barrier

            for(k = nz - 2; k > 0; k--)
            {
                /*
                 * ---------------------------------------------------------------------
                 * form the strictly upper triangular part of the jacobian matrix
                 * ---------------------------------------------------------------------
                 */
                if(timeron)
                {
#pragma omp master
                    timer_start(T_JACU);
                }
                jacu(k);
                if(timeron)
                {
#pragma omp master
                    timer_stop(T_JACU);
                }
                /*
                 * ---------------------------------------------------------------------
                 * perform the upper triangular solution
                 * ---------------------------------------------------------------------
                 */
                if(timeron)
                {
#pragma omp master
                    timer_start(T_BUTS);
                }

                buts(nx, ny, nz, k, omega, rsd, tv, d, a, b, c, ist, iend, jst, jend, nx0,
                     ny0);

                if(timeron)
                {
#pragma omp master
                    timer_stop(T_BUTS);
                }
            }

#pragma omp barrier

            /*
             * ---------------------------------------------------------------------
             * update the variables
             * ---------------------------------------------------------------------
             */
            if(timeron)
            {
#pragma omp master
                timer_start(T_ADD);
            }

#pragma omp for
            for(k = 1; k < nz - 1; k++)
            {
                for(j = jst; j < jend; j++)
                {
                    for(i = ist; i < iend; i++)
                    {
                        for(m = 0; m < 5; m++)
                        {
                            u[k][j][i][m] = u[k][j][i][m] + tmp * rsd[k][j][i][m];
                        }
                    }
                }
            }
            if(timeron)
            {
#pragma omp master
                timer_stop(T_ADD);
            }
            /*
             * ---------------------------------------------------------------------
             * compute the max-norms of newton iteration corrections
             * ---------------------------------------------------------------------
             */

            if((istep % inorm) == 0)
            {
                if(timeron)
                {
#pragma omp master
                    timer_start(T_L2NORM);
                }
                l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd, delunm);
                if(timeron)
                {
#pragma omp master
                    timer_stop(T_L2NORM);
                }
            }
            /*
             * ---------------------------------------------------------------------
             * compute the steady-state residuals
             * ---------------------------------------------------------------------
             */
            rhs();

            /*
             * ---------------------------------------------------------------------
             * compute the max-norms of newton iteration residuals
             * ---------------------------------------------------------------------
             */
            if(((istep % inorm) == 0) || (istep == itmax))
            {
                if(timeron)
                {
#pragma omp master
                    timer_start(T_L2NORM);
                }
                l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd, rsdnm);
                if(timeron)
                {
#pragma omp master
                    timer_stop(T_L2NORM);
                }
            }
            /*
             * ---------------------------------------------------------------------
             * check the newton-iteration residuals against the tolerance levels
             * ---------------------------------------------------------------------
             */
            if((rsdnm[0] < tolrsd[0]) && (rsdnm[1] < tolrsd[1]) &&
               (rsdnm[2] < tolrsd[2]) && (rsdnm[3] < tolrsd[3]) && (rsdnm[4] < tolrsd[4]))
            {
#pragma omp master
                printf(" \n convergence was achieved after %4d pseudo-time steps\n",
                       istep);
                break;
            }
        }
    } /* end parallel */

    timer_stop(1);
    maxtime = timer_read(1);
}

/*
 * ---------------------------------------------------------------------
 * verification routine
 * ---------------------------------------------------------------------
 */
void
verify(double xcr[], double xce[], double xci, char* class_npb, boolean* verified)
{
    double xcrref[5], xceref[5], xciref;
    double xcrdif[5], xcedif[5], xcidif;
    double epsilon, dtref = 0.0;
    int    m;
    /*
     * ---------------------------------------------------------------------
     * tolerance level
     * ---------------------------------------------------------------------
     */
    epsilon    = 1.0e-08;
    *class_npb = 'U';
    *verified  = TRUE;
    for(m = 0; m < 5; m++)
    {
        xcrref[m] = 1.0;
        xceref[m] = 1.0;
    }
    xciref = 1.0;
    if((nx0 == 12) && (ny0 == 12) && (nz0 == 12) && (itmax == 50))
    {
        *class_npb = 'S';
        dtref      = 5.0e-1;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (12X12X12) grid,
         * after 50 time steps, with DT = 5.0d-01
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 1.6196343210976702e-02;
        xcrref[1] = 2.1976745164821318e-03;
        xcrref[2] = 1.5179927653399185e-03;
        xcrref[3] = 1.5029584435994323e-03;
        xcrref[4] = 3.4264073155896461e-02;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (12X12X12) grid,
         * after 50 time steps, with DT = 5.0d-01
         * ---------------------------------------------------------------------
         */
        xceref[0] = 6.4223319957960924e-04;
        xceref[1] = 8.4144342047347926e-05;
        xceref[2] = 5.8588269616485186e-05;
        xceref[3] = 5.8474222595157350e-05;
        xceref[4] = 1.3103347914111294e-03;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (12X12X12) grid,
         * after 50 time steps, with DT = 5.0d-01
         * ---------------------------------------------------------------------
         */
        xciref = 7.8418928865937083e+00;
    }
    else if((nx0 == 33) && (ny0 == 33) && (nz0 == 33) && (itmax == 300))
    {
        *class_npb = 'W'; /* SPEC95fp size */
        dtref      = 1.5e-3;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (33x33x33) grid,
         * after 300 time steps, with DT = 1.5d-3
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.1236511638192e+02;
        xcrref[1] = 0.1317228477799e+01;
        xcrref[2] = 0.2550120713095e+01;
        xcrref[3] = 0.2326187750252e+01;
        xcrref[4] = 0.2826799444189e+02;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (33X33X33) grid,
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.4867877144216e+00;
        xceref[1] = 0.5064652880982e-01;
        xceref[2] = 0.9281818101960e-01;
        xceref[3] = 0.8570126542733e-01;
        xceref[4] = 0.1084277417792e+01;
        /*
         * ---------------------------------------------------------------------
         * rReference value of surface integral, for the (33X33X33) grid,
         * after 300 time steps, with DT = 1.5d-3
         * ---------------------------------------------------------------------
         */
        xciref = 0.1161399311023e+02;
    }
    else if((nx0 == 64) && (ny0 == 64) && (nz0 == 64) && (itmax == 250))
    {
        *class_npb = 'A';
        dtref      = 2.0e+0;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (64X64X64) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 7.7902107606689367e+02;
        xcrref[1] = 6.3402765259692870e+01;
        xcrref[2] = 1.9499249727292479e+02;
        xcrref[3] = 1.7845301160418537e+02;
        xcrref[4] = 1.8384760349464247e+03;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (64X64X64) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xceref[0] = 2.9964085685471943e+01;
        xceref[1] = 2.8194576365003349e+00;
        xceref[2] = 7.3473412698774742e+00;
        xceref[3] = 6.7139225687777051e+00;
        xceref[4] = 7.0715315688392578e+01;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (64X64X64) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xciref = 2.6030925604886277e+01;
    }
    else if((nx0 == 102) && (ny0 == 102) && (nz0 == 102) && (itmax == 250))
    {
        *class_npb = 'B';
        dtref      = 2.0e+0;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (102X102X102) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 3.5532672969982736e+03;
        xcrref[1] = 2.6214750795310692e+02;
        xcrref[2] = 8.8333721850952190e+02;
        xcrref[3] = 7.7812774739425265e+02;
        xcrref[4] = 7.3087969592545314e+03;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (102X102X102)
         * grid, after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xceref[0] = 1.1401176380212709e+02;
        xceref[1] = 8.1098963655421574e+00;
        xceref[2] = 2.8480597317698308e+01;
        xceref[3] = 2.5905394567832939e+01;
        xceref[4] = 2.6054907504857413e+02;
        /*
           c---------------------------------------------------------------------
         * reference value of surface integral, for the (102X102X102) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xciref = 4.7887162703308227e+01;
    }
    else if((nx0 == 162) && (ny0 == 162) && (nz0 == 162) && (itmax == 250))
    {
        *class_npb = 'C';
        dtref      = 2.0e+0;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (162X162X162) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 1.03766980323537846e+04;
        xcrref[1] = 8.92212458801008552e+02;
        xcrref[2] = 2.56238814582660871e+03;
        xcrref[3] = 2.19194343857831427e+03;
        xcrref[4] = 1.78078057261061185e+04;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (162X162X162)
         * grid, after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xceref[0] = 2.15986399716949279e+02;
        xceref[1] = 1.55789559239863600e+01;
        xceref[2] = 5.41318863077207766e+01;
        xceref[3] = 4.82262643154045421e+01;
        xceref[4] = 4.55902910043250358e+02;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (162X162X162) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xciref = 6.66404553572181300e+01;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (162X162X162) grid,
         * after 250 time steps, with DT = 2.0d+00
         * ---------------------------------------------------------------------
         */
        xciref = 6.66404553572181300e+01;
    }
    else if((nx0 == 408) && (ny0 == 408) && (nz0 == 408) && (itmax == 300))
    {
        *class_npb = 'D';
        dtref      = 1.0e+0;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (408X408X408) grid,
         * after 300 time steps, with DT = 1.0d+00
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.4868417937025e+05;
        xcrref[1] = 0.4696371050071e+04;
        xcrref[2] = 0.1218114549776e+05;
        xcrref[3] = 0.1033801493461e+05;
        xcrref[4] = 0.7142398413817e+05;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (408X408X408)
         * grid, after 300 time steps, with DT = 1.0d+00
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.3752393004482e+03;
        xceref[1] = 0.3084128893659e+02;
        xceref[2] = 0.9434276905469e+02;
        xceref[3] = 0.8230686681928e+02;
        xceref[4] = 0.7002620636210e+03;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (408X408X408) grid,
         * after 300 time steps, with DT = 1.0d+00
         * ---------------------------------------------------------------------
         */
        xciref = 0.8334101392503e+02;
    }
    else if((nx0 == 1020) && (ny0 == 1020) && (nz0 == 1020) && (itmax == 300))
    {
        *class_npb = 'E';
        dtref      = 0.5e+0;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual, for the (1020X1020X1020) grid,
         * after 300 time steps, with DT = 0.5d+00
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.2099641687874e+06;
        xcrref[1] = 0.2130403143165e+05;
        xcrref[2] = 0.5319228789371e+05;
        xcrref[3] = 0.4509761639833e+05;
        xcrref[4] = 0.2932360006590e+06;
        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error, for the (1020X1020X1020)
         * grid, after 300 time steps, with DT = 0.5d+00
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.4800572578333e+03;
        xceref[1] = 0.4221993400184e+02;
        xceref[2] = 0.1210851906824e+03;
        xceref[3] = 0.1047888986770e+03;
        xceref[4] = 0.8363028257389e+03;
        /*
         * ---------------------------------------------------------------------
         * reference value of surface integral, for the (1020X1020X1020) grid,
         * after 300 time steps, with DT = 0.5d+00
         * ---------------------------------------------------------------------
         */
        xciref = 0.9512163272273e+02;
    }
    else
    {
        *verified = FALSE;
    }
    /*
     * ---------------------------------------------------------------------
     * verification test for residuals if gridsize is one of
     * the defined grid sizes above (class .ne. 'U')
     * ---------------------------------------------------------------------
     * compute the difference of solution values and the known reference values.
     * ---------------------------------------------------------------------
     */
    for(m = 0; m < 5; m++)
    {
        xcrdif[m] = fabs((xcr[m] - xcrref[m]) / xcrref[m]);
        xcedif[m] = fabs((xce[m] - xceref[m]) / xceref[m]);
    }
    xcidif = fabs((xci - xciref) / xciref);
    /*
     * ---------------------------------------------------------------------
     * output the comparison of computed results to known cases.
     * ---------------------------------------------------------------------
     */
    if(*class_npb != 'U')
    {
        printf("\n Verification being performed for class_npb %c\n", *class_npb);
        printf(" Accuracy setting for epsilon = %20.13E\n", epsilon);
        *verified = (fabs(dt - dtref) <= epsilon);
        if(!(*verified))
        {
            *class_npb = 'U';
            printf(" DT does not match the reference value of %15.8E\n", dtref);
        }
    }
    else
    {
        printf(" Unknown class_npb\n");
    }
    if(*class_npb != 'U')
    {
        printf(" Comparison of RMS-norms of residual\n");
    }
    else
    {
        printf(" RMS-norms of residual\n");
    }
    for(m = 0; m < 5; m++)
    {
        if(*class_npb == 'U')
        {
            printf("          %2d  %20.13E\n", m + 1, xcr[m]);
        }
        else if(xcrdif[m] <= epsilon)
        {
            printf("          %2d  %20.13E%20.13E%20.13E\n", m + 1, xcr[m], xcrref[m],
                   xcrdif[m]);
        }
        else
        {
            *verified = FALSE;
            printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n", m + 1, xcr[m], xcrref[m],
                   xcrdif[m]);
        }
    }
    if(*class_npb != 'U')
    {
        printf(" Comparison of RMS-norms of solution error\n");
    }
    else
    {
        printf(" RMS-norms of solution error\n");
    }
    for(m = 0; m < 5; m++)
    {
        if(*class_npb == 'U')
        {
            printf("          %2d  %20.13E\n", m + 1, xce[m]);
        }
        else if(xcedif[m] <= epsilon)
        {
            printf("          %2d  %20.13E%20.13E%20.13E\n", m + 1, xce[m], xceref[m],
                   xcedif[m]);
        }
        else
        {
            *verified = FALSE;
            printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n", m + 1, xce[m], xceref[m],
                   xcedif[m]);
        }
    }
    if(*class_npb != 'U')
    {
        printf(" Comparison of surface integral\n");
    }
    else
    {
        printf(" Surface integral\n");
    }
    if(*class_npb == 'U')
    {
        printf("              %20.13E\n", xci);
    }
    else if(xcidif <= epsilon)
    {
        printf("              %20.13E%20.13E%20.13E\n", xci, xciref, xcidif);
    }
    else
    {
        *verified = FALSE;
        printf(" FAILURE:     %20.13E%20.13E%20.13E\n", xci, xciref, xcidif);
    }
    if(*class_npb == 'U')
    {
        printf(" No reference values provided\n");
        printf("No verification performed\n");
    }
    else if(*verified)
    {
        printf(" Verification Successful\n");
    }
    else
    {
        printf(" Verification failed\n");
    }
}
