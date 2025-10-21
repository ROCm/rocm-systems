
#include <climits>
#include <ctype.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "lulesh.h"

#include "causal.hpp"

static Kokkos::View<Real_t*> buffer;
static size_t                buffer_size;
static size_t                buffer_offset;
static int                   do_atomic;

void
ResizeBuffer(const size_t size)
{
    buffer_offset = 0;
    if(size / sizeof(Real_t) + 1 > buffer_size)
    {
        buffer_size = size / sizeof(Real_t) + 1;
        buffer      = Kokkos::View<Real_t*>("Buffer", buffer_size);
    }
}

template <class Type>
Type*
AllocateFromBuffer(const Index_t& count)
{
    const Index_t offset = (count * sizeof(Type) + sizeof(Real_t) - 1) / sizeof(Real_t);
    Real_t*       ptr    = buffer.data() + buffer_offset;
    buffer_offset += ((offset + 511) / 512) * 512;
    return static_cast<Type*>(ptr);
}

static inline void
TimeIncrement(Domain& domain)
{
    Real_t targetdt = domain.stoptime() - domain.time();

    if((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0)))
    {
        // CAUSAL_BEGIN("TimeIncrement_Iteration")
        Real_t ratio;
        Real_t olddt = domain.deltatime();

        Real_t gnewdt = Real_t(1.0e+20);
        Real_t newdt;
        if(domain.dtcourant() < gnewdt)
        {
            gnewdt = domain.dtcourant() / Real_t(2.0);
        }
        if(domain.dthydro() < gnewdt)
        {
            gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0);
        }

#if USE_MPI
        MPI_Allreduce(&gnewdt, &newdt, 1,
                      ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE), MPI_MIN,
                      MPI_COMM_WORLD);
#else
        newdt = gnewdt;
#endif

        ratio = newdt / olddt;
        if(ratio >= Real_t(1.0))
        {
            if(ratio < domain.deltatimemultlb())
            {
                newdt = olddt;
            }
            else if(ratio > domain.deltatimemultub())
            {
                newdt = olddt * domain.deltatimemultub();
            }
        }

        if(newdt > domain.dtmax())
        {
            newdt = domain.dtmax();
        }
        domain.deltatime() = newdt;
        CAUSAL_PROGRESS_NAMED("TimeIncrement_Iteration");
        // CAUSAL_END("TimeIncrement_Iteration")
    }

    if((targetdt > domain.deltatime()) &&
       (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))))
    {
        targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0);
    }

    if(targetdt < domain.deltatime())
    {
        domain.deltatime() = targetdt;
    }

    domain.time() += domain.deltatime();

    ++domain.cycle();
}

KOKKOS_INLINE_FUNCTION void
CollectDomainNodesToElemNodes(const Domain& domain, const Index_t* elemToNode,
                              Real_t elemX[8], Real_t elemY[8], Real_t elemZ[8])
{
    Index_t nd0i = elemToNode[0];
    Index_t nd1i = elemToNode[1];
    Index_t nd2i = elemToNode[2];
    Index_t nd3i = elemToNode[3];
    Index_t nd4i = elemToNode[4];
    Index_t nd5i = elemToNode[5];
    Index_t nd6i = elemToNode[6];
    Index_t nd7i = elemToNode[7];

    elemX[0] = domain.c_x(nd0i);
    elemX[1] = domain.c_x(nd1i);
    elemX[2] = domain.c_x(nd2i);
    elemX[3] = domain.c_x(nd3i);
    elemX[4] = domain.c_x(nd4i);
    elemX[5] = domain.c_x(nd5i);
    elemX[6] = domain.c_x(nd6i);
    elemX[7] = domain.c_x(nd7i);

    elemY[0] = domain.c_y(nd0i);
    elemY[1] = domain.c_y(nd1i);
    elemY[2] = domain.c_y(nd2i);
    elemY[3] = domain.c_y(nd3i);
    elemY[4] = domain.c_y(nd4i);
    elemY[5] = domain.c_y(nd5i);
    elemY[6] = domain.c_y(nd6i);
    elemY[7] = domain.c_y(nd7i);

    elemZ[0] = domain.c_z(nd0i);
    elemZ[1] = domain.c_z(nd1i);
    elemZ[2] = domain.c_z(nd2i);
    elemZ[3] = domain.c_z(nd3i);
    elemZ[4] = domain.c_z(nd4i);
    elemZ[5] = domain.c_z(nd5i);
    elemZ[6] = domain.c_z(nd6i);
    elemZ[7] = domain.c_z(nd7i);
}

static inline void
InitStressTermsForElems(Domain& domain, Real_t* sigxx, Real_t* sigyy, Real_t* sigzz,
                        Index_t numElem)
{
    Kokkos::parallel_for(
        "InitStressTermsForElems", numElem, KOKKOS_LAMBDA(const Index_t& i) {
            sigxx[i] = sigyy[i] = sigzz[i] = -domain.p(i) - domain.q(i);
        });
}

KOKKOS_INLINE_FUNCTION void
CalcElemShapeFunctionDerivatives(Real_t const x[], Real_t const y[], Real_t const z[],
                                 Real_t b[][8], Real_t* const volume)
{
    const Real_t x0 = x[0];
    const Real_t x1 = x[1];
    const Real_t x2 = x[2];
    const Real_t x3 = x[3];
    const Real_t x4 = x[4];
    const Real_t x5 = x[5];
    const Real_t x6 = x[6];
    const Real_t x7 = x[7];

    const Real_t y0 = y[0];
    const Real_t y1 = y[1];
    const Real_t y2 = y[2];
    const Real_t y3 = y[3];
    const Real_t y4 = y[4];
    const Real_t y5 = y[5];
    const Real_t y6 = y[6];
    const Real_t y7 = y[7];

    const Real_t z0 = z[0];
    const Real_t z1 = z[1];
    const Real_t z2 = z[2];
    const Real_t z3 = z[3];
    const Real_t z4 = z[4];
    const Real_t z5 = z[5];
    const Real_t z6 = z[6];
    const Real_t z7 = z[7];

    Real_t fjxxi, fjxet, fjxze;
    Real_t fjyxi, fjyet, fjyze;
    Real_t fjzxi, fjzet, fjzze;
    Real_t cjxxi, cjxet, cjxze;
    Real_t cjyxi, cjyet, cjyze;
    Real_t cjzxi, cjzet, cjzze;

    fjxxi = Real_t(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
    fjxet = Real_t(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
    fjxze = Real_t(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

    fjyxi = Real_t(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
    fjyet = Real_t(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
    fjyze = Real_t(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

    fjzxi = Real_t(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
    fjzet = Real_t(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
    fjzze = Real_t(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

    cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
    cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
    cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);

    cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
    cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
    cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);

    cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
    cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
    cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);

    b[0][0] = -cjxxi - cjxet - cjxze;
    b[0][1] = cjxxi - cjxet - cjxze;
    b[0][2] = cjxxi + cjxet - cjxze;
    b[0][3] = -cjxxi + cjxet - cjxze;
    b[0][4] = -b[0][2];
    b[0][5] = -b[0][3];
    b[0][6] = -b[0][0];
    b[0][7] = -b[0][1];

    b[1][0] = -cjyxi - cjyet - cjyze;
    b[1][1] = cjyxi - cjyet - cjyze;
    b[1][2] = cjyxi + cjyet - cjyze;
    b[1][3] = -cjyxi + cjyet - cjyze;
    b[1][4] = -b[1][2];
    b[1][5] = -b[1][3];
    b[1][6] = -b[1][0];
    b[1][7] = -b[1][1];

    b[2][0] = -cjzxi - cjzet - cjzze;
    b[2][1] = cjzxi - cjzet - cjzze;
    b[2][2] = cjzxi + cjzet - cjzze;
    b[2][3] = -cjzxi + cjzet - cjzze;
    b[2][4] = -b[2][2];
    b[2][5] = -b[2][3];
    b[2][6] = -b[2][0];
    b[2][7] = -b[2][1];

    *volume = Real_t(8.) * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

KOKKOS_INLINE_FUNCTION void
SumElemFaceNormal(Real_t* normalX0, Real_t* normalY0, Real_t* normalZ0, Real_t* normalX1,
                  Real_t* normalY1, Real_t* normalZ1, Real_t* normalX2, Real_t* normalY2,
                  Real_t* normalZ2, Real_t* normalX3, Real_t* normalY3, Real_t* normalZ3,
                  const Real_t x0, const Real_t y0, const Real_t z0, const Real_t x1,
                  const Real_t y1, const Real_t z1, const Real_t x2, const Real_t y2,
                  const Real_t z2, const Real_t x3, const Real_t y3, const Real_t z3)
{
    Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
    Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
    Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
    Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
    Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
    Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
    Real_t areaX    = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
    Real_t areaY    = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
    Real_t areaZ    = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

    *normalX0 += areaX;
    *normalX1 += areaX;
    *normalX2 += areaX;
    *normalX3 += areaX;

    *normalY0 += areaY;
    *normalY1 += areaY;
    *normalY2 += areaY;
    *normalY3 += areaY;

    *normalZ0 += areaZ;
    *normalZ1 += areaZ;
    *normalZ2 += areaZ;
    *normalZ3 += areaZ;
}

KOKKOS_INLINE_FUNCTION void
CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8], Real_t pfz[8], const Real_t x[8],
                    const Real_t y[8], const Real_t z[8])
{
    for(Index_t i = 0; i < 8; ++i)
    {
        pfx[i] = Real_t(0.0);
        pfy[i] = Real_t(0.0);
        pfz[i] = Real_t(0.0);
    }

    SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[1], &pfy[1], &pfz[1], &pfx[2],
                      &pfy[2], &pfz[2], &pfx[3], &pfy[3], &pfz[3], x[0], y[0], z[0], x[1],
                      y[1], z[1], x[2], y[2], z[2], x[3], y[3], z[3]);

    SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[4], &pfy[4], &pfz[4], &pfx[5],
                      &pfy[5], &pfz[5], &pfx[1], &pfy[1], &pfz[1], x[0], y[0], z[0], x[4],
                      y[4], z[4], x[5], y[5], z[5], x[1], y[1], z[1]);

    SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1], &pfx[5], &pfy[5], &pfz[5], &pfx[6],
                      &pfy[6], &pfz[6], &pfx[2], &pfy[2], &pfz[2], x[1], y[1], z[1], x[5],
                      y[5], z[5], x[6], y[6], z[6], x[2], y[2], z[2]);

    SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2], &pfx[6], &pfy[6], &pfz[6], &pfx[7],
                      &pfy[7], &pfz[7], &pfx[3], &pfy[3], &pfz[3], x[2], y[2], z[2], x[6],
                      y[6], z[6], x[7], y[7], z[7], x[3], y[3], z[3]);

    SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3], &pfx[7], &pfy[7], &pfz[7], &pfx[4],
                      &pfy[4], &pfz[4], &pfx[0], &pfy[0], &pfz[0], x[3], y[3], z[3], x[7],
                      y[7], z[7], x[4], y[4], z[4], x[0], y[0], z[0]);

    SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4], &pfx[7], &pfy[7], &pfz[7], &pfx[6],
                      &pfy[6], &pfz[6], &pfx[5], &pfy[5], &pfz[5], x[4], y[4], z[4], x[7],
                      y[7], z[7], x[6], y[6], z[6], x[5], y[5], z[5]);
}

KOKKOS_INLINE_FUNCTION void
SumElemStressesToNodeForces(const Real_t B[][8], const Real_t stress_xx,
                            const Real_t stress_yy, const Real_t stress_zz, Real_t fx[],
                            Real_t fy[], Real_t fz[])
{
    for(Index_t i = 0; i < 8; i++)
    {
        fx[i] = -(stress_xx * B[0][i]);
        fy[i] = -(stress_yy * B[1][i]);
        fz[i] = -(stress_zz * B[2][i]);
    }
}

static inline void
IntegrateStressForElems(Domain& domain, Real_t* sigxx, Real_t* sigyy, Real_t* sigzz,
                        Real_t* determ, Index_t numElem, Index_t numNode)
{
    Index_t numElem8 = numElem * 8;
    ResizeBuffer((numElem8 * sizeof(Real_t) + 4096) * 3);
    Real_t* fx_elem = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* fy_elem = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* fz_elem = AllocateFromBuffer<Real_t>(numElem8);

    Kokkos::parallel_for(
        "IntegrateStressForElems A", numElem, KOKKOS_LAMBDA(const int k) {
            const Index_t* const elemToNode = &domain.nodelist(k, 0);
            Real_t               B[3][8];
            Real_t               x_local[8];
            Real_t               y_local[8];
            Real_t               z_local[8];

            CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

            CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);

            CalcElemNodeNormals(B[0], B[1], B[2], x_local, y_local, z_local);

            SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k], &fx_elem[k * 8],
                                        &fy_elem[k * 8], &fz_elem[k * 8]);
        });

    int team_size = 1;
    if(Kokkos::DefaultExecutionSpace().concurrency() > 1024) team_size = 128;

    Kokkos::parallel_for(
        "IntegrateStressForElems B",
        Kokkos::TeamPolicy<>((numNode + 127) / 128, team_size, 2),
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
            const Index_t gnode_begin = team.league_rank() * 128;
            const Index_t gnode_end =
                (gnode_begin + 128 < numNode) ? gnode_begin + 128 : numNode;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, gnode_begin, gnode_end),
                [&](const Index_t& gnode) {
                    Index_t        count      = domain.nodeElemCount(gnode);
                    Index_t*       cornerList = domain.nodeElemCornerList(gnode);
                    reduce_double3 f_tmp;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, count),
                        [&](const Index_t&  i,
                            reduce_double3& tmp) {  // vectorized with ivdep
                            Index_t elem = cornerList[i];
                            tmp.x += fx_elem[elem];
                            tmp.y += fy_elem[elem];
                            tmp.z += fz_elem[elem];
                        },
                        f_tmp);
                    Kokkos::single(Kokkos::PerThread(team), [&]() {
                        domain.fx(gnode) += f_tmp.x;
                        domain.fy(gnode) += f_tmp.y;
                        domain.fz(gnode) += f_tmp.z;
                    });
                });
        });
}

KOKKOS_INLINE_FUNCTION void
VoluDer(const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
        const Real_t x4, const Real_t x5, const Real_t y0, const Real_t y1,
        const Real_t y2, const Real_t y3, const Real_t y4, const Real_t y5,
        const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
        const Real_t z4, const Real_t z5, Real_t& dvdx, Real_t& dvdy, Real_t& dvdz)
{
    const Real_t twelfth = Real_t(1.0) / Real_t(12.0);

    dvdx = (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) + (y0 + y4) * (z3 + z4) -
           (y3 + y4) * (z0 + z4) - (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
    dvdy = -(x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) - (x0 + x4) * (z3 + z4) +
           (x3 + x4) * (z0 + z4) + (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

    dvdz = -(y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) - (y0 + y4) * (x3 + x4) +
           (y3 + y4) * (x0 + x4) + (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

    dvdx *= twelfth;
    dvdy *= twelfth;
    dvdz *= twelfth;
}

KOKKOS_INLINE_FUNCTION
void
CalcElemVolumeDerivative(
    const Int_t&                                                           i,
    const Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>>& dvdx,
    const Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>>& dvdy,
    const Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>>& dvdz,
    const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
#pragma nounroll
    for(int j = 0; j < 4; j++)
    {
        VoluDer(x[(j + 1) % 4], x[(j + 2) % 4], x[(j + 3) % 4], x[(j + 0) % 4 + 4],
                x[(j + 1) % 4 + 4], x[(j + 3) % 4 + 4], y[(j + 1) % 4], y[(j + 2) % 4],
                y[(j + 3) % 4], y[(j + 0) % 4 + 4], y[(j + 1) % 4 + 4],
                y[(j + 3) % 4 + 4], z[(j + 1) % 4], z[(j + 2) % 4], z[(j + 3) % 4],
                z[(j + 0) % 4 + 4], z[(j + 1) % 4 + 4], z[(j + 3) % 4 + 4], dvdx(i, j),
                dvdy(i, j), dvdz(i, j));
        VoluDer(x[(j + 3) % 4 + 4], x[(j + 2) % 4 + 4], x[(j + 1) % 4 + 4],
                x[(j + 0) % 4], x[(j + 3) % 4], x[(j + 1) % 4], y[(j + 3) % 4 + 4],
                y[(j + 2) % 4 + 4], y[(j + 1) % 4 + 4], y[(j + 0) % 4], y[(j + 3) % 4],
                y[(j + 1) % 4], z[(j + 3) % 4 + 4], z[(j + 2) % 4 + 4],
                z[(j + 1) % 4 + 4], z[(j + 0) % 4], z[(j + 3) % 4], z[(j + 1) % 4],
                dvdx(i, j + 4), dvdy(i, j + 4), dvdz(i, j + 4));
    }
}

KOKKOS_INLINE_FUNCTION
void
CalcElemFBHourglassForce(const Real_t* xd, const Real_t hourgam[][8],
                         const Real_t& coefficient, Real_t* hgfx)
{
    Real_t hxx[4];
    for(Index_t i = 0; i < 4; i++)
    {
        hxx[i] = hourgam[i][0] * xd[0] + hourgam[i][1] * xd[1] + hourgam[i][2] * xd[2] +
                 hourgam[i][3] * xd[3] + hourgam[i][4] * xd[4] + hourgam[i][5] * xd[5] +
                 hourgam[i][6] * xd[6] + hourgam[i][7] * xd[7];
    }
    for(Index_t i = 0; i < 8; i++)
    {
        hgfx[i] = coefficient * (hourgam[0][i] * hxx[0] + hourgam[1][i] * hxx[1] +
                                 hourgam[2][i] * hxx[2] + hourgam[3][i] * hxx[3]);
    }
}

struct Gamma
{
    Real_t gamma[4][8];
    Gamma()
    {
        gamma[0][0] = Real_t(1.);
        gamma[0][1] = Real_t(1.);
        gamma[0][2] = Real_t(-1.);
        gamma[0][3] = Real_t(-1.);
        gamma[0][4] = Real_t(-1.);
        gamma[0][5] = Real_t(-1.);
        gamma[0][6] = Real_t(1.);
        gamma[0][7] = Real_t(1.);
        gamma[1][0] = Real_t(1.);
        gamma[1][1] = Real_t(-1.);
        gamma[1][2] = Real_t(-1.);
        gamma[1][3] = Real_t(1.);
        gamma[1][4] = Real_t(-1.);
        gamma[1][5] = Real_t(1.);
        gamma[1][6] = Real_t(1.);
        gamma[1][7] = Real_t(-1.);
        gamma[2][0] = Real_t(1.);
        gamma[2][1] = Real_t(-1.);
        gamma[2][2] = Real_t(1.);
        gamma[2][3] = Real_t(-1.);
        gamma[2][4] = Real_t(1.);
        gamma[2][5] = Real_t(-1.);
        gamma[2][6] = Real_t(1.);
        gamma[2][7] = Real_t(-1.);
        gamma[3][0] = Real_t(-1.);
        gamma[3][1] = Real_t(1.);
        gamma[3][2] = Real_t(-1.);
        gamma[3][3] = Real_t(1.);
        gamma[3][4] = Real_t(1.);
        gamma[3][5] = Real_t(-1.);
        gamma[3][6] = Real_t(1.);
        gamma[3][7] = Real_t(-1.);
    }
};

static inline void
CalcFBHourglassForceForElems(
    Domain& domain, Real_t* determ,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> x8n,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y8n,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> z8n,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dvdx,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dvdy,
    const Kokkos::View<const Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dvdz,
    Real_t hourg, Index_t numElem, Index_t numNode)
{
    Index_t numElem8 = numElem * 8;

    Real_t* fx_elem;
    Real_t* fy_elem;
    Real_t* fz_elem;

    if(do_atomic == 0)
    {
        fx_elem = AllocateFromBuffer<Real_t>(numElem8);
        fy_elem = AllocateFromBuffer<Real_t>(numElem8);
        fz_elem = AllocateFromBuffer<Real_t>(numElem8);
    }

    Gamma G;

    Int_t do_atomic_dev = do_atomic;

    Kokkos::parallel_for(
        "CalcFBHourglassForceForElems A", numElem, KOKKOS_LAMBDA(const int& i2) {
            Real_t *fx_local, *fy_local, *fz_local;
            Real_t  hgfx[8];

            Real_t hourgam[4][8];
            Real_t xd1[8];

            const Index_t* elemToNode = &domain.nodelist(i2, 0);
            Index_t        i3         = 8 * i2;
            Real_t         volinv     = Real_t(1.0) / determ[i2];

            for(Index_t i1 = 0; i1 < 4; ++i1)
            {
                Real_t hourmodx = 0.0;
                for(int j = 0; j < 8; j++)
                    hourmodx += x8n(i2, j) * G.gamma[i1][j];

                Real_t hourmody = 0.0;
                for(int j = 0; j < 8; j++)
                    hourmody += y8n(i2, j) * G.gamma[i1][j];

                Real_t hourmodz = 0.0;
                for(int j = 0; j < 8; j++)
                    hourmodz += z8n(i2, j) * G.gamma[i1][j];

#pragma ivdep
                for(int j = 0; j < 8; j++)
                    hourgam[i1][j] = G.gamma[i1][j] - volinv * (dvdx(i2, j) * hourmodx +
                                                                dvdy(i2, j) * hourmody +
                                                                dvdz(i2, j) * hourmodz);
            }

            const Real_t ss1      = domain.ss(i2);
            const Real_t mass1    = domain.elemMass(i2);
            const Real_t volume13 = CBRT(determ[i2]);

            const Index_t n0si2 = elemToNode[0];
            const Index_t n1si2 = elemToNode[1];
            const Index_t n2si2 = elemToNode[2];
            const Index_t n3si2 = elemToNode[3];
            const Index_t n4si2 = elemToNode[4];
            const Index_t n5si2 = elemToNode[5];
            const Index_t n6si2 = elemToNode[6];
            const Index_t n7si2 = elemToNode[7];

            const Real_t coefficient = -hourg * Real_t(0.01) * ss1 * mass1 / volume13;

            xd1[0] = domain.xd(n0si2);
            xd1[1] = domain.xd(n1si2);
            xd1[2] = domain.xd(n2si2);
            xd1[3] = domain.xd(n3si2);
            xd1[4] = domain.xd(n4si2);
            xd1[5] = domain.xd(n5si2);
            xd1[6] = domain.xd(n6si2);
            xd1[7] = domain.xd(n7si2);

            CalcElemFBHourglassForce(xd1, hourgam, coefficient, hgfx);

            if(!do_atomic_dev)
            {
                fx_local    = &fx_elem[i3];
                fx_local[0] = hgfx[0];
                fx_local[1] = hgfx[1];
                fx_local[2] = hgfx[2];
                fx_local[3] = hgfx[3];
                fx_local[4] = hgfx[4];
                fx_local[5] = hgfx[5];
                fx_local[6] = hgfx[6];
                fx_local[7] = hgfx[7];
            }
            else
            {
                Kokkos::atomic_add(&domain.fx(n0si2), hgfx[0]);
                Kokkos::atomic_add(&domain.fx(n1si2), hgfx[1]);
                Kokkos::atomic_add(&domain.fx(n2si2), hgfx[2]);
                Kokkos::atomic_add(&domain.fx(n3si2), hgfx[3]);
                Kokkos::atomic_add(&domain.fx(n4si2), hgfx[4]);
                Kokkos::atomic_add(&domain.fx(n5si2), hgfx[5]);
                Kokkos::atomic_add(&domain.fx(n6si2), hgfx[6]);
                Kokkos::atomic_add(&domain.fx(n7si2), hgfx[7]);
            }

            xd1[0] = domain.yd(n0si2);
            xd1[1] = domain.yd(n1si2);
            xd1[2] = domain.yd(n2si2);
            xd1[3] = domain.yd(n3si2);
            xd1[4] = domain.yd(n4si2);
            xd1[5] = domain.yd(n5si2);
            xd1[6] = domain.yd(n6si2);
            xd1[7] = domain.yd(n7si2);

            CalcElemFBHourglassForce(xd1, hourgam, coefficient, hgfx);

            if(!do_atomic_dev)
            {
                fy_local    = &fy_elem[i3];
                fy_local[0] = hgfx[0];
                fy_local[1] = hgfx[1];
                fy_local[2] = hgfx[2];
                fy_local[3] = hgfx[3];
                fy_local[4] = hgfx[4];
                fy_local[5] = hgfx[5];
                fy_local[6] = hgfx[6];
                fy_local[7] = hgfx[7];
            }
            else
            {
                Kokkos::atomic_add(&domain.fy(n0si2), hgfx[0]);
                Kokkos::atomic_add(&domain.fy(n1si2), hgfx[1]);
                Kokkos::atomic_add(&domain.fy(n2si2), hgfx[2]);
                Kokkos::atomic_add(&domain.fy(n3si2), hgfx[3]);
                Kokkos::atomic_add(&domain.fy(n4si2), hgfx[4]);
                Kokkos::atomic_add(&domain.fy(n5si2), hgfx[5]);
                Kokkos::atomic_add(&domain.fy(n6si2), hgfx[6]);
                Kokkos::atomic_add(&domain.fy(n7si2), hgfx[7]);
            }

            xd1[0] = domain.zd(n0si2);
            xd1[1] = domain.zd(n1si2);
            xd1[2] = domain.zd(n2si2);
            xd1[3] = domain.zd(n3si2);
            xd1[4] = domain.zd(n4si2);
            xd1[5] = domain.zd(n5si2);
            xd1[6] = domain.zd(n6si2);
            xd1[7] = domain.zd(n7si2);

            CalcElemFBHourglassForce(xd1, hourgam, coefficient, hgfx);

            if(!do_atomic_dev)
            {
                fz_local    = &fz_elem[i3];
                fz_local[0] = hgfx[0];
                fz_local[1] = hgfx[1];
                fz_local[2] = hgfx[2];
                fz_local[3] = hgfx[3];
                fz_local[4] = hgfx[4];
                fz_local[5] = hgfx[5];
                fz_local[6] = hgfx[6];
                fz_local[7] = hgfx[7];
            }
            else
            {
                Kokkos::atomic_add(&domain.fz(n0si2), hgfx[0]);
                Kokkos::atomic_add(&domain.fz(n1si2), hgfx[1]);
                Kokkos::atomic_add(&domain.fz(n2si2), hgfx[2]);
                Kokkos::atomic_add(&domain.fz(n3si2), hgfx[3]);
                Kokkos::atomic_add(&domain.fz(n4si2), hgfx[4]);
                Kokkos::atomic_add(&domain.fz(n5si2), hgfx[5]);
                Kokkos::atomic_add(&domain.fz(n6si2), hgfx[6]);
                Kokkos::atomic_add(&domain.fz(n7si2), hgfx[7]);
            }
        });

    if(!do_atomic)
    {
        int team_size = 1;
        if(Kokkos::DefaultExecutionSpace().concurrency() > 1024) team_size = 128;

        Kokkos::parallel_for(
            "CalcFBHourglassForceForElems B",
            Kokkos::TeamPolicy<>((numNode + 127) / 128, team_size, 2),
            KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
                const Index_t gnode_begin = team.league_rank() * 128;
                const Index_t gnode_end =
                    (gnode_begin + 128 < numNode) ? gnode_begin + 128 : numNode;

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, gnode_begin, gnode_end),
                    [&](const Index_t& gnode) {
                        Index_t        count      = domain.nodeElemCount(gnode);
                        Index_t*       cornerList = domain.nodeElemCornerList(gnode);
                        reduce_double3 f_tmp;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, count),
                            [&](const Index_t&  i,
                                reduce_double3& tmp) {  // vectorized with ivdep
                                Index_t elem = cornerList[i];
                                tmp.x += fx_elem[elem];
                                tmp.y += fy_elem[elem];
                                tmp.z += fz_elem[elem];
                            },
                            f_tmp);
                        Kokkos::single(Kokkos::PerThread(team), [&]() {
                            domain.fx(gnode) += f_tmp.x;
                            domain.fy(gnode) += f_tmp.y;
                            domain.fz(gnode) += f_tmp.z;
                        });
                    });
            });
    }
}

static inline void
CalcHourglassControlForElems(Domain& domain, Real_t determ[], Real_t hgcoef)
{
    Index_t numElem  = domain.numElem();
    Index_t numElem8 = numElem * 8;
    ResizeBuffer((numElem8 * sizeof(Real_t) + 4096) * (do_atomic ? 6 : 9));

    Real_t* dvdx = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* dvdy = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* dvdz = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* x8n  = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* y8n  = AllocateFromBuffer<Real_t>(numElem8);
    Real_t* z8n  = AllocateFromBuffer<Real_t>(numElem8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_x8n(x8n, numElem,
                                                                          8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_y8n(y8n, numElem,
                                                                          8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_z8n(z8n, numElem,
                                                                          8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_dvdx(dvdx, numElem,
                                                                           8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_dvdy(dvdy, numElem,
                                                                           8);
    Kokkos::View<Real_t**, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_dvdz(dvdz, numElem,
                                                                           8);

    int error = 0;
    Kokkos::parallel_reduce(
        "CalcHourglassControlForElems", numElem,
        KOKKOS_LAMBDA(const int i, int& err) {
            Real_t x1[8], y1[8], z1[8];

            Index_t* elemToNode = &domain.nodelist(i, 0);
            CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

            CalcElemVolumeDerivative(i, v_dvdx, v_dvdy, v_dvdz, x1, y1, z1);

            for(Index_t ii = 0; ii < 8; ++ii)
            {
                v_x8n(i, ii) = x1[ii];
                v_y8n(i, ii) = y1[ii];
                v_z8n(i, ii) = z1[ii];
            }

            determ[i] = domain.volo(i) * domain.v(i);

            if(domain.v(i) <= Real_t(0.0))
            {
                err++;
            }
        },
        error);

    if(error)
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
        exit(VolumeError);
#endif

    if(hgcoef > Real_t(0.))
    {
        CalcFBHourglassForceForElems(domain, determ, v_x8n, v_y8n, v_z8n, v_dvdx, v_dvdy,
                                     v_dvdz, hgcoef, numElem, domain.numNode());
    }

    return;
}

static inline void
CalcVolumeForceForElems(Domain& domain)
{
    Index_t numElem = domain.numElem();
    if(numElem != 0)
    {
        Real_t                hgcoef = domain.hgcoef();
        Kokkos::View<Real_t*> sigxx("sigxx", numElem);
        Kokkos::View<Real_t*> sigyy("sigyy", numElem);
        Kokkos::View<Real_t*> sigzz("sigzz", numElem);
        Kokkos::View<Real_t*> determ("determ", numElem);

        InitStressTermsForElems(domain, sigxx.data(), sigyy.data(), sigzz.data(),
                                numElem);

        IntegrateStressForElems(domain, sigxx.data(), sigyy.data(), sigzz.data(),
                                determ.data(), numElem, domain.numNode());

        // check for negative element volume
        int error = 0;
        Kokkos::parallel_reduce(
            "CalcVolumeForceForElems", numElem,
            KOKKOS_LAMBDA(const int k, int& err) {
                if(determ[k] <= Real_t(0.0))
                {
                    err++;
                }
            },
            error);

        if(error)
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
            exit(VolumeError);
#endif

        CalcHourglassControlForElems(domain, determ.data(), hgcoef);
    }
}

static inline void
CalcForceForNodes(Domain& domain)
{
    Index_t numNode = domain.numNode();

#if USE_MPI
    CommRecv(domain, MSG_COMM_SBN, 3, domain.sizeX() + 1, domain.sizeY() + 1,
             domain.sizeZ() + 1, true, false);
#endif

    Kokkos::parallel_for(
        "CalcForceForNodes", numNode, KOKKOS_LAMBDA(const int i) {
            domain.fx(i) = Real_t(0.0);
            domain.fy(i) = Real_t(0.0);
            domain.fz(i) = Real_t(0.0);
        });

    CalcVolumeForceForElems(domain);

#if USE_MPI
    Domain_member fieldData[3];
    fieldData[0] = &Domain::fx;
    fieldData[1] = &Domain::fy;
    fieldData[2] = &Domain::fz;

    CommSend(domain, MSG_COMM_SBN, 3, fieldData, domain.sizeX() + 1, domain.sizeY() + 1,
             domain.sizeZ() + 1, true, false);
    CommSBN(domain, 3, fieldData);
#endif
}

static inline void
CalcAccelerationForNodes(Domain& domain, Index_t numNode)
{
    Kokkos::parallel_for(
        "CalcAccelerationForNodes", numNode, KOKKOS_LAMBDA(const int i) {
            domain.xdd(i) = domain.fx(i) / domain.nodalMass(i);
            domain.ydd(i) = domain.fy(i) / domain.nodalMass(i);
            domain.zdd(i) = domain.fz(i) / domain.nodalMass(i);
        });
}

static inline void
ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
{
    Index_t size      = domain.sizeX();
    Index_t numNodeBC = (size + 1) * (size + 1);

    if(!domain.symmXempty() != 0)
    {
        Kokkos::parallel_for(
            "ApplyAccelerationBoundaryConditionsForNodes A", numNodeBC,
            KOKKOS_LAMBDA(const int i) { domain.xdd(domain.symmX(i)) = Real_t(0.0); });
    }

    if(!domain.symmYempty() != 0)
    {
        Kokkos::parallel_for(
            "ApplyAccelerationBoundaryConditionsForNodes B", numNodeBC,
            KOKKOS_LAMBDA(const int i) { domain.ydd(domain.symmY(i)) = Real_t(0.0); });
    }

    if(!domain.symmZempty() != 0)
    {
        Kokkos::parallel_for(
            "ApplyAccelerationBoundaryConditionsForNodes C", numNodeBC,
            KOKKOS_LAMBDA(const int i) { domain.zdd(domain.symmZ(i)) = Real_t(0.0); });
    }
}

static inline void
CalcVelocityForNodes(Domain& domain, const Real_t dt, const Real_t u_cut, Index_t numNode)
{
    Kokkos::parallel_for(
        "CalcVelocityForNodes", numNode, KOKKOS_LAMBDA(const int i) {
            Real_t xdtmp, ydtmp, zdtmp;

            xdtmp = domain.xd(i) + domain.xdd(i) * dt;
            if(FABS(xdtmp) < u_cut) xdtmp = Real_t(0.0);
            domain.xd(i) = xdtmp;

            ydtmp = domain.yd(i) + domain.ydd(i) * dt;
            if(FABS(ydtmp) < u_cut) ydtmp = Real_t(0.0);
            domain.yd(i) = ydtmp;

            zdtmp = domain.zd(i) + domain.zdd(i) * dt;
            if(FABS(zdtmp) < u_cut) zdtmp = Real_t(0.0);
            domain.zd(i) = zdtmp;
        });
}

static inline void
CalcPositionForNodes(Domain& domain, const Real_t dt, Index_t numNode)
{
    Kokkos::parallel_for(
        "CalcPositionForNodes", numNode, KOKKOS_LAMBDA(const int i) {
            domain.x(i) += domain.xd(i) * dt;
            domain.y(i) += domain.yd(i) * dt;
            domain.z(i) += domain.zd(i) * dt;
        });
}

static inline void
LagrangeNodal(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
    Domain_member fieldData[6];
#endif

    const Real_t delt  = domain.deltatime();
    Real_t       u_cut = domain.u_cut();

    CalcForceForNodes(domain);

#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_EARLY
    CommRecv(domain, MSG_SYNC_POS_VEL, 6, domain.sizeX() + 1, domain.sizeY() + 1,
             domain.sizeZ() + 1, false, false);
#    endif
#endif

    CalcAccelerationForNodes(domain, domain.numNode());

    ApplyAccelerationBoundaryConditionsForNodes(domain);

    CalcVelocityForNodes(domain, delt, u_cut, domain.numNode());

    CalcPositionForNodes(domain, delt, domain.numNode());
#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_EARLY
    fieldData[0] = &Domain::x;
    fieldData[1] = &Domain::y;
    fieldData[2] = &Domain::z;
    fieldData[3] = &Domain::xd;
    fieldData[4] = &Domain::yd;
    fieldData[5] = &Domain::zd;

    CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData, domain.sizeX() + 1,
             domain.sizeY() + 1, domain.sizeZ() + 1, false, false);
    CommSyncPosVel(domain);
#    endif
#endif

    return;
}

KOKKOS_INLINE_FUNCTION Real_t
CalcElemVolume(const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7)
{
    Real_t twelveth = Real_t(1.0) / Real_t(12.0);

    Real_t dx61 = x6 - x1;
    Real_t dy61 = y6 - y1;
    Real_t dz61 = z6 - z1;

    Real_t dx70 = x7 - x0;
    Real_t dy70 = y7 - y0;
    Real_t dz70 = z7 - z0;

    Real_t dx63 = x6 - x3;
    Real_t dy63 = y6 - y3;
    Real_t dz63 = z6 - z3;

    Real_t dx20 = x2 - x0;
    Real_t dy20 = y2 - y0;
    Real_t dz20 = z2 - z0;

    Real_t dx50 = x5 - x0;
    Real_t dy50 = y5 - y0;
    Real_t dz50 = z5 - z0;

    Real_t dx64 = x6 - x4;
    Real_t dy64 = y6 - y4;
    Real_t dz64 = z6 - z4;

    Real_t dx31 = x3 - x1;
    Real_t dy31 = y3 - y1;
    Real_t dz31 = z3 - z1;

    Real_t dx72 = x7 - x2;
    Real_t dy72 = y7 - y2;
    Real_t dz72 = z7 - z2;

    Real_t dx43 = x4 - x3;
    Real_t dy43 = y4 - y3;
    Real_t dz43 = z4 - z3;

    Real_t dx57 = x5 - x7;
    Real_t dy57 = y5 - y7;
    Real_t dz57 = z5 - z7;

    Real_t dx14 = x1 - x4;
    Real_t dy14 = y1 - y4;
    Real_t dz14 = z1 - z4;

    Real_t dx25 = x2 - x5;
    Real_t dy25 = y2 - y5;
    Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3)                               \
    ((x1) * ((y2) * (z3) - (z2) * (y3)) + (x2) * ((z1) * (y3) - (y1) * (z3)) +           \
     (x3) * ((y1) * (z2) - (z1) * (y2)))

    Real_t volume = TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, dy31 + dy72, dy63, dy20,
                                   dz31 + dz72, dz63, dz20) +
                    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, dy43 + dy57, dy64, dy70,
                                   dz43 + dz57, dz64, dz70) +
                    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, dy14 + dy25, dy61, dy50,
                                   dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

    volume *= twelveth;

    return volume;
}

KOKKOS_INLINE_FUNCTION
Real_t
CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
    return CalcElemVolume(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], y[0], y[1],
                          y[2], y[3], y[4], y[5], y[6], y[7], z[0], z[1], z[2], z[3],
                          z[4], z[5], z[6], z[7]);
}

KOKKOS_INLINE_FUNCTION
Real_t
AreaFace(const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
         const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
         const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3)
{
    Real_t fx   = (x2 - x0) - (x3 - x1);
    Real_t fy   = (y2 - y0) - (y3 - y1);
    Real_t fz   = (z2 - z0) - (z3 - z1);
    Real_t gx   = (x2 - x0) + (x3 - x1);
    Real_t gy   = (y2 - y0) + (y3 - y1);
    Real_t gz   = (z2 - z0) + (z3 - z1);
    Real_t area = (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -
                  (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);
    return area;
}

KOKKOS_INLINE_FUNCTION Real_t
CalcElemCharacteristicLength(const Real_t x[8], const Real_t y[8], const Real_t z[8],
                             const Real_t volume)
{
    Real_t a, charLength = Real_t(0.0);

    a = AreaFace(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7], z[4], z[5], z[6], z[7]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[0], x[1], x[5], x[4], y[0], y[1], y[5], y[4], z[0], z[1], z[5], z[4]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[1], x[2], x[6], x[5], y[1], y[2], y[6], y[5], z[1], z[2], z[6], z[5]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[2], x[3], x[7], x[6], y[2], y[3], y[7], y[6], z[2], z[3], z[7], z[6]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[3], x[0], x[4], x[7], y[3], y[0], y[4], y[7], z[3], z[0], z[4], z[7]);
    charLength = MAX(a, charLength);

    charLength = Real_t(4.0) * volume / SQRT(charLength);

    return charLength;
}

KOKKOS_INLINE_FUNCTION void
CalcElemVelocityGradient(const Real_t* const xvel, const Real_t* const yvel,
                         const Real_t* const zvel, const Real_t b[][8], const Real_t detJ,
                         Real_t* const d)
{
    const Real_t        inv_detJ = Real_t(1.0) / detJ;
    Real_t              dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
    const Real_t* const pfx = b[0];
    const Real_t* const pfy = b[1];
    const Real_t* const pfz = b[2];

    d[0] = inv_detJ * (pfx[0] * (xvel[0] - xvel[6]) + pfx[1] * (xvel[1] - xvel[7]) +
                       pfx[2] * (xvel[2] - xvel[4]) + pfx[3] * (xvel[3] - xvel[5]));

    d[1] = inv_detJ * (pfy[0] * (yvel[0] - yvel[6]) + pfy[1] * (yvel[1] - yvel[7]) +
                       pfy[2] * (yvel[2] - yvel[4]) + pfy[3] * (yvel[3] - yvel[5]));

    d[2] = inv_detJ * (pfz[0] * (zvel[0] - zvel[6]) + pfz[1] * (zvel[1] - zvel[7]) +
                       pfz[2] * (zvel[2] - zvel[4]) + pfz[3] * (zvel[3] - zvel[5]));

    dyddx = inv_detJ * (pfx[0] * (yvel[0] - yvel[6]) + pfx[1] * (yvel[1] - yvel[7]) +
                        pfx[2] * (yvel[2] - yvel[4]) + pfx[3] * (yvel[3] - yvel[5]));

    dxddy = inv_detJ * (pfy[0] * (xvel[0] - xvel[6]) + pfy[1] * (xvel[1] - xvel[7]) +
                        pfy[2] * (xvel[2] - xvel[4]) + pfy[3] * (xvel[3] - xvel[5]));

    dzddx = inv_detJ * (pfx[0] * (zvel[0] - zvel[6]) + pfx[1] * (zvel[1] - zvel[7]) +
                        pfx[2] * (zvel[2] - zvel[4]) + pfx[3] * (zvel[3] - zvel[5]));

    dxddz = inv_detJ * (pfz[0] * (xvel[0] - xvel[6]) + pfz[1] * (xvel[1] - xvel[7]) +
                        pfz[2] * (xvel[2] - xvel[4]) + pfz[3] * (xvel[3] - xvel[5]));

    dzddy = inv_detJ * (pfy[0] * (zvel[0] - zvel[6]) + pfy[1] * (zvel[1] - zvel[7]) +
                        pfy[2] * (zvel[2] - zvel[4]) + pfy[3] * (zvel[3] - zvel[5]));

    dyddz = inv_detJ * (pfz[0] * (yvel[0] - yvel[6]) + pfz[1] * (yvel[1] - yvel[7]) +
                        pfz[2] * (yvel[2] - yvel[4]) + pfz[3] * (yvel[3] - yvel[5]));
    d[5]  = Real_t(.5) * (dxddy + dyddx);
    d[4]  = Real_t(.5) * (dxddz + dzddx);
    d[3]  = Real_t(.5) * (dzddy + dyddz);
}

void
CalcKinematicsForElems(Domain& domain, Real_t deltaTime, Index_t numElem)
{
    Kokkos::parallel_for(
        "CalcKinematicsForElems", numElem, KOKKOS_LAMBDA(const int k) {
            Real_t B[3][8];
            Real_t D[6];
            Real_t x_local[8];
            Real_t y_local[8];
            Real_t z_local[8];
            Real_t xd_local[8];
            Real_t yd_local[8];
            Real_t zd_local[8];
            Real_t detJ = Real_t(0.0);

            Real_t               volume;
            Real_t               relativeVolume;
            const Index_t* const elemToNode = &domain.nodelist(k, 0);

            CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

            volume         = CalcElemVolume(x_local, y_local, z_local);
            relativeVolume = volume / domain.volo(k);
            domain.vnew(k) = relativeVolume;
            domain.delv(k) = relativeVolume - domain.v(k);

            domain.arealg(k) =
                CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

            for(Index_t lnode = 0; lnode < 8; ++lnode)
            {
                Index_t gnode   = elemToNode[lnode];
                xd_local[lnode] = domain.c_xd(gnode);
                yd_local[lnode] = domain.c_yd(gnode);
                zd_local[lnode] = domain.c_zd(gnode);
            }

            Real_t dt2 = Real_t(0.5) * deltaTime;
            for(Index_t j = 0; j < 8; ++j)
            {
                x_local[j] -= dt2 * xd_local[j];
                y_local[j] -= dt2 * yd_local[j];
                z_local[j] -= dt2 * zd_local[j];
            }

            CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);

            CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);

            domain.dxx(k) = D[0];
            domain.dyy(k) = D[1];
            domain.dzz(k) = D[2];
        });
}

static inline void
CalcLagrangeElements(Domain& domain)
{
    Index_t numElem = domain.numElem();
    if(numElem > 0)
    {
        const Real_t deltatime = domain.deltatime();

        domain.AllocateStrains(numElem);

        CalcKinematicsForElems(domain, deltatime, numElem);

        int error = 0;
        Kokkos::parallel_reduce(
            "CalcLagrangeElements", numElem,
            KOKKOS_LAMBDA(const int k, int& err) {
                Real_t vdov      = domain.dxx(k) + domain.dyy(k) + domain.dzz(k);
                Real_t vdovthird = vdov / Real_t(3.0);

                domain.vdov(k) = vdov;
                domain.dxx(k) -= vdovthird;
                domain.dyy(k) -= vdovthird;
                domain.dzz(k) -= vdovthird;

                if(domain.vnew(k) <= Real_t(0.0))
                {
                    err++;
                }
            },
            error);

        if(error)
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
            exit(VolumeError);
#endif

        domain.DeallocateStrains();
    }
}

static inline void
CalcMonotonicQGradientsForElems(Domain& domain)
{
    Index_t numElem = domain.numElem();

    Kokkos::parallel_for(
        "CalcMonotonicQGradientsForElems", numElem, KOKKOS_LAMBDA(const int i) {
            const Real_t ptiny = Real_t(1.e-36);
            Real_t       ax, ay, az;
            Real_t       dxv, dyv, dzv;

            const Index_t* elemToNode = &domain.nodelist(i, 0);
            Index_t        n0         = elemToNode[0];
            Index_t        n1         = elemToNode[1];
            Index_t        n2         = elemToNode[2];
            Index_t        n3         = elemToNode[3];
            Index_t        n4         = elemToNode[4];
            Index_t        n5         = elemToNode[5];
            Index_t        n6         = elemToNode[6];
            Index_t        n7         = elemToNode[7];

            Real_t x0 = domain.x(n0);
            Real_t x1 = domain.x(n1);
            Real_t x2 = domain.x(n2);
            Real_t x3 = domain.x(n3);
            Real_t x4 = domain.x(n4);
            Real_t x5 = domain.x(n5);
            Real_t x6 = domain.x(n6);
            Real_t x7 = domain.x(n7);

            Real_t y0 = domain.y(n0);
            Real_t y1 = domain.y(n1);
            Real_t y2 = domain.y(n2);
            Real_t y3 = domain.y(n3);
            Real_t y4 = domain.y(n4);
            Real_t y5 = domain.y(n5);
            Real_t y6 = domain.y(n6);
            Real_t y7 = domain.y(n7);

            Real_t z0 = domain.z(n0);
            Real_t z1 = domain.z(n1);
            Real_t z2 = domain.z(n2);
            Real_t z3 = domain.z(n3);
            Real_t z4 = domain.z(n4);
            Real_t z5 = domain.z(n5);
            Real_t z6 = domain.z(n6);
            Real_t z7 = domain.z(n7);

            Real_t xv0 = domain.xd(n0);
            Real_t xv1 = domain.xd(n1);
            Real_t xv2 = domain.xd(n2);
            Real_t xv3 = domain.xd(n3);
            Real_t xv4 = domain.xd(n4);
            Real_t xv5 = domain.xd(n5);
            Real_t xv6 = domain.xd(n6);
            Real_t xv7 = domain.xd(n7);

            Real_t yv0 = domain.yd(n0);
            Real_t yv1 = domain.yd(n1);
            Real_t yv2 = domain.yd(n2);
            Real_t yv3 = domain.yd(n3);
            Real_t yv4 = domain.yd(n4);
            Real_t yv5 = domain.yd(n5);
            Real_t yv6 = domain.yd(n6);
            Real_t yv7 = domain.yd(n7);

            Real_t zv0 = domain.zd(n0);
            Real_t zv1 = domain.zd(n1);
            Real_t zv2 = domain.zd(n2);
            Real_t zv3 = domain.zd(n3);
            Real_t zv4 = domain.zd(n4);
            Real_t zv5 = domain.zd(n5);
            Real_t zv6 = domain.zd(n6);
            Real_t zv7 = domain.zd(n7);

            Real_t vol  = domain.volo(i) * domain.vnew(i);
            Real_t norm = Real_t(1.0) / (vol + ptiny);

            Real_t dxj = Real_t(-0.25) * ((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
            Real_t dyj = Real_t(-0.25) * ((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
            Real_t dzj = Real_t(-0.25) * ((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));

            Real_t dxi = Real_t(0.25) * ((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
            Real_t dyi = Real_t(0.25) * ((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
            Real_t dzi = Real_t(0.25) * ((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));

            Real_t dxk = Real_t(0.25) * ((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
            Real_t dyk = Real_t(0.25) * ((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
            Real_t dzk = Real_t(0.25) * ((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));

            ax = dyi * dzj - dzi * dyj;
            ay = dzi * dxj - dxi * dzj;
            az = dxi * dyj - dyi * dxj;

            domain.delx_zeta(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

            ax *= norm;
            ay *= norm;
            az *= norm;

            dxv = Real_t(0.25) * ((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
            dyv = Real_t(0.25) * ((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
            dzv = Real_t(0.25) * ((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));

            domain.delv_zeta(i) = ax * dxv + ay * dyv + az * dzv;

            ax = dyj * dzk - dzj * dyk;
            ay = dzj * dxk - dxj * dzk;
            az = dxj * dyk - dyj * dxk;

            domain.delx_xi(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

            ax *= norm;
            ay *= norm;
            az *= norm;

            dxv = Real_t(0.25) * ((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
            dyv = Real_t(0.25) * ((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
            dzv = Real_t(0.25) * ((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));

            domain.delv_xi(i) = ax * dxv + ay * dyv + az * dzv;

            ax = dyk * dzi - dzk * dyi;
            ay = dzk * dxi - dxk * dzi;
            az = dxk * dyi - dyk * dxi;

            domain.delx_eta(i) = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

            ax *= norm;
            ay *= norm;
            az *= norm;

            dxv = Real_t(-0.25) * ((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
            dyv = Real_t(-0.25) * ((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
            dzv = Real_t(-0.25) * ((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));

            domain.delv_eta(i) = ax * dxv + ay * dyv + az * dzv;
        });
}

static inline void
CalcMonotonicQRegionForElems(Domain& domain, Int_t r, Real_t ptiny)
{
    Real_t monoq_limiter_mult = domain.monoq_limiter_mult();
    Real_t monoq_max_slope    = domain.monoq_max_slope();
    Real_t qlc_monoq          = domain.qlc_monoq();
    Real_t qqc_monoq          = domain.qqc_monoq();

    Kokkos::parallel_for(
        "CalcMonotonicQRegionForElems", domain.regElemSize(r),
        KOKKOS_LAMBDA(const int i) {
            Index_t ielem = domain.regElemlist(r, i);
            Real_t  qlin, qquad;
            Real_t  phixi, phieta, phizeta;
            Int_t   bcMask = domain.elemBC(ielem);
            Real_t  delvm = 0.0, delvp = 0.0;

            Real_t norm = Real_t(1.) / (domain.delv_xi(ielem) + ptiny);

            switch(bcMask & XI_M)
            {
                case XI_M_COMM:
                case 0: delvm = domain.delv_xi(domain.lxim(ielem)); break;
                case XI_M_SYMM: delvm = domain.delv_xi(ielem); break;
                case XI_M_FREE: delvm = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvm = 0;
                    break;
            }
            switch(bcMask & XI_P)
            {
                case XI_P_COMM:
                case 0: delvp = domain.delv_xi(domain.lxip(ielem)); break;
                case XI_P_SYMM: delvp = domain.delv_xi(ielem); break;
                case XI_P_FREE: delvp = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvp = 0;
                    break;
            }

            delvm = delvm * norm;
            delvp = delvp * norm;

            phixi = Real_t(.5) * (delvm + delvp);

            delvm *= monoq_limiter_mult;
            delvp *= monoq_limiter_mult;

            if(delvm < phixi) phixi = delvm;
            if(delvp < phixi) phixi = delvp;
            if(phixi < Real_t(0.)) phixi = Real_t(0.);
            if(phixi > monoq_max_slope) phixi = monoq_max_slope;

            norm = Real_t(1.) / (domain.delv_eta(ielem) + ptiny);

            switch(bcMask & ETA_M)
            {
                case ETA_M_COMM:
                case 0: delvm = domain.delv_eta(domain.letam(ielem)); break;
                case ETA_M_SYMM: delvm = domain.delv_eta(ielem); break;
                case ETA_M_FREE: delvm = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvm = 0;
                    break;
            }
            switch(bcMask & ETA_P)
            {
                case ETA_P_COMM:
                case 0: delvp = domain.delv_eta(domain.letap(ielem)); break;
                case ETA_P_SYMM: delvp = domain.delv_eta(ielem); break;
                case ETA_P_FREE: delvp = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvp = 0;
                    break;
            }

            delvm = delvm * norm;
            delvp = delvp * norm;

            phieta = Real_t(.5) * (delvm + delvp);

            delvm *= monoq_limiter_mult;
            delvp *= monoq_limiter_mult;

            if(delvm < phieta) phieta = delvm;
            if(delvp < phieta) phieta = delvp;
            if(phieta < Real_t(0.)) phieta = Real_t(0.);
            if(phieta > monoq_max_slope) phieta = monoq_max_slope;

            norm = Real_t(1.) / (domain.delv_zeta(ielem) + ptiny);

            switch(bcMask & ZETA_M)
            {
                case ZETA_M_COMM:
                case 0: delvm = domain.delv_zeta(domain.lzetam(ielem)); break;
                case ZETA_M_SYMM: delvm = domain.delv_zeta(ielem); break;
                case ZETA_M_FREE: delvm = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvm = 0;
                    break;
            }
            switch(bcMask & ZETA_P)
            {
                case ZETA_P_COMM:
                case 0: delvp = domain.delv_zeta(domain.lzetap(ielem)); break;
                case ZETA_P_SYMM: delvp = domain.delv_zeta(ielem); break;
                case ZETA_P_FREE: delvp = Real_t(0.0); break;
                default:
                    printf("Error in switch at %s line %d\n", __FILE__, __LINE__);
                    delvp = 0;
                    break;
            }

            delvm = delvm * norm;
            delvp = delvp * norm;

            phizeta = Real_t(.5) * (delvm + delvp);

            delvm *= monoq_limiter_mult;
            delvp *= monoq_limiter_mult;

            if(delvm < phizeta) phizeta = delvm;
            if(delvp < phizeta) phizeta = delvp;
            if(phizeta < Real_t(0.)) phizeta = Real_t(0.);
            if(phizeta > monoq_max_slope) phizeta = monoq_max_slope;

            if(domain.vdov(ielem) > Real_t(0.))
            {
                qlin  = Real_t(0.);
                qquad = Real_t(0.);
            }
            else
            {
                Real_t delvxxi   = domain.delv_xi(ielem) * domain.delx_xi(ielem);
                Real_t delvxeta  = domain.delv_eta(ielem) * domain.delx_eta(ielem);
                Real_t delvxzeta = domain.delv_zeta(ielem) * domain.delx_zeta(ielem);

                if(delvxxi > Real_t(0.)) delvxxi = Real_t(0.);
                if(delvxeta > Real_t(0.)) delvxeta = Real_t(0.);
                if(delvxzeta > Real_t(0.)) delvxzeta = Real_t(0.);

                Real_t rho =
                    domain.elemMass(ielem) / (domain.volo(ielem) * domain.vnew(ielem));

                qlin =
                    -qlc_monoq * rho *
                    (delvxxi * (Real_t(1.) - phixi) + delvxeta * (Real_t(1.) - phieta) +
                     delvxzeta * (Real_t(1.) - phizeta));

                qquad = qqc_monoq * rho *
                        (delvxxi * delvxxi * (Real_t(1.) - phixi * phixi) +
                         delvxeta * delvxeta * (Real_t(1.) - phieta * phieta) +
                         delvxzeta * delvxzeta * (Real_t(1.) - phizeta * phizeta));
            }

            domain.qq(ielem) = qquad;
            domain.ql(ielem) = qlin;
        });
}

static inline void
CalcMonotonicQForElems(Domain& domain)
{
    const Real_t ptiny = Real_t(1.e-36);

    for(Index_t r = 0; r < domain.numReg(); ++r)
    {
        if(domain.regElemSize(r) > 0)
        {
            CalcMonotonicQRegionForElems(domain, r, ptiny);
        }
    }
}

static inline void
CalcQForElems(Domain& domain)
{
    Index_t numElem = domain.numElem();

    if(numElem != 0)
    {
        Int_t allElem = numElem +                             /* local elem */
                        2 * domain.sizeX() * domain.sizeY() + /* plane ghosts */
                        2 * domain.sizeX() * domain.sizeZ() + /* row ghosts */
                        2 * domain.sizeY() * domain.sizeZ();  /* col ghosts */

        domain.AllocateGradients(numElem, allElem);

#if USE_MPI
        CommRecv(domain, MSG_MONOQ, 3, domain.sizeX(), domain.sizeY(), domain.sizeZ(),
                 true, true);
#endif
        CalcMonotonicQGradientsForElems(domain);

#if USE_MPI
        Domain_member fieldData[3];

        fieldData[0] = &Domain::delv_xi;
        fieldData[1] = &Domain::delv_eta;
        fieldData[2] = &Domain::delv_zeta;

        CommSend(domain, MSG_MONOQ, 3, fieldData, domain.sizeX(), domain.sizeY(),
                 domain.sizeZ(), true, true);

        CommMonoQ(domain);
#endif

        CalcMonotonicQForElems(domain);

        domain.DeallocateGradients();

        Index_t idx = 0;
        Kokkos::parallel_reduce(
            "CalcQForElems", numElem,
            KOKKOS_LAMBDA(const Index_t& i, Index_t& count) {
                if(domain.q(i) > domain.qstop())
                {
                    count++;
                }
            },
            idx);

        if(idx > 0)
        {
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, QStopError);
#else
            exit(QStopError);
#endif
        }
    }
}

KOKKOS_INLINE_FUNCTION
void
CalcPressureForElem(Real_t& p_new_i, Real_t& bvc_i, Real_t& pbvc_i, const Real_t& e_old_i,
                    const Real_t& compression_i, const Real_t& vnewc_e,
                    const Real_t& pmin, const Real_t& p_cut, const Real_t& eosvmax)
{
    const Real_t c1s = Real_t(2.0) / Real_t(3.0);
    bvc_i            = c1s * (compression_i + Real_t(1.));

    pbvc_i = c1s;

    p_new_i = bvc_i * e_old_i;

    if(FABS(p_new_i) < p_cut) p_new_i = Real_t(0.0);

    if(vnewc_e >= eosvmax) /* impossible condition here? */
        p_new_i = Real_t(0.0);

    if(p_new_i < pmin) p_new_i = pmin;
}

static inline void
CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new, Real_t* bvc, Real_t* pbvc,
                   Real_t* p_old, Real_t* e_old, Real_t* q_old, Real_t* compression,
                   Real_t* compHalfStep, Real_t* vnewc, Real_t* work, Real_t* delvc,
                   Real_t pmin, Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin,
                   Real_t* qq_old, Real_t* ql_old, Real_t rho0, Real_t eosvmax,
                   Index_t length, Domain& domain, Index_t r)
{
    Kokkos::parallel_for(
        "CalcEnergyForElems", length, KOKKOS_LAMBDA(const int i) {
            const Real_t delvc_i = delvc[i];
            const Real_t p_old_i = p_old[i];
            const Real_t q_old_i = q_old[i];
            Real_t e_new_i = e_old[i] - Real_t(0.5) * delvc_i * (p_old_i + q_old_i) +
                             Real_t(0.5) * work[i];

            if(e_new_i < emin)
            {
                e_new_i = emin;
            }

            Real_t       bvc_i, pbvc_i;
            Real_t       pHalfStep_i;
            const Real_t vnewc_e        = vnewc[domain.regElemlist(r, i)];
            const Real_t compHalfStep_i = compHalfStep[i];
            CalcPressureForElem(pHalfStep_i, bvc_i, pbvc_i, e_new_i, compHalfStep_i,
                                vnewc_e, pmin, p_cut, eosvmax);

            Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep_i);

            Real_t       q_new_i;
            const Real_t ql_old_i = ql_old[i];
            const Real_t qq_old_i = qq_old[i];
            if(delvc_i > Real_t(0.))
            {
                q_new_i /* = qq_old[i] = ql_old[i] */ = Real_t(0.);
            }
            else
            {
                Real_t ssc =
                    (pbvc_i * e_new_i + vhalf * vhalf * bvc_i * pHalfStep_i) / rho0;

                if(ssc <= Real_t(.1111111e-36))
                {
                    ssc = Real_t(.3333333e-18);
                }
                else
                {
                    ssc = SQRT(ssc);
                }

                q_new_i = (ssc * ql_old_i + qq_old_i);
            }

            e_new_i = e_new_i + Real_t(0.5) * delvc_i *
                                    (Real_t(3.0) * (p_old_i + q_old_i) -
                                     Real_t(4.0) * (pHalfStep_i + q_new_i));

            e_new_i += Real_t(0.5) * work[i];

            if(FABS(e_new_i) < e_cut)
            {
                e_new_i = Real_t(0.);
            }
            if(e_new_i < emin)
            {
                e_new_i = emin;
            }
            Real_t       p_new_i;
            const Real_t compression_i = compression[i];
            CalcPressureForElem(p_new_i, bvc_i, pbvc_i, e_new_i, compression_i, vnewc_e,
                                pmin, p_cut, eosvmax);

            const Real_t sixth = Real_t(1.0) / Real_t(6.0);
            Real_t       q_tilde;

            if(delvc_i > Real_t(0.))
            {
                q_tilde = Real_t(0.);
            }
            else
            {
                Real_t ssc =
                    (pbvc_i * e_new_i + vnewc_e * vnewc_e * bvc_i * p_new_i) / rho0;

                if(ssc <= Real_t(.1111111e-36))
                {
                    ssc = Real_t(.3333333e-18);
                }
                else
                {
                    ssc = SQRT(ssc);
                }

                q_tilde = (ssc * ql_old_i + qq_old_i);
            }

            e_new_i =
                e_new_i - (Real_t(7.0) * (p_old_i + q_old_i) -
                           Real_t(8.0) * (pHalfStep_i + q_new_i) + (p_new_i + q_tilde)) *
                              delvc_i * sixth;

            if(FABS(e_new_i) < e_cut)
            {
                e_new_i = Real_t(0.);
            }
            if(e_new_i < emin)
            {
                e_new_i = emin;
            }

            CalcPressureForElem(p_new_i, bvc_i, pbvc_i, e_new_i, compression_i, vnewc_e,
                                pmin, p_cut, eosvmax);
            bvc[i]   = bvc_i;
            pbvc[i]  = pbvc_i;
            p_new[i] = p_new_i;

            if(delvc_i <= Real_t(0.))
            {
                Real_t ssc =
                    (pbvc_i * e_new_i + vnewc_e * vnewc_e * bvc_i * p_new_i) / rho0;

                if(ssc <= Real_t(.1111111e-36))
                {
                    ssc = Real_t(.3333333e-18);
                }
                else
                {
                    ssc = SQRT(ssc);
                }

                q_new_i = (ssc * ql_old_i + qq_old_i);

                if(FABS(q_new_i) < q_cut) q_new_i = Real_t(0.);
            }
            q_new[i] = q_new_i;
            e_new[i] = e_new_i;
        });

    return;
}

static inline void
CalcSoundSpeedForElems(Domain& domain, Real_t* vnewc, Real_t rho0, Real_t* enewc,
                       Real_t* pnewc, Real_t* pbvc, Real_t* bvc, Real_t ss4o3,
                       Index_t len, Index_t r)
{
    Kokkos::parallel_for(
        "CalcSoundSpeedForElems", len, KOKKOS_LAMBDA(const int i) {
            Index_t ielem = domain.regElemlist(r, i);
            Real_t  ssTmp =
                (pbvc[i] * enewc[i] + vnewc[ielem] * vnewc[ielem] * bvc[i] * pnewc[i]) /
                rho0;
            if(ssTmp <= Real_t(.1111111e-36))
            {
                ssTmp = Real_t(.3333333e-18);
            }
            else
            {
                ssTmp = SQRT(ssTmp);
            }
            domain.ss(ielem) = ssTmp;
        });
}

static inline void
EvalEOSForElems(Domain& domain, Real_t* vnewc, Int_t numElemReg, Index_t r, Int_t rep)
{
    Real_t e_cut = domain.e_cut();
    Real_t p_cut = domain.p_cut();
    Real_t ss4o3 = domain.ss4o3();
    Real_t q_cut = domain.q_cut();

    Real_t eosvmax = domain.eosvmax();
    Real_t eosvmin = domain.eosvmin();
    Real_t pmin    = domain.pmin();
    Real_t emin    = domain.emin();
    Real_t rho0    = domain.refdens();

    ResizeBuffer((numElemReg * sizeof(Real_t) + 4096) * 16);

    Real_t* e_old        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* delvc        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* p_old        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* q_old        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* compression  = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* compHalfStep = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* qq_old       = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* ql_old       = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* work         = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* p_new        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* e_new        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* q_new        = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* bvc          = AllocateFromBuffer<Real_t>(numElemReg);
    Real_t* pbvc         = AllocateFromBuffer<Real_t>(numElemReg);

    for(Int_t j = 0; j < rep; j++)
    {
        Kokkos::parallel_for(
            "EvalEOSForElems A", numElemReg, KOKKOS_LAMBDA(const int i) {
                Index_t ielem            = domain.regElemlist(r, i);
                e_old[i]                 = domain.c_e(ielem);
                delvc[i]                 = domain.c_delv(ielem);
                p_old[i]                 = domain.c_p(ielem);
                q_old[i]                 = domain.c_q(ielem);
                qq_old[i]                = domain.c_qq(ielem);
                ql_old[i]                = domain.c_ql(ielem);
                const Real_t vnewc_ielem = vnewc[ielem];
                Real_t       vchalf;
                compression[i]  = Real_t(1.) / vnewc_ielem - Real_t(1.);
                vchalf          = vnewc_ielem - delvc[i] * Real_t(.5);
                compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);

                if(eosvmin != Real_t(0.))
                {
                    if(vnewc_ielem <= eosvmin)
                    { /* impossible due to calling func? */
                        compHalfStep[i] = compression[i];
                    }
                }
                if(eosvmax != Real_t(0.))
                {
                    if(vnewc_ielem >= eosvmax)
                    { /* impossible due to calling func? */
                        p_old[i]        = Real_t(0.);
                        compression[i]  = Real_t(0.);
                        compHalfStep[i] = Real_t(0.);
                    }
                }
                work[i] = Real_t(0.);
            });

        CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old,
                           compression, compHalfStep, vnewc, work, delvc, pmin, p_cut,
                           e_cut, q_cut, emin, qq_old, ql_old, rho0, eosvmax, numElemReg,
                           domain, r);
    }

    Kokkos::parallel_for(
        "EvalEOSForElems F", numElemReg, KOKKOS_LAMBDA(const int i) {
            Index_t ielem   = domain.regElemlist(r, i);
            domain.p(ielem) = p_new[i];
            domain.e(ielem) = e_new[i];
            domain.q(ielem) = q_new[i];
        });

    CalcSoundSpeedForElems(domain, vnewc, rho0, e_new, p_new, pbvc, bvc, ss4o3,
                           numElemReg, r);
}

static inline void
ApplyMaterialPropertiesForElems(Domain& domain)
{
    Index_t numElem = domain.numElem();

    if(numElem != 0)
    {
        Real_t                eosvmin = domain.eosvmin();
        Real_t                eosvmax = domain.eosvmax();
        Kokkos::View<Real_t*> vnewc("vnewc", numElem);

        Kokkos::parallel_for(
            "ApplyMaterialPropertiesForElems A", numElem,
            KOKKOS_LAMBDA(const int i) { vnewc[i] = domain.vnew(i); });

        if(eosvmin != Real_t(0.))
        {
            Kokkos::parallel_for(
                "ApplyMaterialPropertiesForElems B", numElem, KOKKOS_LAMBDA(const int i) {
                    if(vnewc[i] < eosvmin) vnewc[i] = eosvmin;
                });
        }

        if(eosvmax != Real_t(0.))
        {
            Kokkos::parallel_for(
                "ApplyMaterialPropertiesForElems C", numElem, KOKKOS_LAMBDA(const int i) {
                    if(vnewc[i] > eosvmax) vnewc[i] = eosvmax;
                });
        }

        int error = 0;
        Kokkos::parallel_reduce(
            "ApplyMaterialPropertiesForElems", numElem,
            KOKKOS_LAMBDA(const int i, int& err) {
                Real_t vc = domain.v(i);
                if(eosvmin != Real_t(0.))
                {
                    if(vc < eosvmin) vc = eosvmin;
                }
                if(eosvmax != Real_t(0.))
                {
                    if(vc > eosvmax) vc = eosvmax;
                }
                if(vc <= 0.)
                {
                    err++;
                }
            },
            error);

        if(error != 0)
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
            exit(VolumeError);
#endif

        for(Int_t r = 0; r < domain.numReg(); r++)
        {
            Index_t numElemReg = domain.regElemSize(r);
            //      Index_t *regElemList = domain.regElemlist(r);
            Int_t rep;
            if(r < domain.numReg() / 2)
                rep = 1;
            else if(r < (domain.numReg() - (domain.numReg() + 15) / 20))
                rep = 1 + domain.cost();
            else
                rep = 10 * (1 + domain.cost());
            EvalEOSForElems(domain, vnewc.data(), numElemReg, r, rep);
        }
    }
}

static inline void
UpdateVolumesForElems(Domain& domain, Real_t v_cut, Index_t length)
{
    if(length != 0)
    {
        Kokkos::parallel_for(
            "UpdateVolumesForElems", length, KOKKOS_LAMBDA(const int i) {
                Real_t tmpV = domain.vnew(i);

                if(FABS(tmpV - Real_t(1.0)) < v_cut) tmpV = Real_t(1.0);

                domain.v(i) = tmpV;
            });
    }

    return;
}

static inline void
LagrangeElements(Domain& domain, Index_t numElem)
{
    CalcLagrangeElements(domain);

    CalcQForElems(domain);

    ApplyMaterialPropertiesForElems(domain);

    UpdateVolumesForElems(domain, domain.v_cut(), numElem);
}

static inline void
CalcCourantConstraintForElems(Domain& domain, Index_t length, Index_t r, Real_t qqc,
                              Real_t& dtcourant)
{
    typedef Kokkos::View<Real_t*> view_real_t;

    Real_t  qqc2          = Real_t(64.0) * qqc * qqc;
    Real_t  dtcourant_tmp = dtcourant;
    Index_t courant_elem  = -1;

    MinFinder result;

    Kokkos::parallel_reduce(
        "CalcCourantConstraintForElems", length,
        KOKKOS_LAMBDA(const int i, MinFinder& minf) {
            Index_t indx = domain.regElemlist(r, i);
            Real_t  dtf  = domain.ss(indx) * domain.ss(indx);

            if(domain.vdov(indx) < Real_t(0.))
            {
                dtf = dtf + qqc2 * domain.arealg(indx) * domain.arealg(indx) *
                                domain.vdov(indx) * domain.vdov(indx);
            }

            dtf = SQRT(dtf);
            dtf = domain.arealg(indx) / dtf;

            MinFinder tmp(dtf, i);
            if(domain.vdov(indx) != Real_t(0.))
            {
                minf += tmp;
            }
        },
        result);

    dtcourant_tmp = result.val;

    if(dtcourant_tmp > dtcourant)
    {
        dtcourant_tmp = dtcourant;
    }

    courant_elem = result.i;

    if(courant_elem != -1)
    {
        dtcourant = dtcourant_tmp;
    }

    return;
}

static inline void
CalcHydroConstraintForElems(Domain& domain, Index_t length, Index_t r, Real_t dvovmax,
                            Real_t& dthydro)
{
    typedef Kokkos::View<Real_t*> view_real_t;

    Real_t    dthydro_tmp = dthydro;
    Index_t   hydro_elem  = -1;
    MinFinder result;

    Kokkos::parallel_reduce(
        "CalcHydroConstraintForElems", length,
        KOKKOS_LAMBDA(const int i, MinFinder& minf) {
            Index_t indx = domain.regElemlist(r, i);

            if(domain.vdov(indx) != Real_t(0.))
            {
                Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx)) + Real_t(1.e-20));

                MinFinder tmp(dtdvov, i);
                if(domain.vdov(indx) != Real_t(0.))
                {
                    minf += tmp;
                }
            }
        },
        result);

    if(result.val > dthydro)
    {
        result.val = dthydro;
    }

    if(result.i != -1)
    {
        dthydro = result.val;
    }

    return;
}

static inline void
CalcTimeConstraintsForElems(Domain& domain)
{
    domain.dtcourant() = 1.0e+20;
    domain.dthydro()   = 1.0e+20;

    for(Index_t r = 0; r < domain.numReg(); ++r)
    {
        CalcCourantConstraintForElems(domain, domain.regElemSize(r), r, domain.qqc(),
                                      domain.dtcourant());

        CalcHydroConstraintForElems(domain, domain.regElemSize(r), r, domain.dvovmax(),
                                    domain.dthydro());
    }
}

static inline void
LagrangeLeapFrog(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_LATE
    Domain_member fieldData[6];
#endif
    LagrangeNodal(domain);

#ifdef SEDOV_SYNC_POS_VEL_LATE
#endif
    LagrangeElements(domain, domain.numElem());

#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_LATE
    CommRecv(domain, MSG_SYNC_POS_VEL, 6, domain.sizeX() + 1, domain.sizeY() + 1,
             domain.sizeZ() + 1, false, false);

    fieldData[0] = &Domain::x;
    fieldData[1] = &Domain::y;
    fieldData[2] = &Domain::z;
    fieldData[3] = &Domain::xd;
    fieldData[4] = &Domain::yd;
    fieldData[5] = &Domain::zd;

    CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData, domain.sizeX() + 1,
             domain.sizeY() + 1, domain.sizeZ() + 1, false, false);
#    endif
#endif

    CalcTimeConstraintsForElems(domain);

#if USE_MPI
#    ifdef SEDOV_SYNC_POS_VEL_LATE
    CommSyncPosVel(domain);
#    endif
#endif
}

int
main(int argc, char* argv[])
{
    Int_t              numRanks;
    Int_t              myRank;
    struct cmdLineOpts opts;

#if USE_MPI
    Domain_member fieldData;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#else
    numRanks = 1;
    myRank   = 0;
#endif

    Kokkos::initialize(argc, argv);
    {
        Kokkos::Tools::pushRegion("initialization");
        opts.its       = 9999999;
        opts.nx        = 30;
        opts.numReg    = 11;
        opts.numFiles  = (int) (numRanks + 10) / 9;
        opts.showProg  = 0;
        opts.quiet     = 0;
        opts.viz       = 0;
        opts.balance   = 1;
        opts.cost      = 1;
        opts.do_atomic = 0;

        ParseCommandLineOptions(argc, argv, myRank, &opts);

        if(opts.do_atomic == 1)
            do_atomic = 1;
        else
            do_atomic = 0;

        if((myRank == 0) && (opts.quiet == 0))
        {
            printf("Running problem size %d^3 per domain until completion\n", opts.nx);
            printf("Num processors: %d\n", numRanks);
            printf("Total number of elements: %lld\n\n",
                   (long long int) (numRanks * opts.nx * opts.nx * opts.nx));
            printf("To run other sizes, use -s <integer>.\n");
            printf("To run a fixed number of iterations, use -i <integer>.\n");
            printf("To run a more or less balanced region set, use -b <integer>.\n");
            printf("To change the relative costs of regions, use -c <integer>.\n");
            printf("To print out progress, use -p\n");
            printf("To write an output file for VisIt, use -v\n");
            printf("See help (-h) for more options\n\n");
        }

        Int_t col, row, plane, side;
        InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

        // Build the main data structure and initialize it
        Domain locDom(numRanks, col, row, plane, opts.nx, side, opts.numReg, opts.balance,
                      opts.cost);

#if USE_MPI
        fieldData = &Domain::nodalMass;

        // Initial domain boundary communication
        CommRecv(locDom, MSG_COMM_SBN, 1, locDom.sizeX() + 1, locDom.sizeY() + 1,
                 locDom.sizeZ() + 1, true, false);
        CommSend(locDom, MSG_COMM_SBN, 1, &fieldData, locDom.sizeX() + 1,
                 locDom.sizeY() + 1, locDom.sizeZ() + 1, true, false);
        CommSBN(locDom, 1, &fieldData);

        // End initialization
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        Kokkos::Tools::popRegion();

#if USE_MPI
        double start = MPI_Wtime();
#else
        timeval start;
        gettimeofday(&start, nullptr);
#endif

        uint32_t _time_incrp = 0;
        uint32_t _leap_frogp = 0;
        Kokkos::Tools::createProfileSection("TimeIncr", &_time_incrp);
        Kokkos::Tools::createProfileSection("LeapFrog", &_leap_frogp);

        while((locDom.time() < locDom.stoptime()) && (locDom.cycle() < opts.its))
        {
            Kokkos::Tools::startSection(_time_incrp);
            // CAUSAL_BEGIN("Iteration")
            TimeIncrement(locDom);
            Kokkos::Tools::stopSection(_time_incrp);

            Kokkos::Tools::startSection(_leap_frogp);
            LagrangeLeapFrog(locDom);
            Kokkos::Tools::stopSection(_leap_frogp);
            CAUSAL_PROGRESS_NAMED("Iteration")
            // CAUSAL_END("Iteration")

            if((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0))
            {
                printf("cycle = %d, time = %e, dt=%e\n", locDom.cycle(),
                       double(locDom.time()), double(locDom.deltatime()));
            }
            Kokkos::Tools::markEvent("completed_timestep");
            CAUSAL_PROGRESS
        }

        Kokkos::Tools::destroyProfileSection(_time_incrp);
        Kokkos::Tools::destroyProfileSection(_leap_frogp);

        double elapsed_time;
#if USE_MPI
        elapsed_time = MPI_Wtime() - start;
#else
        timeval end;
        gettimeofday(&end, NULL);
        elapsed_time = (double) (end.tv_sec - start.tv_sec) +
                       ((double) (end.tv_usec - start.tv_usec)) / 1000000;
#endif
        double elapsed_timeG;
#if USE_MPI
        MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
#else
        elapsed_timeG = elapsed_time;
#endif

        Kokkos::Tools::pushRegion("finalization");
        if(opts.viz)
        {
            DumpToVisit(locDom, opts.numFiles, myRank, numRanks);
        }

        if((myRank == 0) && (opts.quiet == 0))
        {
            VerifyAndWriteFinalOutput(elapsed_timeG, locDom, opts.nx, numRanks);
        }
        Kokkos::Tools::popRegion();

        buffer = Kokkos::View<Real_t*>();
    }
    Kokkos::finalize();
#if USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
