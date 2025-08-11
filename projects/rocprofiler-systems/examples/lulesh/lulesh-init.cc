#include <math.h>
#if USE_MPI
#    include <mpi.h>
#endif
#include "lulesh.h"
#include <cstdlib>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static KOKKOS_INLINE_FUNCTION Real_t
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

/******************************************/

KOKKOS_INLINE_FUNCTION
Real_t
CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
    return CalcElemVolume(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], y[0], y[1],
                          y[2], y[3], y[4], y[5], y[6], y[7], z[0], z[1], z[2], z[3],
                          z[4], z[5], z[6], z[7]);
}

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Index_t colLoc, Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
: m_e_cut(Real_t(1.0e-7))
, m_p_cut(Real_t(1.0e-7))
, m_q_cut(Real_t(1.0e-7))
, m_v_cut(Real_t(1.0e-10))
, m_u_cut(Real_t(1.0e-7))
, m_hgcoef(Real_t(3.0))
, m_ss4o3(Real_t(4.0) / Real_t(3.0))
, m_qstop(Real_t(1.0e+12))
, m_monoq_max_slope(Real_t(1.0))
, m_monoq_limiter_mult(Real_t(2.0))
, m_qlc_monoq(Real_t(0.5))
, m_qqc_monoq(Real_t(2.0) / Real_t(3.0))
, m_qqc(Real_t(2.0))
, m_eosvmax(Real_t(1.0e+9))
, m_eosvmin(Real_t(1.0e-9))
, m_pmin(Real_t(0.))
, m_emin(Real_t(-1.0e+15))
, m_dvovmax(Real_t(0.1))
, m_refdens(Real_t(1.0))
,
//
// set pointers to (potentially) "new'd" arrays to null to
// simplify deallocation.
//
m_regNumList(0)
//   m_nodeElemStart(0),
//   m_nodeElemCornerList(0),
// m_regElemSize(0),
// m_regElemlist(0)
#if USE_MPI
, commDataSend(0)
, commDataRecv(0)
#endif
{
    Index_t edgeElems = nx;
    Index_t edgeNodes = edgeElems + 1;
    this->cost()      = cost;

    m_tp       = tp;
    m_numRanks = numRanks;

    ///////////////////////////////
    //   Initialize Sedov Mesh
    ///////////////////////////////

    // construct a uniform box for this processor

    m_colLoc   = colLoc;
    m_rowLoc   = rowLoc;
    m_planeLoc = planeLoc;

    m_sizeX   = edgeElems;
    m_sizeY   = edgeElems;
    m_sizeZ   = edgeElems;
    m_numElem = edgeElems * edgeElems * edgeElems;

    m_numNode = edgeNodes * edgeNodes * edgeNodes;

    m_regNumList = Allocate<Index_t>(numElem());  // material indexset

    // Elem-centered
    AllocateElemPersistent(numElem());

    // Node-centered
    AllocateNodePersistent(numNode());

    SetupCommBuffers(edgeNodes);

    // Basic Field Initialization
    Kokkos::deep_copy(m_e, 0.0);
    Kokkos::deep_copy(m_p, 0.0);
    Kokkos::deep_copy(m_q, 0.0);
    Kokkos::deep_copy(m_ss, 0.0);

    // Note - v initializes to 1.0, not 0.0!
    Kokkos::deep_copy(m_v, 1.0);

    Kokkos::deep_copy(m_xd, 0.0);
    Kokkos::deep_copy(m_yd, 0.0);
    Kokkos::deep_copy(m_zd, 0.0);

    Kokkos::deep_copy(m_xdd, 0.0);
    Kokkos::deep_copy(m_ydd, 0.0);
    Kokkos::deep_copy(m_zdd, 0.0);

    Kokkos::deep_copy(m_nodalMass, 0.0);

    BuildMesh(nx, edgeNodes, edgeElems);

    SetupThreadSupportStructures();

    // Setup region index sets. For now, these are constant sized
    // throughout the run, but could be changed every cycle to
    // simulate effects of ALE on the lagrange solver
    CreateRegionIndexSets(nr, balance);

    // Setup symmetry nodesets
    SetupSymmetryPlanes(edgeNodes);

    // Setup element connectivities
    SetupElementConnectivities(edgeElems);

    // Setup symmetry planes and free surface boundary arrays
    SetupBoundaryConditions(edgeElems);

    // Setup defaults

    // These can be changed (requires recompile) if you want to run
    // with a fixed timestep, or to a different end time, but it's
    // probably easier/better to just run a fixed number of timesteps
    // using the -i flag in 2.x

    dtfixed()  = Real_t(-1.0e-6);  // Negative means use courant condition
    stoptime() = Real_t(1.0e-2);   // *Real_t(edgeElems*tp/45.0) ;

    // Initial conditions
    deltatimemultlb() = Real_t(1.1);
    deltatimemultub() = Real_t(1.2);
    dtcourant()       = Real_t(1.0e+20);
    dthydro()         = Real_t(1.0e+20);
    dtmax()           = Real_t(1.0e-2);
    time()            = Real_t(0.);
    cycle()           = Int_t(0);

    // With C++17 requirement we could just run this on the device
    // without creating temporary host copies
    auto h_nodelist  = Kokkos::create_mirror_view(m_nodelist);
    auto h_x         = Kokkos::create_mirror_view(m_x);
    auto h_y         = Kokkos::create_mirror_view(m_y);
    auto h_z         = Kokkos::create_mirror_view(m_z);
    auto h_volo      = Kokkos::create_mirror_view(m_volo);
    auto h_elemMass  = Kokkos::create_mirror_view(m_elemMass);
    auto h_nodalMass = Kokkos::create_mirror_view(m_nodalMass);
    Kokkos::deep_copy(h_nodelist, m_nodelist);
    Kokkos::deep_copy(h_x, m_x);
    Kokkos::deep_copy(h_y, m_y);
    Kokkos::deep_copy(h_z, m_z);
    // initialize field data
    for(Index_t i = 0; i < numElem(); ++i)
    {
        Real_t x_local[8], y_local[8], z_local[8];
        for(Index_t lnode = 0; lnode < 8; ++lnode)
        {
            Index_t gnode  = h_nodelist(i, lnode);
            x_local[lnode] = h_x(gnode);
            y_local[lnode] = h_y(gnode);
            z_local[lnode] = h_z(gnode);
        }

        // volume calculations
        Real_t volume = CalcElemVolume(x_local, y_local, z_local);
        h_volo(i)     = volume;
        h_elemMass(i) = volume;
        for(Index_t j = 0; j < 8; ++j)
        {
            Index_t idx = h_nodelist(i, j);
            h_nodalMass(idx) += volume / Real_t(8.0);
        }
    }

    Kokkos::deep_copy(m_volo, h_volo);
    Kokkos::deep_copy(m_elemMass, h_elemMass);
    Kokkos::deep_copy(m_nodalMass, h_nodalMass);

    // deposit initial energy
    // An energy of 3.948746e+7 is correct for a problem with
    // 45 zones along a side - we need to scale it
    const Real_t ebase = Real_t(3.948746e+7);
    Real_t       scale = (nx * m_tp) / Real_t(45.0);
    Real_t       einit = ebase * scale * scale * scale;
    if(m_rowLoc + m_colLoc + m_planeLoc == 0)
    {
        // Dump into the first zone (which we know is in the corner)
        // of the domain that sits at the origin
        Kokkos::deep_copy(Kokkos::subview(m_e, 0), einit);
        // e(0) = einit;
    }
    // set initial deltatime base on analytic CFL calculation
    deltatime() = (Real_t(.5) * cbrt(h_volo(0))) / sqrt(Real_t(2.0) * einit);

}  // End constructor

////////////////////////////////////////////////////////////////////////////////
Domain::~Domain()
{
    /*   Release(&m_regNumList);
       Release(&m_nodeElemStart);
       Release(&m_nodeElemCornerList);
       Release(&m_regElemSize);
       for (Index_t i=0 ; i<numReg() ; ++i) {
         Release(&m_regElemlist[i]);
       }
       Release(&m_regElemlist);

    #if USE_MPI
       Release(&commDataSend);
       Release(&commDataRecv);
    #endif
    */
}  // End destructor

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
    Index_t meshEdgeElems = m_tp * nx;

    auto h_x = Kokkos::create_mirror_view(m_x);
    auto h_y = Kokkos::create_mirror_view(m_y);
    auto h_z = Kokkos::create_mirror_view(m_z);

    // initialize nodal coordinates
    Index_t nidx = 0;
    Real_t  tz   = Real_t(1.125) * Real_t(m_planeLoc * nx) / Real_t(meshEdgeElems);
    for(Index_t plane = 0; plane < edgeNodes; ++plane)
    {
        Real_t ty = Real_t(1.125) * Real_t(m_rowLoc * nx) / Real_t(meshEdgeElems);
        for(Index_t row = 0; row < edgeNodes; ++row)
        {
            Real_t tx = Real_t(1.125) * Real_t(m_colLoc * nx) / Real_t(meshEdgeElems);
            for(Index_t col = 0; col < edgeNodes; ++col)
            {
                h_x(nidx) = tx;
                h_y(nidx) = ty;
                h_z(nidx) = tz;
                ++nidx;
                // tx += ds ; // may accumulate roundoff...
                tx = Real_t(1.125) * Real_t(m_colLoc * nx + col + 1) /
                     Real_t(meshEdgeElems);
            }
            // ty += ds ;  // may accumulate roundoff...
            ty = Real_t(1.125) * Real_t(m_rowLoc * nx + row + 1) / Real_t(meshEdgeElems);
        }
        // tz += ds ;  // may accumulate roundoff...
        tz = Real_t(1.125) * Real_t(m_planeLoc * nx + plane + 1) / Real_t(meshEdgeElems);
    }

    Kokkos::deep_copy(m_x, h_x);
    Kokkos::deep_copy(m_y, h_y);
    Kokkos::deep_copy(m_z, h_z);

    auto h_nodelist = Kokkos::create_mirror_view(m_nodelist);
    // embed hexehedral elements in nodal point lattice
    Index_t zidx = 0;
    nidx         = 0;
    for(Index_t plane = 0; plane < edgeElems; ++plane)
    {
        for(Index_t row = 0; row < edgeElems; ++row)
        {
            for(Index_t col = 0; col < edgeElems; ++col)
            {
                h_nodelist(zidx, 0) = nidx;
                h_nodelist(zidx, 1) = nidx + 1;
                h_nodelist(zidx, 2) = nidx + edgeNodes + 1;
                h_nodelist(zidx, 3) = nidx + edgeNodes;
                h_nodelist(zidx, 4) = nidx + edgeNodes * edgeNodes;
                h_nodelist(zidx, 5) = nidx + edgeNodes * edgeNodes + 1;
                h_nodelist(zidx, 6) = nidx + edgeNodes * edgeNodes + edgeNodes + 1;
                h_nodelist(zidx, 7) = nidx + edgeNodes * edgeNodes + edgeNodes;
                ++zidx;
                ++nidx;
            }
            ++nidx;
        }
        nidx += edgeNodes;
    }
    Kokkos::deep_copy(m_nodelist, h_nodelist);
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupThreadSupportStructures()
{
    // set up node-centered indexing of elements
    Kokkos::View<Index_t*, Kokkos::HostSpace> nodeElemCount("nodeElemCount", numNode());
    auto h_nodelist = Kokkos::create_mirror_view(m_nodelist);
    Kokkos::deep_copy(h_nodelist, m_nodelist);

    for(Index_t i = 0; i < numElem(); ++i)
    {
        for(Index_t j = 0; j < 8; ++j)
        {
            ++(nodeElemCount[h_nodelist(i, j)]);
        }
    }

    m_nodeElemStart      = Kokkos::View<Index_t*>("m_nodeElemStart", numNode() + 1);
    auto h_nodeElemStart = Kokkos::create_mirror_view(m_nodeElemStart);

    h_nodeElemStart[0] = 0;

    for(Index_t i = 1; i <= numNode(); ++i)
    {
        h_nodeElemStart[i] = h_nodeElemStart[i - 1] + nodeElemCount[i - 1];
    }

    m_nodeElemCornerList =
        Kokkos::View<Index_t*>("nodeElemCornerList", h_nodeElemStart[numNode()]);
    auto h_nodeElemCornerList = Kokkos::create_mirror_view(m_nodeElemCornerList);

    for(Index_t i = 0; i < numNode(); ++i)
    {
        nodeElemCount[i] = 0;
    }

    for(Index_t i = 0; i < numElem(); ++i)
    {
        for(Index_t j = 0; j < 8; ++j)
        {
            Index_t m                    = h_nodelist(i, j);
            Index_t k                    = i * 8 + j;
            Index_t offset               = h_nodeElemStart[m] + nodeElemCount[m];
            h_nodeElemCornerList[offset] = k;
            ++(nodeElemCount[m]);
        }
    }

    Index_t clSize = h_nodeElemStart[numNode()];
    for(Index_t i = 0; i < clSize; ++i)
    {
        Index_t clv = h_nodeElemCornerList[i];
        if((clv < 0) || (clv > numElem() * 8))
        {
            fprintf(
                stderr,
                "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
            exit(-1);
#endif
        }
    }
    Kokkos::deep_copy(m_nodeElemCornerList, h_nodeElemCornerList);
    Kokkos::deep_copy(m_nodeElemStart, h_nodeElemStart);
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
    // allocate a buffer large enough for nodal ghost data
    Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ())) + 1;
    m_maxPlaneSize      = CACHE_ALIGN_REAL(maxEdgeSize * maxEdgeSize);
    m_maxEdgeSize       = CACHE_ALIGN_REAL(maxEdgeSize);

    // assume communication to 6 neighbors by default
    m_rowMin   = (m_rowLoc == 0) ? 0 : 1;
    m_rowMax   = (m_rowLoc == m_tp - 1) ? 0 : 1;
    m_colMin   = (m_colLoc == 0) ? 0 : 1;
    m_colMax   = (m_colLoc == m_tp - 1) ? 0 : 1;
    m_planeMin = (m_planeLoc == 0) ? 0 : 1;
    m_planeMax = (m_planeLoc == m_tp - 1) ? 0 : 1;

#if USE_MPI
    // account for face communication
    Index_t comBufSize =
        (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
        m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM;

    // account for edge communication
    comBufSize +=
        ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
         (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
         (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
         (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
        m_maxEdgeSize * MAX_FIELDS_PER_MPI_COMM;

    // account for corner communication
    // factor of 16 is so each buffer has its own cache line
    comBufSize +=
        ((m_rowMin & m_colMin & m_planeMin) + (m_rowMin & m_colMin & m_planeMax) +
         (m_rowMin & m_colMax & m_planeMin) + (m_rowMin & m_colMax & m_planeMax) +
         (m_rowMax & m_colMin & m_planeMin) + (m_rowMax & m_colMin & m_planeMax) +
         (m_rowMax & m_colMax & m_planeMin) + (m_rowMax & m_colMax & m_planeMax)) *
        CACHE_COHERENCE_PAD_REAL;

    this->commDataSend = Allocate<Real_t>(comBufSize);
    this->commDataRecv = Allocate<Real_t>(comBufSize);
    // prevent floating point exceptions
    memset(this->commDataSend, 0, comBufSize * sizeof(Real_t));
    memset(this->commDataRecv, 0, comBufSize * sizeof(Real_t));
#endif

    // Boundary nodesets
    if(m_colLoc == 0) Kokkos::resize(m_symmX, edgeNodes * edgeNodes);
    if(m_rowLoc == 0) Kokkos::resize(m_symmY, edgeNodes * edgeNodes);
    if(m_planeLoc == 0) Kokkos::resize(m_symmZ, edgeNodes * edgeNodes);
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI
    Index_t myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    srand(myRank);
#else
    srand(0);
    Index_t myRank = 0;
#endif
    this->numReg()     = nr;
    m_regElemSize      = Allocate<Index_t>(numReg());
    auto row_map       = Kokkos::View<Index_t*>("regElemlist::row_map", numReg() + 1);
    auto h_row_map     = Kokkos::create_mirror_view(row_map);
    auto entries       = Kokkos::View<Index_t*>("regElemlist::entries", numElem());
    m_regElemlist      = t_regElemlist(entries, row_map);
    auto h_regElemlist = typename t_regElemlist::HostMirror(
        Kokkos::create_mirror_view(m_regElemlist.entries), h_row_map);
    Index_t nextIndex = 0;
    // if we only have one region just fill it
    // Fill out the regNumList with material numbers, which are always
    // the region index plus one
    if(numReg() == 1)
    {
        while(nextIndex < numElem())
        {
            this->regNumList(nextIndex) = 1;
            nextIndex++;
        }
        regElemSize(0) = 0;
    }
    // If we have more than one region distribute the elements.
    else
    {
        Int_t                                   regionNum;
        Int_t                                   regionVar;
        Int_t                                   lastReg = -1;
        Int_t                                   binSize;
        Index_t                                 elements;
        Index_t                                 runto           = 0;
        Int_t                                   costDenominator = 0;
        Kokkos::View<Int_t*, Kokkos::HostSpace> regBinEnd("regBinEnd", numReg());
        // Determine the relative weights of all the regions.  This is based off the -b
        // flag.  Balance is the value passed into b.
        for(Index_t i = 0; i < numReg(); ++i)
        {
            regElemSize(i) = 0;
            costDenominator += pow((i + 1), balance);  // Total sum of all regions weights
            regBinEnd[i] =
                costDenominator;  // Chance of hitting a given region is (regBinEnd[i] -
                                  // regBinEdn[i-1])/costDenominator
        }
        // Until all elements are assigned
        while(nextIndex < numElem())
        {
            // pick the region
            regionVar = rand() % costDenominator;
            Index_t i = 0;
            while(regionVar >= regBinEnd[i])
                i++;
            // rotate the regions based on MPI rank.  Rotation is Rank % NumRegions this
            // makes each domain have a different region with the highest representation
            regionNum = ((i + myRank) % numReg()) + 1;
            // make sure we don't pick the same region twice in a row
            while(regionNum == lastReg)
            {
                regionVar = rand() % costDenominator;
                i         = 0;
                while(regionVar >= regBinEnd[i])
                    i++;
                regionNum = ((i + myRank) % numReg()) + 1;
            }
            // Pick the bin size of the region and determine the number of elements.
            binSize = rand() % 1000;
            if(binSize < 773)
            {
                elements = rand() % 15 + 1;
            }
            else if(binSize < 937)
            {
                elements = rand() % 16 + 16;
            }
            else if(binSize < 970)
            {
                elements = rand() % 32 + 32;
            }
            else if(binSize < 974)
            {
                elements = rand() % 64 + 64;
            }
            else if(binSize < 978)
            {
                elements = rand() % 128 + 128;
            }
            else if(binSize < 981)
            {
                elements = rand() % 256 + 256;
            }
            else
                elements = rand() % 1537 + 512;
            runto = elements + nextIndex;
            // Store the elements.  If we hit the end before we run out of elements then
            // just stop.
            while(nextIndex < runto && nextIndex < numElem())
            {
                this->regNumList(nextIndex) = regionNum;
                nextIndex++;
            }
            lastReg = regionNum;
        }
    }
    // Convert regNumList to region index sets
    // First, count size of each region
    for(Index_t i = 0; i < numElem(); ++i)
    {
        int r = this->regNumList(i) - 1;  // region index == regnum-1
        regElemSize(r)++;
    }
    // Second, allocate each region index set
    for(Index_t i = 0; i < numReg(); ++i)
    {
        h_row_map(i + 1) = regElemSize(i);
        regElemSize(i)   = 0;
    }
    // Third, fill index sets
    for(Index_t i = 0; i < numElem(); ++i)
    {
        Index_t r      = regNumList(i) - 1;  // region index == regnum-1
        Index_t regndx = regElemSize(r)++;   // Note increment
        h_regElemlist.entries(h_row_map(r) + regndx) = i;
    }
    Kokkos::deep_copy(m_regElemlist.entries, h_regElemlist.entries);
    Kokkos::deep_copy(row_map, h_row_map);
}

/////////////////////////////////////////////////////////////
void
Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
    Index_t nidx    = 0;
    auto    h_symmZ = Kokkos::create_mirror_view(m_symmZ);
    auto    h_symmY = Kokkos::create_mirror_view(m_symmY);
    auto    h_symmX = Kokkos::create_mirror_view(m_symmX);
    for(Index_t i = 0; i < edgeNodes; ++i)
    {
        Index_t planeInc = i * edgeNodes * edgeNodes;
        Index_t rowInc   = i * edgeNodes;
        for(Index_t j = 0; j < edgeNodes; ++j)
        {
            if(m_planeLoc == 0)
            {
                h_symmZ[nidx] = rowInc + j;
            }
            if(m_rowLoc == 0)
            {
                h_symmY[nidx] = planeInc + j;
            }
            if(m_colLoc == 0)
            {
                h_symmX[nidx] = planeInc + j * edgeNodes;
            }
            ++nidx;
        }
    }
    Kokkos::deep_copy(m_symmZ, h_symmZ);
    Kokkos::deep_copy(m_symmY, h_symmY);
    Kokkos::deep_copy(m_symmX, h_symmX);
}

/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Int_t edgeElems)
{
    // With C++17 we wouldn't need to do this and could run this on the GPU
    // using class lambdas
    auto h_lxim = Kokkos::create_mirror_view(m_lxim);
    auto h_lxip = Kokkos::create_mirror_view(m_lxip);
    h_lxim(0)   = 0;
    for(Index_t i = 1; i < numElem(); ++i)
    {
        h_lxim(i)     = i - 1;
        h_lxip(i - 1) = i;
    }
    h_lxip(numElem() - 1) = numElem() - 1;
    Kokkos::deep_copy(m_lxim, h_lxim);
    Kokkos::deep_copy(m_lxip, h_lxip);

    auto h_letam = Kokkos::create_mirror_view(m_letam);
    auto h_letap = Kokkos::create_mirror_view(m_letap);
    for(Index_t i = 0; i < edgeElems; ++i)
    {
        h_letam(i)                         = i;
        h_letap(numElem() - edgeElems + i) = numElem() - edgeElems + i;
    }
    for(Index_t i = edgeElems; i < numElem(); ++i)
    {
        h_letam(i)             = i - edgeElems;
        h_letap(i - edgeElems) = i;
    }
    Kokkos::deep_copy(m_letam, h_letam);
    Kokkos::deep_copy(m_letap, h_letap);

    auto h_lzetam = Kokkos::create_mirror_view(m_lzetam);
    auto h_lzetap = Kokkos::create_mirror_view(m_lzetap);
    for(Index_t i = 0; i < edgeElems * edgeElems; ++i)
    {
        h_lzetam(i) = i;
        h_lzetap(numElem() - edgeElems * edgeElems + i) =
            numElem() - edgeElems * edgeElems + i;
    }
    for(Index_t i = edgeElems * edgeElems; i < numElem(); ++i)
    {
        h_lzetam(i)                         = i - edgeElems * edgeElems;
        h_lzetap(i - edgeElems * edgeElems) = i;
    }
    Kokkos::deep_copy(m_lzetam, h_lzetam);
    Kokkos::deep_copy(m_lzetap, h_lzetap);
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Int_t edgeElems)
{
    Index_t ghostIdx[6];  // offsets to ghost locations
    auto    h_elemBC = Kokkos::create_mirror_view(m_elemBC);
    auto    h_lzetam = Kokkos::create_mirror_view(m_lzetam);
    auto    h_lzetap = Kokkos::create_mirror_view(m_lzetap);
    auto    h_letam  = Kokkos::create_mirror_view(m_letam);
    auto    h_letap  = Kokkos::create_mirror_view(m_letap);
    auto    h_lxim   = Kokkos::create_mirror_view(m_lxim);
    auto    h_lxip   = Kokkos::create_mirror_view(m_lxip);
    Kokkos::deep_copy(h_lzetam, m_lzetam);
    Kokkos::deep_copy(h_lzetap, m_lzetap);
    Kokkos::deep_copy(h_letam, m_letam);
    Kokkos::deep_copy(h_letap, m_letap);
    Kokkos::deep_copy(h_lxim, m_lxim);
    Kokkos::deep_copy(h_lxip, m_lxip);

    // set up boundary condition information
    for(Index_t i = 0; i < numElem(); ++i)
    {
        h_elemBC(i) = Int_t(0);
    }

    for(Index_t i = 0; i < 6; ++i)
    {
        ghostIdx[i] = INT_MIN;
    }

    Int_t pidx = numElem();
    if(m_planeMin != 0)
    {
        ghostIdx[0] = pidx;
        pidx += sizeX() * sizeY();
    }

    if(m_planeMax != 0)
    {
        ghostIdx[1] = pidx;
        pidx += sizeX() * sizeY();
    }

    if(m_rowMin != 0)
    {
        ghostIdx[2] = pidx;
        pidx += sizeX() * sizeZ();
    }

    if(m_rowMax != 0)
    {
        ghostIdx[3] = pidx;
        pidx += sizeX() * sizeZ();
    }

    if(m_colMin != 0)
    {
        ghostIdx[4] = pidx;
        pidx += sizeY() * sizeZ();
    }

    if(m_colMax != 0)
    {
        ghostIdx[5] = pidx;
    }

    // symmetry plane or free surface BCs
    for(Index_t i = 0; i < edgeElems; ++i)
    {
        Index_t planeInc = i * edgeElems * edgeElems;
        Index_t rowInc   = i * edgeElems;
        for(Index_t j = 0; j < edgeElems; ++j)
        {
            if(m_planeLoc == 0)
            {
                h_elemBC(rowInc + j) |= ZETA_M_SYMM;
            }
            else
            {
                h_elemBC(rowInc + j) |= ZETA_M_COMM;
                h_lzetam(rowInc + j) = ghostIdx[0] + rowInc + j;
            }

            if(m_planeLoc == m_tp - 1)
            {
                h_elemBC(rowInc + j + numElem() - edgeElems * edgeElems) |= ZETA_P_FREE;
            }
            else
            {
                h_elemBC(rowInc + j + numElem() - edgeElems * edgeElems) |= ZETA_P_COMM;
                h_lzetap(rowInc + j + numElem() - edgeElems * edgeElems) =
                    ghostIdx[1] + rowInc + j;
            }

            if(m_rowLoc == 0)
            {
                h_elemBC(planeInc + j) |= ETA_M_SYMM;
            }
            else
            {
                h_elemBC(planeInc + j) |= ETA_M_COMM;
                h_letam(planeInc + j) = ghostIdx[2] + rowInc + j;
            }

            if(m_rowLoc == m_tp - 1)
            {
                h_elemBC(planeInc + j + edgeElems * edgeElems - edgeElems) |= ETA_P_FREE;
            }
            else
            {
                h_elemBC(planeInc + j + edgeElems * edgeElems - edgeElems) |= ETA_P_COMM;
                h_letap(planeInc + j + edgeElems * edgeElems - edgeElems) =
                    ghostIdx[3] + rowInc + j;
            }

            if(m_colLoc == 0)
            {
                h_elemBC(planeInc + j * edgeElems) |= XI_M_SYMM;
            }
            else
            {
                h_elemBC(planeInc + j * edgeElems) |= XI_M_COMM;
                h_lxim(planeInc + j * edgeElems) = ghostIdx[4] + rowInc + j;
            }

            if(m_colLoc == m_tp - 1)
            {
                h_elemBC(planeInc + j * edgeElems + edgeElems - 1) |= XI_P_FREE;
            }
            else
            {
                h_elemBC(planeInc + j * edgeElems + edgeElems - 1) |= XI_P_COMM;
                h_lxip(planeInc + j * edgeElems + edgeElems - 1) =
                    ghostIdx[5] + rowInc + j;
            }
        }
    }
    Kokkos::deep_copy(m_elemBC, h_elemBC);
    Kokkos::deep_copy(m_lzetam, h_lzetam);
    Kokkos::deep_copy(m_lzetap, h_lzetap);
    Kokkos::deep_copy(m_letam, h_letam);
    Kokkos::deep_copy(m_letap, h_letap);
    Kokkos::deep_copy(m_lxim, h_lxim);
    Kokkos::deep_copy(m_lxip, h_lxip);
}

///////////////////////////////////////////////////////////////////////////
void
InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t* col, Int_t* row, Int_t* plane,
               Int_t* side)
{
    Int_t testProcs;
    Int_t dx, dy, dz;
    Int_t myDom;

    // Assume cube processor layout for now
    testProcs = Int_t(cbrt(Real_t(numRanks)) + 0.5);
    if(testProcs * testProcs * testProcs != numRanks)
    {
        printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
    }
    if(sizeof(Real_t) != 4 && sizeof(Real_t) != 8)
    {
        printf("MPI operations only support float and double right now...\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
    }
    if(MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL)
    {
        printf("corner element comm buffers too small.  Fix code.\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
    }

    dx = testProcs;
    dy = testProcs;
    dz = testProcs;

    // temporary test
    if(dx * dy * dz != numRanks)
    {
        printf("error -- must have as many domains as procs\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
    }
    Int_t remainder = dx * dy * dz % numRanks;
    if(myRank < remainder)
    {
        myDom = myRank * (1 + (dx * dy * dz / numRanks));
    }
    else
    {
        myDom = remainder * (1 + (dx * dy * dz / numRanks)) +
                (myRank - remainder) * (dx * dy * dz / numRanks);
    }

    *col   = myDom % dx;
    *row   = (myDom / dx) % dy;
    *plane = myDom / (dx * dy);
    *side  = testProcs;

    return;
}
