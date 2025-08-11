
#if !defined(USE_MPI)
#    error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

// OpenMP will be compiled in if this flag is set to 1 AND the compiler beging
// used supports it (i.e. the _OPENMP symbol is defined)
#define USE_OMP 1

#if USE_MPI
#    include <mpi.h>

/*
   define one of these three symbols:

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

#    define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <Kokkos_Vector.hpp>

#include <math.h>
#include <vector>

//**************************************************
// Allow flexibility for arithmetic representations
//**************************************************

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Precision specification
typedef float       real4;
typedef double      real8;
typedef long double real10;  // 10 bytes on x86

typedef int   Index_t;  // array subscript and loop index
typedef real8 Real_t;   // floating point representation
typedef int   Int_t;    // integer representation

enum
{
    VolumeError = -1,
    QStopError  = -2
};

KOKKOS_INLINE_FUNCTION real4
SQRT(real4 arg)
{
    return sqrtf(arg);
}
KOKKOS_INLINE_FUNCTION real8
SQRT(real8 arg)
{
    return sqrt(arg);
}
KOKKOS_INLINE_FUNCTION real10
SQRT(real10 arg)
{
    return sqrtl(arg);
}

KOKKOS_INLINE_FUNCTION real4
CBRT(real4 arg)
{
    return cbrtf(arg);
}
KOKKOS_INLINE_FUNCTION real8
CBRT(real8 arg)
{
    return cbrt(arg);
}
KOKKOS_INLINE_FUNCTION real10
CBRT(real10 arg)
{
    return cbrtl(arg);
}

KOKKOS_INLINE_FUNCTION real4
FABS(real4 arg)
{
    return fabsf(arg);
}
KOKKOS_INLINE_FUNCTION real8
FABS(real8 arg)
{
    return fabs(arg);
}
KOKKOS_INLINE_FUNCTION real10
FABS(real10 arg)
{
    return fabsl(arg);
}

// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M      0x00007
#define XI_M_SYMM 0x00001
#define XI_M_FREE 0x00002
#define XI_M_COMM 0x00004

#define XI_P      0x00038
#define XI_P_SYMM 0x00008
#define XI_P_FREE 0x00010
#define XI_P_COMM 0x00020

#define ETA_M      0x001c0
#define ETA_M_SYMM 0x00040
#define ETA_M_FREE 0x00080
#define ETA_M_COMM 0x00100

#define ETA_P      0x00e00
#define ETA_P_SYMM 0x00200
#define ETA_P_FREE 0x00400
#define ETA_P_COMM 0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN     1024
#define MSG_SYNC_POS_VEL 2048
#define MSG_MONOQ        3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n)                                                              \
    (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL - 1))

//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

/*
 * The implementation of the data abstraction used for lulesh
 * resides entirely in the Domain class below.  You can change
 * grouping and interleaving of fields here to maximize data layout
 * efficiency for your underlying architecture or compiler.
 *
 * For example, fields can be implemented as STL objects or
 * raw array pointers.  As another example, individual fields
 * m_x, m_y, m_z could be budled into
 *
 *    struct { Real_t x, y, z ; } *m_coord ;
 *
 * allowing accessor functions such as
 *
 *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
 *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
 *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
 */

class Domain
{
public:
    // Constructor
    Domain(Int_t numRanks, Index_t colLoc, Index_t rowLoc, Index_t planeLoc, Index_t nx,
           Int_t tp, Int_t nr, Int_t balance, Int_t cost);

    // Destructor
    KOKKOS_FUNCTION ~Domain();

    //
    // ALLOCATION
    //

    void AllocateNodePersistent(Int_t numNode)  // Node-centered
    {
        Kokkos::resize(m_x, numNode);  // coordinates
        Kokkos::resize(m_y, numNode);
        Kokkos::resize(m_z, numNode);

        Kokkos::resize(m_xd, numNode);  // velocities
        Kokkos::resize(m_yd, numNode);
        Kokkos::resize(m_zd, numNode);

        Kokkos::resize(m_xdd, numNode);  // accelerations
        Kokkos::resize(m_ydd, numNode);
        Kokkos::resize(m_zdd, numNode);

        Kokkos::resize(m_fx, numNode);  // forces
        Kokkos::resize(m_fy, numNode);
        Kokkos::resize(m_fz, numNode);

        Kokkos::resize(m_nodalMass, numNode);  // mass

        m_c_x  = m_x;
        m_c_y  = m_y;
        m_c_z  = m_z;
        m_c_xd = m_xd;
        m_c_yd = m_yd;
        m_c_zd = m_zd;
    }

    void AllocateElemPersistent(Int_t numElem)  // Elem-centered
    {
        Kokkos::resize(m_nodelist, numElem);

        // elem connectivities through face
        Kokkos::resize(m_lxim, numElem);
        Kokkos::resize(m_lxip, numElem);
        Kokkos::resize(m_letam, numElem);
        Kokkos::resize(m_letap, numElem);
        Kokkos::resize(m_lzetam, numElem);
        Kokkos::resize(m_lzetap, numElem);

        Kokkos::resize(m_elemBC, numElem);

        Kokkos::resize(m_e, numElem);
        Kokkos::resize(m_p, numElem);

        Kokkos::resize(m_q, numElem);
        Kokkos::resize(m_ql, numElem);
        Kokkos::resize(m_qq, numElem);

        Kokkos::resize(m_v, numElem);

        Kokkos::resize(m_volo, numElem);
        Kokkos::resize(m_delv, numElem);
        Kokkos::resize(m_vdov, numElem);

        Kokkos::resize(m_arealg, numElem);

        Kokkos::resize(m_ss, numElem);

        Kokkos::resize(m_elemMass, numElem);

        Kokkos::resize(m_vnew, numElem);

        m_c_e    = m_e;
        m_c_p    = m_p;
        m_c_q    = m_q;
        m_c_ql   = m_ql;
        m_c_qq   = m_qq;
        m_c_delv = m_delv;
    }

    void AllocateGradients(Int_t numElem, Int_t allElem)
    {
        // Position gradients
        Kokkos::resize(m_delx_xi, numElem);
        Kokkos::resize(m_delx_eta, numElem);
        Kokkos::resize(m_delx_zeta, numElem);

        // Velocity gradients
        Kokkos::resize(m_delv_xi, allElem);
        Kokkos::resize(m_delv_eta, allElem);
        Kokkos::resize(m_delv_zeta, allElem);
    }

    void DeallocateGradients()
    {
        m_delx_zeta = Kokkos::View<Real_t*>();
        m_delx_eta  = Kokkos::View<Real_t*>();
        m_delx_xi   = Kokkos::View<Real_t*>();

        m_delv_zeta = Kokkos::View<Real_t*>();
        m_delv_eta  = Kokkos::View<Real_t*>();
        m_delv_xi   = Kokkos::View<Real_t*>();
    }

    void AllocateStrains(Int_t numElem)
    {
        Kokkos::resize(m_dxx, numElem);
        Kokkos::resize(m_dyy, numElem);
        Kokkos::resize(m_dzz, numElem);
    }

    void DeallocateStrains()
    {
        m_dzz = Kokkos::View<Real_t*>();
        m_dyy = Kokkos::View<Real_t*>();
        m_dxx = Kokkos::View<Real_t*>();
    }

    //
    // ACCESSORS
    //
    KOKKOS_INLINE_FUNCTION
    const Kokkos::View<Real_t*>& e_view() const { return m_e; }

    // Node-centered

    // Nodal coordinates
    KOKKOS_INLINE_FUNCTION Real_t& x(const Index_t idx) const { return m_x[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& y(const Index_t idx) const { return m_y[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& z(const Index_t idx) const { return m_z[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_x(const Index_t idx) const { return m_c_x[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_y(const Index_t idx) const { return m_c_y[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_z(const Index_t idx) const { return m_c_z[idx]; }

    // Nodal velocities
    KOKKOS_INLINE_FUNCTION Real_t& xd(const Index_t idx) const { return m_xd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& yd(const Index_t idx) const { return m_yd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& zd(const Index_t idx) const { return m_zd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_xd(const Index_t idx) const { return m_c_xd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_yd(const Index_t idx) const { return m_c_yd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_zd(const Index_t idx) const { return m_c_zd[idx]; }

    // Nodal accelerations
    KOKKOS_INLINE_FUNCTION Real_t& xdd(const Index_t idx) const { return m_xdd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& ydd(const Index_t idx) const { return m_ydd[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& zdd(const Index_t idx) const { return m_zdd[idx]; }

    // Nodal forces
    KOKKOS_INLINE_FUNCTION Real_t& fx(const Index_t idx) const { return m_fx[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& fy(const Index_t idx) const { return m_fy[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& fz(const Index_t idx) const { return m_fz[idx]; }

    // Nodal mass
    KOKKOS_INLINE_FUNCTION Real_t& nodalMass(const Index_t idx) const
    {
        return m_nodalMass[idx];
    }

    // Nodes on symmertry planes
    KOKKOS_INLINE_FUNCTION Index_t symmX(const Index_t idx) const { return m_symmX[idx]; }
    KOKKOS_INLINE_FUNCTION Index_t symmY(const Index_t idx) const { return m_symmY[idx]; }
    KOKKOS_INLINE_FUNCTION Index_t symmZ(const Index_t idx) const { return m_symmZ[idx]; }
    KOKKOS_INLINE_FUNCTION bool    symmXempty() { return m_symmX.data() == nullptr; }
    KOKKOS_INLINE_FUNCTION bool    symmYempty() { return m_symmY.data() == nullptr; }
    KOKKOS_INLINE_FUNCTION bool    symmZempty() { return m_symmZ.data() == nullptr; }

    //
    // Element-centered
    //
    Index_t& regElemSize(Index_t idx) { return m_regElemSize[idx]; }
    Index_t& regNumList(Index_t idx) { return m_regNumList[idx]; }
    Index_t* regNumList() { return &m_regNumList[0]; }
    Index_t* regElemlist(Int_t r) const
    {
        return &m_regElemlist.entries(m_regElemlist.row_map(r));
    }
    KOKKOS_INLINE_FUNCTION Index_t regElemlist(const Int_t r, Index_t idx) const
    {
        return m_regElemlist.entries(m_regElemlist.row_map(r) + idx);
    }

    KOKKOS_INLINE_FUNCTION Index_t& nodelist(Index_t i, Index_t j) const
    {
        return m_nodelist(i, j);
    }

    // elem connectivities through face
    KOKKOS_INLINE_FUNCTION Index_t& lxim(const Index_t idx) const { return m_lxim[idx]; }
    KOKKOS_INLINE_FUNCTION Index_t& lxip(const Index_t idx) const { return m_lxip[idx]; }
    KOKKOS_INLINE_FUNCTION Index_t& letam(const Index_t idx) const
    {
        return m_letam[idx];
    }
    KOKKOS_INLINE_FUNCTION Index_t& letap(const Index_t idx) const
    {
        return m_letap[idx];
    }
    KOKKOS_INLINE_FUNCTION Index_t& lzetam(const Index_t idx) const
    {
        return m_lzetam[idx];
    }
    KOKKOS_INLINE_FUNCTION Index_t& lzetap(const Index_t idx) const
    {
        return m_lzetap[idx];
    }

    // elem face symm/free-surface flag
    KOKKOS_INLINE_FUNCTION Int_t& elemBC(const Index_t idx) const
    {
        return m_elemBC[idx];
    }

    // Principal strains - temporary
    KOKKOS_INLINE_FUNCTION Real_t& dxx(const Index_t idx) const { return m_dxx[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& dyy(const Index_t idx) const { return m_dyy[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& dzz(const Index_t idx) const { return m_dzz[idx]; }

    // New relative volume - temporary
    KOKKOS_INLINE_FUNCTION Real_t& vnew(const Index_t idx) const { return m_vnew[idx]; }

    // Velocity gradient - temporary
    KOKKOS_INLINE_FUNCTION Real_t& delv_xi(const Index_t idx) const
    {
        return m_delv_xi[idx];
    }
    KOKKOS_INLINE_FUNCTION Real_t& delv_eta(const Index_t idx) const
    {
        return m_delv_eta[idx];
    }
    KOKKOS_INLINE_FUNCTION Real_t& delv_zeta(const Index_t idx) const
    {
        return m_delv_zeta[idx];
    }

    // Position gradient - temporary
    KOKKOS_INLINE_FUNCTION Real_t& delx_xi(const Index_t idx) const
    {
        return m_delx_xi[idx];
    }
    KOKKOS_INLINE_FUNCTION Real_t& delx_eta(const Index_t idx) const
    {
        return m_delx_eta[idx];
    }
    KOKKOS_INLINE_FUNCTION Real_t& delx_zeta(const Index_t idx) const
    {
        return m_delx_zeta[idx];
    }
    // Energy
    KOKKOS_INLINE_FUNCTION Real_t& e(const Index_t idx) const { return m_e[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_e(const Index_t idx) const { return m_c_e[idx]; }

    // Pressure
    KOKKOS_INLINE_FUNCTION Real_t& p(const Index_t idx) const { return m_p[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_p(const Index_t idx) const { return m_c_p[idx]; }

    // Artificial viscosity
    KOKKOS_INLINE_FUNCTION Real_t& q(const Index_t idx) const { return m_q[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_q(const Index_t idx) const { return m_c_q[idx]; }

    // Linear term for q
    KOKKOS_INLINE_FUNCTION Real_t& ql(const Index_t idx) const { return m_ql[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_ql(const Index_t idx) const { return m_c_ql[idx]; }
    // Quadratic term for q
    KOKKOS_INLINE_FUNCTION Real_t& qq(const Index_t idx) const { return m_qq[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_qq(const Index_t idx) const { return m_c_qq[idx]; }

    // Relative volume
    KOKKOS_INLINE_FUNCTION Real_t& v(const Index_t idx) const { return m_v[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t& delv(const Index_t idx) const { return m_delv[idx]; }
    KOKKOS_INLINE_FUNCTION Real_t  c_delv(const Index_t idx) const
    {
        return m_c_delv[idx];
    }

    // Reference volume
    KOKKOS_INLINE_FUNCTION Real_t& volo(Index_t idx) const { return m_volo[idx]; }

    // volume derivative over volume
    KOKKOS_INLINE_FUNCTION Real_t& vdov(Index_t idx) const { return m_vdov[idx]; }

    // Element characteristic length
    KOKKOS_INLINE_FUNCTION Real_t& arealg(Index_t idx) const { return m_arealg[idx]; }

    // Sound speed
    KOKKOS_INLINE_FUNCTION Real_t& ss(const Index_t idx) const { return m_ss[idx]; }

    // Element mass
    KOKKOS_INLINE_FUNCTION Real_t& elemMass(const Index_t idx) const
    {
        return m_elemMass[idx];
    }

    KOKKOS_INLINE_FUNCTION Index_t nodeElemCount(Index_t idx) const
    {
        return m_nodeElemStart[idx + 1] - m_nodeElemStart[idx];
    }

    KOKKOS_INLINE_FUNCTION Index_t* nodeElemCornerList(Index_t idx) const
    {
        return &m_nodeElemCornerList[m_nodeElemStart[idx]];
    }

    // Parameters

    // Cutoffs
    KOKKOS_INLINE_FUNCTION Real_t u_cut() const { return m_u_cut; }
    KOKKOS_INLINE_FUNCTION Real_t e_cut() const { return m_e_cut; }
    KOKKOS_INLINE_FUNCTION Real_t p_cut() const { return m_p_cut; }
    KOKKOS_INLINE_FUNCTION Real_t q_cut() const { return m_q_cut; }
    KOKKOS_INLINE_FUNCTION Real_t v_cut() const { return m_v_cut; }

    // Other constants (usually are settable via input file in real codes)
    KOKKOS_INLINE_FUNCTION Real_t hgcoef() const { return m_hgcoef; }
    KOKKOS_INLINE_FUNCTION Real_t qstop() const { return m_qstop; }
    KOKKOS_INLINE_FUNCTION Real_t monoq_max_slope() const { return m_monoq_max_slope; }
    KOKKOS_INLINE_FUNCTION Real_t monoq_limiter_mult() const
    {
        return m_monoq_limiter_mult;
    }
    KOKKOS_INLINE_FUNCTION Real_t ss4o3() const { return m_ss4o3; }
    KOKKOS_INLINE_FUNCTION Real_t qlc_monoq() const { return m_qlc_monoq; }
    KOKKOS_INLINE_FUNCTION Real_t qqc_monoq() const { return m_qqc_monoq; }
    KOKKOS_INLINE_FUNCTION Real_t qqc() const { return m_qqc; }

    KOKKOS_INLINE_FUNCTION Real_t eosvmax() const { return m_eosvmax; }
    KOKKOS_INLINE_FUNCTION Real_t eosvmin() const { return m_eosvmin; }
    KOKKOS_INLINE_FUNCTION Real_t pmin() const { return m_pmin; }
    KOKKOS_INLINE_FUNCTION Real_t emin() const { return m_emin; }
    KOKKOS_INLINE_FUNCTION Real_t dvovmax() const { return m_dvovmax; }
    KOKKOS_INLINE_FUNCTION Real_t refdens() const { return m_refdens; }

    // Timestep controls, etc...
    Real_t& time() { return m_time; }
    Real_t& deltatime() { return m_deltatime; }
    Real_t& deltatimemultlb() { return m_deltatimemultlb; }
    Real_t& deltatimemultub() { return m_deltatimemultub; }
    Real_t& stoptime() { return m_stoptime; }
    Real_t& dtcourant() { return m_dtcourant; }
    Real_t& dthydro() { return m_dthydro; }
    Real_t& dtmax() { return m_dtmax; }
    Real_t& dtfixed() { return m_dtfixed; }

    Int_t&   cycle() { return m_cycle; }
    Index_t& numRanks() { return m_numRanks; }

    Index_t& colLoc() { return m_colLoc; }
    Index_t& rowLoc() { return m_rowLoc; }
    Index_t& planeLoc() { return m_planeLoc; }
    Index_t& tp() { return m_tp; }

    Index_t& sizeX() { return m_sizeX; }
    Index_t& sizeY() { return m_sizeY; }
    Index_t& sizeZ() { return m_sizeZ; }
    Index_t& numReg() { return m_numReg; }
    Int_t&   cost() { return m_cost; }
    Index_t& numElem() { return m_numElem; }
    Index_t& numNode() { return m_numNode; }

    Index_t& maxPlaneSize() { return m_maxPlaneSize; }
    Index_t& maxEdgeSize() { return m_maxEdgeSize; }

    //
    // MPI-Related additional data
    //

#if USE_MPI
    // Communication Work space
    Real_t* commDataSend;
    Real_t* commDataRecv;

    // Maximum number of block neighbors
    MPI_Request recvRequest[26];  // 6 faces + 12 edges + 8 corners
    MPI_Request sendRequest[26];  // 6 faces + 12 edges + 8 corners
#endif

private:
    void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
    void SetupThreadSupportStructures();
    void CreateRegionIndexSets(Int_t nreg, Int_t balance);
    void SetupCommBuffers(Int_t edgeNodes);
    void SetupSymmetryPlanes(Int_t edgeNodes);
    void SetupElementConnectivities(Int_t edgeElems);
    void SetupBoundaryConditions(Int_t edgeElems);

    //
    // IMPLEMENTATION
    //

    /* Node-centered */
    Kokkos::View<Real_t*> m_x; /* coordinates */
    Kokkos::View<Real_t*> m_y;
    Kokkos::View<Real_t*> m_z;
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_x; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_y; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_z; /* coordinates */

    Kokkos::View<Real_t*> m_xd; /* velocities */
    Kokkos::View<Real_t*> m_yd;
    Kokkos::View<Real_t*> m_zd;
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_xd; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_yd; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_zd; /* coordinates */

    Kokkos::View<Real_t*> m_xdd; /* accelerations */
    Kokkos::View<Real_t*> m_ydd;
    Kokkos::View<Real_t*> m_zdd;

    Kokkos::View<Real_t*> m_fx; /* forces */
    Kokkos::View<Real_t*> m_fy;
    Kokkos::View<Real_t*> m_fz;

    Kokkos::View<Real_t*> m_nodalMass; /* mass */

    Kokkos::View<Index_t*> m_symmX; /* symmetry plane nodesets */
    Kokkos::View<Index_t*> m_symmY;
    Kokkos::View<Index_t*> m_symmZ;

    // Element-centered

    // Region information
    Int_t    m_numReg;
    Int_t    m_cost;         // imbalance cost
    Index_t* m_regElemSize;  // Size of region sets
    Index_t* m_regNumList;   // Region number per domain element
    // Index_t **m_regElemlist; // region indexset
    using t_regElemlist =
        Kokkos::StaticCrsGraph<Index_t, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace,
                               Kokkos::MemoryTraits<0>, Index_t>;
    t_regElemlist m_regElemlist;

    Kokkos::View<Index_t* [8], Kokkos::LayoutRight>
        m_nodelist; /* elemToNode connectivity */

    Kokkos::View<Index_t*> m_lxim; /* element connectivity across each face */
    Kokkos::View<Index_t*> m_lxip;
    Kokkos::View<Index_t*> m_letam;
    Kokkos::View<Index_t*> m_letap;
    Kokkos::View<Index_t*> m_lzetam;
    Kokkos::View<Index_t*> m_lzetap;

    Kokkos::View<Int_t*> m_elemBC; /* symmetry/free-surface flags for each elem face */

    Kokkos::View<Real_t*> m_dxx; /* principal strains -- temporary */
    Kokkos::View<Real_t*> m_dyy;
    Kokkos::View<Real_t*> m_dzz;

    Kokkos::View<Real_t*> m_delv_xi; /* velocity gradient -- temporary */
    Kokkos::View<Real_t*> m_delv_eta;
    Kokkos::View<Real_t*> m_delv_zeta;

    Kokkos::View<Real_t*> m_delx_xi; /* coordinate gradient -- temporary */
    Kokkos::View<Real_t*> m_delx_eta;
    Kokkos::View<Real_t*> m_delx_zeta;

    Kokkos::View<Real_t*> m_e; /* energy */

    Kokkos::View<Real_t*> m_p;  /* pressure */
    Kokkos::View<Real_t*> m_q;  /* q */
    Kokkos::View<Real_t*> m_ql; /* linear term for q */
    Kokkos::View<Real_t*> m_qq; /* quadratic term for q */

    Kokkos::View<Real_t*> m_v;    /* relative volume */
    Kokkos::View<Real_t*> m_volo; /* reference volume */
    Kokkos::View<Real_t*> m_vnew; /* new relative volume -- temporary */
    Kokkos::View<Real_t*> m_delv; /* m_vnew - m_v */
    Kokkos::View<Real_t*> m_vdov; /* volume derivative over volume */

    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_e; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_p; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_q; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_ql; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_qq; /* coordinates */
    Kokkos::View<const Real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        m_c_delv; /* coordinates */

    Kokkos::View<Real_t*> m_arealg; /* characteristic length of an element */

    Kokkos::View<Real_t*> m_ss; /* "sound speed" */

    Kokkos::View<Real_t*> m_elemMass; /* mass */

    // Cutoffs (treat as constants)
    const Real_t m_e_cut;  // energy tolerance
    const Real_t m_p_cut;  // pressure tolerance
    const Real_t m_q_cut;  // q tolerance
    const Real_t m_v_cut;  // relative volume tolerance
    const Real_t m_u_cut;  // velocity tolerance

    // Other constants (usually setable, but hardcoded in this proxy app)

    const Real_t m_hgcoef;  // hourglass control
    const Real_t m_ss4o3;
    const Real_t m_qstop;  // excessive q indicator
    const Real_t m_monoq_max_slope;
    const Real_t m_monoq_limiter_mult;
    const Real_t m_qlc_monoq;  // linear term coef for q
    const Real_t m_qqc_monoq;  // quadratic term coef for q
    const Real_t m_qqc;
    const Real_t m_eosvmax;
    const Real_t m_eosvmin;
    const Real_t m_pmin;     // pressure floor
    const Real_t m_emin;     // energy floor
    const Real_t m_dvovmax;  // maximum allowable volume change
    const Real_t m_refdens;  // reference density

    // Variables to keep track of timestep, simulation time, and cycle
    Real_t m_dtcourant;  // courant constraint
    Real_t m_dthydro;    // volume change constraint
    Int_t  m_cycle;      // iteration count for simulation
    Real_t m_dtfixed;    // fixed time increment
    Real_t m_time;       // current time
    Real_t m_deltatime;  // variable time increment
    Real_t m_deltatimemultlb;
    Real_t m_deltatimemultub;
    Real_t m_dtmax;     // maximum allowable time increment
    Real_t m_stoptime;  // end time for simulation

    Int_t m_numRanks;

    Index_t m_colLoc;
    Index_t m_rowLoc;
    Index_t m_planeLoc;
    Index_t m_tp;

    Index_t m_sizeX;
    Index_t m_sizeY;
    Index_t m_sizeZ;
    Index_t m_numElem;
    Index_t m_numNode;

    Index_t m_maxPlaneSize;
    Index_t m_maxEdgeSize;

    // OMP hack
    Kokkos::View<Index_t*> m_nodeElemStart;
    Kokkos::View<Index_t*> m_nodeElemCornerList;

    // Used in setup
    Index_t m_rowMin, m_rowMax;
    Index_t m_colMin, m_colMax;
    Index_t m_planeMin, m_planeMax;
};
typedef Real_t& (Domain::*Domain_member)(Index_t) const;

struct cmdLineOpts
{
    Int_t its;        // -i
    Int_t nx;         // -s
    Int_t numReg;     // -r
    Int_t numFiles;   // -f
    Int_t showProg;   // -p
    Int_t quiet;      // -q
    Int_t viz;        // -v
    Int_t cost;       // -c
    Int_t balance;    // -b
    Int_t do_atomic;  // -a
};

// Function Prototypes

// lulesh-par
/*Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]);*/

// lulesh-util
void
ParseCommandLineOptions(int argc, char* argv[], Int_t myRank, struct cmdLineOpts* opts);
void
VerifyAndWriteFinalOutput(Real_t elapsed_time, Domain& locDom, Int_t nx, Int_t numRanks);

// lulesh-viz
void
DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void
CommRecv(Domain& domain, Int_t msgType, Index_t xferFields, Index_t dx, Index_t dy,
         Index_t dz, bool doRecv, bool planeOnly);
void
CommSend(Domain& domain, Int_t msgType, Index_t xferFields, Domain_member* fieldData,
         Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly);
void
CommSBN(Domain& domain, Int_t xferFields, Domain_member* fieldData);
void
CommSyncPosVel(Domain& domain);
void
CommMonoQ(Domain& domain);

// lulesh-init
void
InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t* col, Int_t* row, Int_t* plane,
               Int_t* side);

/*********************************/
/* Data structure implementation */
/*********************************/

/* might want to add access methods so that memory can be */
/* better managed, as in luleshFT */

template <typename T>
T*
Allocate(size_t size)
{
    return static_cast<T*>(
        Kokkos::kokkos_malloc<Kokkos::HostSpace>(sizeof(T) * size + 8));
}

template <typename T>
void
Release(T** ptr)
{
    if(*ptr != NULL)
    {
        Kokkos::kokkos_free<Kokkos::HostSpace>(*ptr);
        *ptr = NULL;
    }
}

struct MinFinder
{
    Real_t val;
    int    i;
    KOKKOS_INLINE_FUNCTION

    MinFinder()
    : val(100000000000000000000.0000)
    , i(-1)
    {}

    KOKKOS_INLINE_FUNCTION
    MinFinder(const double& val_, const int& i_)
    : val(val_)
    , i(i_)
    {}

    KOKKOS_INLINE_FUNCTION
    MinFinder(const MinFinder& src)
    : val(src.val)
    , i(src.i)
    {}

    // overloading += operator to do the max assignment
    KOKKOS_INLINE_FUNCTION
    void operator+=(MinFinder& src)
    {
        if(src.val < val)
        {
            val = src.val;
            i   = src.i;
        }
    }
    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile MinFinder& src) volatile
    {
        if(src.val < val)
        {
            val = src.val;
            i   = src.i;
        }
    }
};

struct reduce_double3
{
    double x, y, z;
    KOKKOS_INLINE_FUNCTION
    reduce_double3()
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }
    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile reduce_double3& src) volatile
    {
        x += src.x;
        y += src.y;
        z += src.z;
    }
    KOKKOS_INLINE_FUNCTION
    void operator+=(const reduce_double3& src)
    {
        x += src.x;
        y += src.y;
        z += src.z;
    }
};
