#include "lulesh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VIZ_MESH

#    ifdef __cplusplus
extern "C"
{
#    endif
#    include "silo.h"
#    if USE_MPI
#        include "pmpio.h"
#    endif
#    ifdef __cplusplus
}
#    endif

// Function prototypes
static void
DumpDomainToVisit(DBfile* db, Domain& domain, int myRank);
static

#    if USE_MPI
    // For some reason, earlier versions of g++ (e.g. 4.2) won't let me
    // put the 'static' qualifier on this prototype, even if it's done
    // consistently in the prototype and definition
    void
    DumpMultiblockObjects(DBfile* db, PMPIO_baton_t* bat, char basename[], int numRanks);

// Callback prototypes for PMPIO interface (only useful if we're
// running parallel)
static void*
LULESH_PMPIO_Create(const char* fname, const char* dname, void* udata);
static void*
LULESH_PMPIO_Open(const char* fname, const char* dname, PMPIO_iomode_t ioMode,
                  void* udata);
static void
LULESH_PMPIO_Close(void* file, void* udata);

#    else
    void
    DumpMultiblockObjects(DBfile* db, char basename[], int numRanks);
#    endif

/**********************************************************************/
void
DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks)
{
    char    subdirName[32];
    char    basename[32];
    DBfile* db;

    sprintf(basename, "lulesh_plot_c%d", domain.cycle());
    sprintf(subdirName, "data_%d", myRank);

#    if USE_MPI

    PMPIO_baton_t* bat =
        PMPIO_Init(numFiles, PMPIO_WRITE, MPI_COMM_WORLD, 10101, LULESH_PMPIO_Create,
                   LULESH_PMPIO_Open, LULESH_PMPIO_Close, NULL);

    int myiorank = PMPIO_GroupRank(bat, myRank);

    char fileName[64];

    if(myiorank == 0)
        strcpy(fileName, basename);
    else
        sprintf(fileName, "%s.%03d", basename, myiorank);

    db = (DBfile*) PMPIO_WaitForBaton(bat, fileName, subdirName);

    DumpDomainToVisit(db, domain, myRank);

    // Processor 0 writes out bit of extra data to its file that
    // describes how to stitch all the pieces together
    if(myRank == 0)
    {
        DumpMultiblockObjects(db, bat, basename, numRanks);
    }

    PMPIO_HandOffBaton(bat, db);

    PMPIO_Finish(bat);
#    else

    db = (DBfile*) DBCreate(basename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5X);

    if(db)
    {
        DBMkDir(db, subdirName);
        DBSetDir(db, subdirName);
        DumpDomainToVisit(db, domain, myRank);
        DumpMultiblockObjects(db, basename, numRanks);
    }
    else
    {
        printf("Error writing out viz file - rank %d\n", myRank);
    }

#    endif
}

/**********************************************************************/

static void
DumpDomainToVisit(DBfile* db, Domain& domain, int myRank)
{
    int ok = 0;

    /* Create an option list that will give some hints to VisIt for
     * printing out the cycle and time in the annotations */
    DBoptlist* optlist;

    /* Write out the mesh connectivity in fully unstructured format */
    int  shapetype[1] = { DB_ZONETYPE_HEX };
    int  shapesize[1] = { 8 };
    int  shapecnt[1]  = { domain.numElem() };
    int* conn         = Allocate<int>(domain.numElem() * 8);
    int  ci           = 0;
    for(int ei = 0; ei < domain.numElem(); ++ei)
    {
        Index_t* elemToNode = domain.nodelist(ei);
        for(int ni = 0; ni < 8; ++ni)
        {
            conn[ci++] = elemToNode[ni];
        }
    }
    ok += DBPutZonelist2(db, "connectivity", domain.numElem(), 3, conn,
                         domain.numElem() * 8, 0, 0, 0, /* Not carrying ghost zones */
                         shapetype, shapesize, shapecnt, 1, NULL);
    Release<int>(&conn);

    /* Write out the mesh coordinates associated with the mesh */
    const char* coordnames[3] = { "X", "Y", "Z" };
    float*      coords[3];
    coords[0] = Allocate<float>(domain.numNode());
    coords[1] = Allocate<float>(domain.numNode());
    coords[2] = Allocate<float>(domain.numNode());
    for(int ni = 0; ni < domain.numNode(); ++ni)
    {
        coords[0][ni] = float(domain.x(ni));
        coords[1][ni] = float(domain.y(ni));
        coords[2][ni] = float(domain.z(ni));
    }
    optlist = DBMakeOptlist(2);
    ok += DBAddOption(optlist, DBOPT_DTIME, &domain.time());
    ok += DBAddOption(optlist, DBOPT_CYCLE, &domain.cycle());
    ok += DBPutUcdmesh(db, "mesh", 3, (char**) &coordnames[0], (float**) coords,
                       domain.numNode(), domain.numElem(), "connectivity", 0, DB_FLOAT,
                       optlist);
    ok += DBFreeOptlist(optlist);
    Release<float>(&coords[2]);
    Release<float>(&coords[1]);
    Release<float>(&coords[0]);

    /* Write out the materials */
    int* matnums = Allocate<int>(domain.numReg());
    int  dims[1] = { domain.numElem() };  // No mixed elements
    for(int i = 0; i < domain.numReg(); ++i)
        matnums[i] = i + 1;

    ok += DBPutMaterial(db, "regions", "mesh", domain.numReg(), matnums,
                        domain.regNumList(), dims, 1, NULL, NULL, NULL, NULL, 0, DB_FLOAT,
                        NULL);
    Release<int>(&matnums);

    /* Write out pressure, energy, relvol, q */

    float* e = Allocate<float>(domain.numElem());
    for(int ei = 0; ei < domain.numElem(); ++ei)
    {
        e[ei] = float(domain.e(ei));
    }
    ok += DBPutUcdvar1(db, "e", "mesh", e, domain.numElem(), NULL, 0, DB_FLOAT,
                       DB_ZONECENT, NULL);
    Release<float>(&e);

    float* p = Allocate<float>(domain.numElem());
    for(int ei = 0; ei < domain.numElem(); ++ei)
    {
        p[ei] = float(domain.p(ei));
    }
    ok += DBPutUcdvar1(db, "p", "mesh", p, domain.numElem(), NULL, 0, DB_FLOAT,
                       DB_ZONECENT, NULL);
    Release<float>(&p);

    float* v = Allocate<float>(domain.numElem());
    for(int ei = 0; ei < domain.numElem(); ++ei)
    {
        v[ei] = float(domain.v(ei));
    }
    ok += DBPutUcdvar1(db, "v", "mesh", v, domain.numElem(), NULL, 0, DB_FLOAT,
                       DB_ZONECENT, NULL);
    Release<float>(&v);

    float* q = Allocate<float>(domain.numElem());
    for(int ei = 0; ei < domain.numElem(); ++ei)
    {
        q[ei] = float(domain.q(ei));
    }
    ok += DBPutUcdvar1(db, "q", "mesh", q, domain.numElem(), NULL, 0, DB_FLOAT,
                       DB_ZONECENT, NULL);
    Release<float>(&q);

    /* Write out nodal speed, velocities */
    float* zd    = Allocate<float>(domain.numNode());
    float* yd    = Allocate<float>(domain.numNode());
    float* xd    = Allocate<float>(domain.numNode());
    float* speed = Allocate<float>(domain.numNode());
    for(int ni = 0; ni < domain.numNode(); ++ni)
    {
        xd[ni] = float(domain.xd(ni));
        yd[ni] = float(domain.yd(ni));
        zd[ni] = float(domain.zd(ni));
        speed[ni] =
            float(sqrt((xd[ni] * xd[ni]) + (yd[ni] * yd[ni]) + (zd[ni] * zd[ni])));
    }

    ok += DBPutUcdvar1(db, "speed", "mesh", speed, domain.numNode(), NULL, 0, DB_FLOAT,
                       DB_NODECENT, NULL);
    Release<float>(&speed);

    ok += DBPutUcdvar1(db, "xd", "mesh", xd, domain.numNode(), NULL, 0, DB_FLOAT,
                       DB_NODECENT, NULL);
    Release<float>(&xd);

    ok += DBPutUcdvar1(db, "yd", "mesh", yd, domain.numNode(), NULL, 0, DB_FLOAT,
                       DB_NODECENT, NULL);
    Release<float>(&yd);

    ok += DBPutUcdvar1(db, "zd", "mesh", zd, domain.numNode(), NULL, 0, DB_FLOAT,
                       DB_NODECENT, NULL);
    Release<float>(&zd);

    if(ok != 0)
    {
        printf("Error writing out viz file - rank %d\n", myRank);
    }
}

/**********************************************************************/

#    if USE_MPI
void
DumpMultiblockObjects(DBfile* db, PMPIO_baton_t* bat, char basename[], int numRanks)
#    else
void
DumpMultiblockObjects(DBfile* db, char basename[], int numRanks)
#    endif
{
    /* MULTIBLOCK objects to tie together multiple files */
    char**  multimeshObjs;
    char**  multimatObjs;
    char*** multivarObjs;
    int*    blockTypes;
    int*    varTypes;
    int     ok = 0;
    // Make sure this list matches what's written out above
    char vars[][10] = { "p", "e", "v", "q", "speed", "xd", "yd", "zd" };
    int  numvars    = sizeof(vars) / sizeof(vars[0]);

    // Reset to the root directory of the silo file
    DBSetDir(db, "/");

    // Allocate a bunch of space for building up the string names
    multimeshObjs = Allocate<char*>(numRanks);
    multimatObjs  = Allocate<char*>(numRanks);
    multivarObjs  = Allocate<char**>(numvars);
    blockTypes    = Allocate<int>(numRanks);
    varTypes      = Allocate<int>(numRanks);

    for(int v = 0; v < numvars; ++v)
    {
        multivarObjs[v] = Allocate<char*>(numRanks);
    }

    for(int i = 0; i < numRanks; ++i)
    {
        multimeshObjs[i] = Allocate<char>(64);
        multimatObjs[i]  = Allocate<char>(64);
        for(int v = 0; v < numvars; ++v)
        {
            multivarObjs[v][i] = Allocate<char>(64);
        }
        blockTypes[i] = DB_UCDMESH;
        varTypes[i]   = DB_UCDVAR;
    }

    // Build up the multiobject names
    for(int i = 0; i < numRanks; ++i)
    {
#    if USE_MPI
        int iorank = PMPIO_GroupRank(bat, i);
#    else
        int iorank = 0;
#    endif

        // delete multivarObjs[i];
        if(iorank == 0)
        {
            snprintf(multimeshObjs[i], 64, "/data_%d/mesh", i);
            snprintf(multimatObjs[i], 64, "/data_%d/regions", i);
            for(int v = 0; v < numvars; ++v)
            {
                snprintf(multivarObjs[v][i], 64, "/data_%d/%s", i, vars[v]);
            }
        }
        else
        {
            snprintf(multimeshObjs[i], 64, "%s.%03d:/data_%d/mesh", basename, iorank, i);
            snprintf(multimatObjs[i], 64, "%s.%03d:/data_%d/regions", basename, iorank,
                     i);
            for(int v = 0; v < numvars; ++v)
            {
                snprintf(multivarObjs[v][i], 64, "%s.%03d:/data_%d/%s", basename, iorank,
                         i, vars[v]);
            }
        }
    }

    // Now write out the objects
    ok += DBPutMultimesh(db, "mesh", numRanks, (char**) multimeshObjs, blockTypes, NULL);
    ok += DBPutMultimat(db, "regions", numRanks, (char**) multimatObjs, NULL);
    for(int v = 0; v < numvars; ++v)
    {
        ok += DBPutMultivar(db, vars[v], numRanks, (char**) multivarObjs[v], varTypes,
                            NULL);
    }

    for(int v = 0; v < numvars; ++v)
    {
        for(int i = 0; i < numRanks; i++)
        {
            Release<char>(&multivarObjs[v][i]);
        }
        Release<char*>(&multivarObjs[v]);
    }

    // Clean up
    for(int i = 0; i < numRanks; i++)
    {
        Release<char>(&multimeshObjs[i]);
        Release<char>(&multimatObjs[i]);
    }
    Release<char*>(&multimeshObjs);
    Release<char*>(&multimatObjs);
    Release<char**>(&multivarObjs);
    Release<int>(&blockTypes);
    Release<int>(&varTypes);

    if(ok != 0)
    {
        printf("Error writing out multiXXX objs to viz file - rank 0\n");
    }
}

#    if USE_MPI

/**********************************************************************/

static void*
LULESH_PMPIO_Create(const char* fname, const char* dname, void* udata)
{
    /* Create the file */
    DBfile* db = DBCreate(fname, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5X);

    /* Put the data in a subdirectory, so VisIt only sees the multimesh
     * objects we write out in the base file */
    if(db)
    {
        DBMkDir(db, dname);
        DBSetDir(db, dname);
    }
    return (void*) db;
}

/**********************************************************************/

static void*
LULESH_PMPIO_Open(const char* fname, const char* dname, PMPIO_iomode_t ioMode,
                  void* udata)
{
    /* Open the file */
    DBfile* db = DBOpen(fname, DB_UNKNOWN, DB_APPEND);

    /* Put the data in a subdirectory, so VisIt only sees the multimesh
     * objects we write out in the base file */
    if(db)
    {
        DBMkDir(db, dname);
        DBSetDir(db, dname);
    }
    return (void*) db;
}

/**********************************************************************/

static void
LULESH_PMPIO_Close(void* file, void* udata)
{
    DBfile* db = (DBfile*) file;
    if(db) DBClose(db);
}
#    endif

#else

void
DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks)
{
    if(myRank == 0)
    {
        printf("Must enable -DVIZ_MESH at compile time to call DumpDomain\n");
    }
}

#endif
