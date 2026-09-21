!!!!!!!!!!!!!!
! HIP runtime graph: empty node + 3-D memcpy node forms (Fortran 2003 interfaces)
!
! Builds a diamond-shaped graph by hand:
!   nodeH2D  (hipGraphAddMemcpyNode, 3-D parms, host->device)
!     -> nodeEmpty (hipGraphAddEmptyNode)
!        -> nodeMemset (zeros device buffer)
!        -> nodeD2H   (hipGraphAddMemcpyNode, 3-D parms, device->host)
!   nodeMemset -> nodeD2H  (completes the diamond)
! Verifies node count, then that D2H delivers the memset-zeroed values.
!!!!!!!!!!!!!!
!
program graph_empty_node
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 64    ! elements (int32)
  integer(c_int), target    :: hsrc(n), hdst(n)
  type(c_ptr) :: dptr      = c_null_ptr
  type(c_ptr) :: graph     = c_null_ptr
  type(c_ptr) :: gexec     = c_null_ptr
  type(c_ptr) :: errnode   = c_null_ptr
  type(c_ptr) :: stream    = c_null_ptr
  type(c_ptr) :: nodeH2D   = c_null_ptr
  type(c_ptr) :: nodeEmpty  = c_null_ptr
  type(c_ptr) :: nodeMemset = c_null_ptr
  type(c_ptr) :: nodeD2H   = c_null_ptr
  type(c_ptr) :: nodes_out(8)
  type(hipMemcpy3DParms) :: cp
  type(hipMemsetParams)  :: mp
  integer(c_size_t) :: nbytes
  integer(c_size_t), target :: numnodes
  integer :: i

  write(*,"(a)",advance="no") &
    "-- Running test 'hip graph_empty_node' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 4_c_size_t   ! 4 bytes per int32

  do i = 1, n
    hsrc(i) = i
  end do
  hdst = 0

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipGraphCreate(graph, 0))

  ! The 1-D array is laid out as an n x 1 x 1 volume; extent%width is in bytes.
  cp%srcArray = c_null_ptr
  cp%srcPos   = hipPos(0_c_size_t, 0_c_size_t, 0_c_size_t)
  cp%srcPtr   = hipPitchedPtr(c_loc(hsrc(1)), nbytes, int(n, c_size_t), 1_c_size_t)
  cp%dstArray = c_null_ptr
  cp%dstPos   = hipPos(0_c_size_t, 0_c_size_t, 0_c_size_t)
  cp%dstPtr   = hipPitchedPtr(dptr, nbytes, int(n, c_size_t), 1_c_size_t)
  cp%extent   = hipExtent(nbytes, 1_c_size_t, 1_c_size_t)
  cp%kind     = int(hipMemcpyHostToDevice, c_int)
  call hipCheck(hipGraphAddMemcpyNode(nodeH2D, graph, c_null_ptr, 0_c_size_t, cp))

  call hipCheck(hipGraphAddEmptyNode(nodeEmpty, graph, nodeH2D, 1_c_size_t))

  mp%dst         = dptr
  mp%elementSize = 4_c_int
  mp%width       = int(n, c_size_t)
  mp%height      = 1_c_size_t
  mp%pitch       = nbytes
  mp%value       = 0_c_int
  call hipCheck(hipGraphAddMemsetNode(nodeMemset, graph, nodeEmpty, 1_c_size_t, mp))

  cp%srcArray = c_null_ptr
  cp%srcPos   = hipPos(0_c_size_t, 0_c_size_t, 0_c_size_t)
  cp%srcPtr   = hipPitchedPtr(dptr, nbytes, int(n, c_size_t), 1_c_size_t)
  cp%dstArray = c_null_ptr
  cp%dstPos   = hipPos(0_c_size_t, 0_c_size_t, 0_c_size_t)
  cp%dstPtr   = hipPitchedPtr(c_loc(hdst(1)), nbytes, int(n, c_size_t), 1_c_size_t)
  cp%extent   = hipExtent(nbytes, 1_c_size_t, 1_c_size_t)
  cp%kind     = int(hipMemcpyDeviceToHost, c_int)
  call hipCheck(hipGraphAddMemcpyNode(nodeD2H, graph, nodeEmpty, 1_c_size_t, cp))

  ! Complete the diamond: D2H must also wait for memset so it reads zeros.
  call hipCheck(hipGraphAddDependencies(graph, nodeMemset, nodeD2H, 1_c_size_t))

  numnodes = size(nodes_out, kind=c_size_t)
  call hipCheck(hipGraphGetNodes(graph, nodes_out(1), c_loc(numnodes)))
  if (numnodes /= 4) then
    write(*,*) "FAILED! graph node count =", numnodes, " (expected 4)"
    call exit(1)
  end if

  ! ---- Instantiate, launch, synchronize ----
  call hipCheck(hipGraphInstantiate(gexec, graph, errnode, c_null_ptr, 0_c_size_t))
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))

  ! ---- Verify: memset runs after H2D and before D2H, so hdst must be zero ----
  do i = 1, n
    if (hdst(i) /= 0) then
      write(*,*) "FAILED! hdst(", i, ") =", hdst(i), " (expected 0)"
      call exit(1)
    end if
  end do

  call hipCheck(hipGraphExecDestroy(gexec))
  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program graph_empty_node
