!!!!!!!!!!!!!!
! HIP runtime graphs built from explicit nodes (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Builds a graph by hand rather than by stream capture: two 1-D memcpy nodes
! (H2D then D2H) linked with an explicit dependency, then instantiated and
! launched. Exercises hipGraphCreate, hipGraphAddMemcpyNode1D,
! hipGraphAddDependencies, hipGraphGetNodes, hipGraphInstantiate,
! hipGraphLaunch, hipGraphExecDestroy and hipGraphDestroy.
!!!!!!!!!!!!!!
!
program graph_nodes
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  real(c_double), target :: hsrc(n), hdst(n)
  type(c_ptr) :: graph = c_null_ptr, gexec = c_null_ptr, errnode = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr, dptr = c_null_ptr
  type(c_ptr) :: nodeH2D = c_null_ptr, nodeD2H = c_null_ptr
  integer(c_size_t), target :: numnodes
  type(c_ptr) :: nodes_out(8)   ! capacity buffer for hipGraphGetNodes
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip graph_nodes' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8
  do i = 1, n
     hsrc(i) = real(i, c_double)
  end do
  hdst = 0.0d0

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipGraphCreate(graph, 0))

  ! Two 1-D memcpy nodes, initially without dependencies.
  call hipCheck(hipGraphAddMemcpyNode1D(nodeH2D, graph, c_null_ptr, 0_c_size_t, &
                                        dptr, c_loc(hsrc(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipGraphAddMemcpyNode1D(nodeD2H, graph, c_null_ptr, 0_c_size_t, &
                                        c_loc(hdst(1)), dptr, nbytes, hipMemcpyDeviceToHost))

  ! Make the device->host copy depend on the host->device copy.
  call hipCheck(hipGraphAddDependencies(graph, nodeH2D, nodeD2H, 1_c_size_t))

  ! Query the node count: pass an array and its capacity in numnodes; on return
  ! numnodes holds the actual count. (nodes is a by-reference c_ptr in the
  ! binding, so a real capacity buffer is used rather than a null query.)
  numnodes = size(nodes_out, kind=c_size_t)
  call hipCheck(hipGraphGetNodes(graph, nodes_out(1), c_loc(numnodes)))
  if (numnodes /= 2) then
     write(*,*) "FAILED! graph node count = ", numnodes, " (expected 2)"
     call exit(1)
  end if

  ! Instantiate and launch.
  call hipCheck(hipGraphInstantiate(gexec, graph, errnode, c_null_ptr, 0_c_size_t))
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))

  do i = 1, n
     if (hdst(i) /= hsrc(i)) then
        write(*,*) "FAILED! hdst(", i, ") = ", hdst(i), " expected ", hsrc(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipGraphExecDestroy(gexec))
  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program graph_nodes
