!!!!!!!!!!!!!!
! HIP graph memset node built from a hipMemsetParams struct (Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Adds a memset node to a graph via hipGraphAddMemsetNode (which takes a
! hipMemsetParams derived type), instantiates and launches it, and verifies the
! device buffer was set. Complements the memcpy-node / stream-capture graph
! tests with the struct-parameterised node API.
!!!!!!!!!!!!!!
!
program graph_memset_node
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  integer(c_int8_t), target :: hbuf(n)
  type(c_ptr) :: graph = c_null_ptr, gexec = c_null_ptr, errnode = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr, dptr = c_null_ptr, node = c_null_ptr
  type(hipMemsetParams) :: mp
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip graph_memset_node' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t)

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipGraphCreate(graph, 0))

  ! Describe a 1-D byte memset (value 9) over the whole buffer.
  mp%dst         = dptr
  mp%elementSize = 1
  mp%width       = nbytes   ! elements (bytes)
  mp%height      = 1
  mp%pitch       = nbytes
  mp%value       = 9

  call hipCheck(hipGraphAddMemsetNode(node, graph, c_null_ptr, 0_c_size_t, mp))

  call hipCheck(hipGraphInstantiate(gexec, graph, errnode, c_null_ptr, 0_c_size_t))
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))

  hbuf = 0
  call hipCheck(hipMemcpy(c_loc(hbuf(1)), dptr, nbytes, hipMemcpyDeviceToHost))
  do i = 1, n
     if (hbuf(i) /= 9_c_int8_t) then
        write(*,*) "FAILED! hbuf(", i, ") = ", hbuf(i), " (expected 9)"
        call exit(1)
     end if
  end do

  call hipCheck(hipGraphExecDestroy(gexec))
  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program graph_memset_node
