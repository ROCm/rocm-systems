!!!!!!!!!!!!!!
! HIP runtime occupancy calculator (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipOccupancyMaxPotentialBlockSize and
! hipOccupancyMaxActiveBlocksPerMultiprocessor on the kernel of the existing
! vecadd test, and cross-checks the results against the device properties.
!!!!!!!!!!!!!!
!
program occupancy
  use iso_c_binding
  use hip

  implicit none

  ! The occupancy entry points take the host stub of a kernel, so the kernel is
  ! named by the mangled symbol its HIP translation unit exports.
  interface
     subroutine vector_add(out, a, b, n) bind(c, name="_Z10vector_addPdS_S_i")
       use iso_c_binding
       implicit none
       type(c_ptr), value :: out, a, b
       integer(c_int), value :: n
     end subroutine vector_add
  end interface

  type(hipDeviceProp_t)  :: prop
  integer(c_int), target :: gridsize, blocksize, nblocks, nblocks_small

  write(*,"(a)",advance="no") "-- Running test 'hip occupancy' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipGetDeviceProperties(prop, 0))

  gridsize = 0
  blocksize = 0
  call hipCheck(hipOccupancyMaxPotentialBlockSize(c_loc(gridsize), c_loc(blocksize), &
                                                  c_funloc(vector_add), 0_c_size_t, 0))
  if (blocksize <= 0 .or. blocksize > prop%maxThreadsPerBlock) then
     write(*,*) "FAILED! hipOccupancyMaxPotentialBlockSize block size", blocksize, &
                " outside 1 ..", prop%maxThreadsPerBlock
     call exit(1)
  end if
  if (gridsize <= 0) then
     write(*,*) "FAILED! hipOccupancyMaxPotentialBlockSize grid size", gridsize
     call exit(1)
  end if

  nblocks = 0
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(c_loc(nblocks), &
                                                             c_funloc(vector_add), &
                                                             blocksize, 0_c_size_t))
  if (nblocks <= 0) then
     write(*,*) "FAILED! hipOccupancyMaxActiveBlocksPerMultiprocessor returned", nblocks
     call exit(1)
  end if
  if (nblocks * blocksize > prop%maxThreadsPerMultiProcessor) then
     write(*,*) "FAILED! occupancy", nblocks, "blocks of", blocksize, &
                "threads exceeds", prop%maxThreadsPerMultiProcessor
     call exit(1)
  end if

  ! Smaller blocks can never fit fewer times on a multiprocessor.
  nblocks_small = 0
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(c_loc(nblocks_small), &
                                                             c_funloc(vector_add), &
                                                             blocksize/2, 0_c_size_t))
  if (nblocks_small < nblocks) then
     write(*,*) "FAILED!", nblocks_small, "blocks of", blocksize/2, "threads but", &
                nblocks, "blocks of", blocksize
     call exit(1)
  end if

  ! Requesting all of the shared memory leaves room for a single block at most.
  nblocks_small = 0
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(c_loc(nblocks_small), &
                                                             c_funloc(vector_add), &
                                                             blocksize, &
                                                             prop%sharedMemPerBlock))
  if (nblocks_small > 1) then
     write(*,*) "FAILED!", nblocks_small, "blocks fit while each claims all shared memory"
     call exit(1)
  end if

  write(*,*) "PASSED!"

end program occupancy
