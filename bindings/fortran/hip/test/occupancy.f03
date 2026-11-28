!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!
! HIP runtime occupancy calculator (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipOccupancyMaxPotentialBlockSize and
! hipOccupancyMaxActiveBlocksPerMultiprocessor on the kernel of
! vecadd_kernel.hip.cpp, which is compiled into this test, and cross-checks the
! results against the device properties.
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
  integer(c_int) :: gridsize, blocksize, nblocks, nblocks_small
  integer(c_size_t) :: smem_cu, max_fit

  write(*,"(a)",advance="no") "-- Running test 'hip occupancy' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipGetDeviceProperties(prop, 0))

  gridsize = 0
  blocksize = 0
  call hipCheck(hipOccupancyMaxPotentialBlockSize(gridsize, blocksize, &
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
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(nblocks, &
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
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(nblocks_small, &
                                                             c_funloc(vector_add), &
                                                             blocksize/2, 0_c_size_t))
  if (nblocks_small < nblocks) then
     write(*,*) "FAILED!", nblocks_small, "blocks of", blocksize/2, "threads but", &
                nblocks, "blocks of", blocksize
     call exit(1)
  end if

  ! Each block claiming the per-block shared memory maximum: no more blocks fit
  ! than the multiprocessor's shared memory holds, which can be more than one.
  nblocks_small = 0
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(nblocks_small, &
                                                             c_funloc(vector_add), &
                                                             blocksize, &
                                                             prop%sharedMemPerBlock))
  smem_cu = prop%maxSharedMemoryPerMultiProcessor
  if (smem_cu == 0) smem_cu = prop%sharedMemPerMultiprocessor
  if (smem_cu > 0 .and. prop%sharedMemPerBlock > 0) then
     max_fit = max(1_c_size_t, smem_cu / prop%sharedMemPerBlock)
     if (nblocks_small > max_fit) then
        write(*,*) "FAILED!", nblocks_small, "blocks of", prop%sharedMemPerBlock, &
                   "bytes of shared memory fit in", smem_cu, "bytes per multiprocessor"
        call exit(1)
     end if
  end if

  write(*,*) "PASSED!"

end program occupancy
