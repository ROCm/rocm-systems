!!!!!!!!!!!!!!
! HIP runtime cooperative kernel launch (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Launches the kernel of the existing vecadd test with
! hipLaunchCooperativeKernel on an occupancy-sized grid, checks the result, and
! confirms an oversized grid is rejected.
!!!!!!!!!!!!!!
!
program cooperative_launch
  use iso_c_binding
  use hip

  implicit none

  ! hipLaunchCooperativeKernel takes the host stub of a kernel, so the kernel is
  ! named by the mangled symbol its HIP translation unit exports.
  interface
     subroutine vector_add(out, a, b, n) bind(c, name="_Z10vector_addPdS_S_i")
       use iso_c_binding
       implicit none
       type(c_ptr), value :: out, a, b
       integer(c_int), value :: n
     end subroutine vector_add
  end interface

  integer(c_int), parameter :: n = 4096
  integer(c_int), parameter :: blocksize = 256

  real(c_double), target :: ha(n), hb(n), hout(n)
  type(c_ptr), target :: da = c_null_ptr, db = c_null_ptr, dout = c_null_ptr
  type(c_ptr), target :: args(4)
  integer(c_int), target :: nn, nblocks
  integer(c_int) :: coop, ncu, i
  integer(kind(hipSuccess)) :: stat
  integer(c_size_t) :: nbytes
  type(dim3) :: grid, block

  write(*,"(a)",advance="no") "-- Running test 'hip cooperative_launch' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipDeviceGetAttribute(coop, hipDeviceAttributeCooperativeLaunch, 0))
  if (coop == 0) then
     write(*,*) "PASSED! (cooperative launch unsupported on this device)"
     stop
  end if

  nbytes = int(n, c_size_t) * 8
  nn = n
  do i = 1, n
     ha(i) = real(i, c_double)
     hb(i) = real(2*i, c_double)
     hout(i) = 0.0_c_double
  end do

  call hipCheck(hipMalloc(da, nbytes))
  call hipCheck(hipMalloc(db, nbytes))
  call hipCheck(hipMalloc(dout, nbytes))
  call hipCheck(hipMemcpy(da, c_loc(ha(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(hb(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dout, c_loc(hout(1)), nbytes, hipMemcpyHostToDevice))

  args(1) = c_loc(dout)
  args(2) = c_loc(da)
  args(3) = c_loc(db)
  args(4) = c_loc(nn)

  ! A cooperative grid must be co-resident, so it is capped by the occupancy of
  ! the kernel times the number of compute units.
  call hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(c_loc(nblocks), &
                                                             c_funloc(vector_add), &
                                                             blocksize, 0_c_size_t))
  call hipCheck(hipDeviceGetAttribute(ncu, hipDeviceAttributeMultiprocessorCount, 0))

  grid = dim3(nblocks * ncu, 1, 1)
  block = dim3(blocksize, 1, 1)
  call hipCheck(hipLaunchCooperativeKernel(transfer(c_funloc(vector_add), c_null_ptr), &
                                           grid, block, args(1), 0, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(hout(1)), dout, nbytes, hipMemcpyDeviceToHost))
  do i = 1, n
     if (abs(hout(i) - (ha(i) + hb(i))) > 1.0d-12) then
        write(*,*) "FAILED! hout(", i, ") = ", hout(i), " expected ", ha(i) + hb(i)
        call exit(1)
     end if
  end do

  ! More blocks than can be resident at once cannot be launched cooperatively.
  grid = dim3(nblocks * ncu * 64, 1, 1)
  stat = hipLaunchCooperativeKernel(transfer(c_funloc(vector_add), c_null_ptr), &
                                    grid, block, args(1), 0, c_null_ptr)
  if (stat /= hipErrorCooperativeLaunchTooLarge) then
     write(*,*) "FAILED! oversized cooperative launch returned", stat, &
                " expected", int(hipErrorCooperativeLaunchTooLarge)
     call exit(1)
  end if
  stat = hipGetLastError()

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dout))

  write(*,*) "PASSED!"

end program cooperative_launch
