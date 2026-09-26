!!!!!!!!!!!!!!
! HIP runtime hipMallocManaged (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Allocates unified (managed) memory, writes it from the host, overwrites it
! from the device (hipMemset), and checks the host sees the device's write.
!!!!!!!!!!!!!!
!
program malloc_managed
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  type(c_ptr) :: mptr = c_null_ptr
  real(c_double), pointer :: p(:)
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip malloc_managed' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMallocManaged(mptr, nbytes, hipMemAttachGlobal))
  call c_f_pointer(mptr, p, [n])

  ! Host writes the managed buffer.
  do i = 1, n
     p(i) = real(i, c_double)
  end do

  ! Device overwrites it with zero bytes.
  call hipCheck(hipMemset(mptr, 0, nbytes))
  call hipCheck(hipDeviceSynchronize())

  ! Host must observe the device's write.
  do i = 1, n
     if (p(i) /= 0.0d0) then
        write(*,*) "FAILED! p(", i, ") = ", p(i), " (expected 0)"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(mptr))

  write(*,*) "PASSED!"

end program malloc_managed
