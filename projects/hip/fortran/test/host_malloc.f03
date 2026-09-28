!!!!!!!!!!!!!!
! HIP runtime hipHostMalloc (pinned host memory, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Allocates pinned host memory, uses it as the source of a host->device copy,
! copies back into a plain host array and verifies the round trip.
!!!!!!!!!!!!!!
!
program host_malloc
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  type(c_ptr) :: hpinned = c_null_ptr, dptr = c_null_ptr
  real(c_double), pointer :: hp(:)
  real(c_double), target  :: hcheck(n)
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip host_malloc' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipHostMalloc(hpinned, nbytes, hipHostMallocDefault))
  call c_f_pointer(hpinned, hp, [n])

  do i = 1, n
     hp(i) = real(i, c_double)
  end do

  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipMemcpy(dptr, hpinned, nbytes, hipMemcpyHostToDevice))

  hcheck = 0.0d0
  call hipCheck(hipMemcpy(c_loc(hcheck(1)), dptr, nbytes, hipMemcpyDeviceToHost))

  do i = 1, n
     if (hcheck(i) /= hp(i)) then
        write(*,*) "FAILED! hcheck(", i, ") = ", hcheck(i), " expected ", hp(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr))
  call hipCheck(hipHostFree(hpinned))

  write(*,*) "PASSED!"

end program host_malloc
