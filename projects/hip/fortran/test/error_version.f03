!!!!!!!!!!!!!!
! HIP runtime error-handling and version queries
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipGetLastError, hipPeekAtLastError, hipGetErrorName,
! hipGetErrorString, hipRuntimeGetVersion and hipDriverGetVersion. Deliberately
! provokes hipErrorInvalidDevice via an out-of-range hipSetDevice and validates
! the peek-vs-clear semantics.
!!!!!!!!!!!!!!
!
program error_version
  use iso_c_binding
  use hip

  implicit none

  integer(kind(hipSuccess)) :: stat, stat2
  integer(c_int)            :: ndev, rver, dver
  real(c_float), target     :: hval
  type(c_ptr)               :: dptr = c_null_ptr
  integer(c_size_t)         :: nbytes
  character(len=256)        :: ename, estr

  write(*,"(a)",advance="no") "-- Running test 'hip error_version' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))

  ! Clear any pre-existing sticky error (ignore its value).
  stat = hipGetLastError()

  ! Provoke a well-defined error: device index far past hipGetDeviceCount.
  call hipCheck(hipGetDeviceCount(ndev))
  stat = hipSetDevice(ndev + 999)
  if (stat /= hipErrorInvalidDevice) then
     write(*,*) "FAILED! hipSetDevice(bad) returned", stat, &
                " expected", int(hipErrorInvalidDevice)
     call exit(1)
  end if

  ! Peek must return the same error without clearing it.
  stat = hipPeekAtLastError()
  if (stat /= hipErrorInvalidDevice) then
     write(*,*) "FAILED! first hipPeekAtLastError returned", stat
     call exit(1)
  end if

  ! Second peek: error still set, peek never clears.
  stat2 = hipPeekAtLastError()
  if (stat2 /= hipErrorInvalidDevice) then
     write(*,*) "FAILED! second hipPeekAtLastError returned", stat2, &
                "(should still be set)"
     call exit(1)
  end if

  ! GetLastError must return the error and then clear it.
  stat = hipGetLastError()
  if (stat /= hipErrorInvalidDevice) then
     write(*,*) "FAILED! hipGetLastError returned", stat, &
                " expected", int(hipErrorInvalidDevice)
     call exit(1)
  end if

  ! State is now cleared: next GetLastError must return hipSuccess.
  stat2 = hipGetLastError()
  if (stat2 /= hipSuccess) then
     write(*,*) "FAILED! hipGetLastError after clear returned", stat2, &
                "(expected hipSuccess)"
     call exit(1)
  end if

  ! Version queries: both must return a positive integer.
  call hipCheck(hipRuntimeGetVersion(rver))
  if (rver <= 0) then
     write(*,*) "FAILED! hipRuntimeGetVersion =", rver
     call exit(1)
  end if

  call hipCheck(hipDriverGetVersion(dver))
  if (dver <= 0) then
     write(*,*) "FAILED! hipDriverGetVersion =", dver
     call exit(1)
  end if

  ! Both accessors describe the error the runtime just reported.
  ename = c_string(hipGetErrorName(hipErrorInvalidDevice))
  if (trim(ename) /= "hipErrorInvalidDevice") then
     write(*,*) "FAILED! hipGetErrorName = '", trim(ename), "'"
     call exit(1)
  end if

  estr = c_string(hipGetErrorString(hipErrorInvalidDevice))
  if (len_trim(estr) == 0 .or. index(estr, "device") == 0) then
     write(*,*) "FAILED! hipGetErrorString = '", trim(estr), "'"
     call exit(1)
  end if

  ! Confirm the runtime still accepts work after the error cycle.
  nbytes = int(4, c_size_t)   ! one real(c_float)
  call hipCheck(hipMalloc(dptr, nbytes))
  hval = 3.14
  call hipCheck(hipMemcpy(dptr, c_loc(hval), nbytes, hipMemcpyHostToDevice))
  hval = 0.0
  call hipCheck(hipMemcpy(c_loc(hval), dptr, nbytes, hipMemcpyDeviceToHost))
  if (abs(hval - 3.14) > 1.0e-6) then
     write(*,*) "FAILED! post-error memcpy roundtrip: got", hval
     call exit(1)
  end if
  call hipCheck(hipFree(dptr))

  write(*,*) "PASSED!"

contains

  ! Copy a null-terminated C string returned by reference into a Fortran string.
  function c_string(cptr) result(res)
    type(c_ptr), intent(in) :: cptr
    character(len=256) :: res
    character(kind=c_char), pointer :: chars(:)
    integer :: i

    res = " "
    if (.not. c_associated(cptr)) return
    call c_f_pointer(cptr, chars, [len(res)])
    do i = 1, len(res)
       if (chars(i) == c_null_char) exit
       res(i:i) = chars(i)
    end do
  end function c_string

end program error_version
