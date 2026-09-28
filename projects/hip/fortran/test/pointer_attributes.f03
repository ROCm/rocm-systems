!!!!!!!!!!!!!!
! HIP runtime hipPointerGetAttributes (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Queries the attributes of the three allocation kinds hipfort can produce -
! plain device memory, managed memory and pinned host memory - and checks the
! runtime reports the expected memory type, owning device and pointers.
!!!!!!!!!!!!!!
!
program pointer_attributes
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  type(c_ptr) :: dptr = c_null_ptr, mptr = c_null_ptr, hptr = c_null_ptr
  type(hipPointerAttribute_t) :: attr
  integer(c_size_t) :: nbytes

  write(*,"(a)",advance="no") "-- Running test 'hip pointer_attributes' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  call hipCheck(hipSetDevice(0))

  ! Device memory: device type, owned by device 0, no host address.
  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipPointerGetAttributes(attr, dptr))
  call expect_type(attr%type, hipMemoryTypeDevice, "hipMalloc")
  if (attr%device /= 0) then
     write(*,*) "FAILED! hipMalloc device = ", attr%device, " (expected 0)"
     call exit(1)
  end if
  if (.not. c_associated(attr%devicePointer, dptr)) then
     write(*,*) "FAILED! hipMalloc devicePointer does not match the allocation"
     call exit(1)
  end if
  if (attr%isManaged /= 0) then
     write(*,*) "FAILED! hipMalloc isManaged = ", attr%isManaged, " (expected 0)"
     call exit(1)
  end if

  ! Managed memory: reported as managed and addressable from both sides.
  call hipCheck(hipMallocManaged(mptr, nbytes, hipMemAttachGlobal))
  call hipCheck(hipPointerGetAttributes(attr, mptr))
  call expect_type(attr%type, hipMemoryTypeManaged, "hipMallocManaged")
  if (attr%isManaged /= 1) then
     write(*,*) "FAILED! hipMallocManaged isManaged = ", attr%isManaged, " (expected 1)"
     call exit(1)
  end if
  if (.not. c_associated(attr%hostPointer, mptr)) then
     write(*,*) "FAILED! hipMallocManaged hostPointer does not match the allocation"
     call exit(1)
  end if

  ! Pinned host memory: host type with a valid device-side address.
  call hipCheck(hipHostMalloc(hptr, nbytes, hipHostMallocDefault))
  call hipCheck(hipPointerGetAttributes(attr, hptr))
  call expect_type(attr%type, hipMemoryTypeHost, "hipHostMalloc")
  if (.not. c_associated(attr%hostPointer, hptr)) then
     write(*,*) "FAILED! hipHostMalloc hostPointer does not match the allocation"
     call exit(1)
  end if
  if (.not. c_associated(attr%devicePointer)) then
     write(*,*) "FAILED! hipHostMalloc devicePointer is null"
     call exit(1)
  end if

  call hipCheck(hipFree(dptr))
  call hipCheck(hipFree(mptr))
  call hipCheck(hipHostFree(hptr))

  write(*,*) "PASSED!"

contains

  subroutine expect_type(got, want, what)
    integer(c_int), intent(in) :: got
    integer(kind(hipMemoryTypeDevice)), intent(in) :: want
    character(len=*), intent(in) :: what
    if (got /= int(want, c_int)) then
       write(*,*) "FAILED! ", what, " memory type = ", got, " (expected ", int(want, c_int), ")"
       call exit(1)
    end if
  end subroutine expect_type

end program pointer_attributes
