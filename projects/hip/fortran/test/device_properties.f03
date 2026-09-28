!!!!!!!!!!!!!!
! HIP runtime device-properties test
!
! Exercises hipSetDeviceFlags, hipGetDeviceProperties (via hipfort_auxiliary),
! cross-checks warpSize and multiProcessorCount against hipDeviceGetAttribute,
! then verifies the runtime is still functional after hipDeviceReset.
!!!!!!!!!!!!!!
!
program device_properties
  use iso_c_binding
  use hip

  implicit none

  type(hipDeviceProp_t) :: prop
  integer(c_int)        :: warp_attr, nmp_attr
  integer(c_int)        :: namelen, i
  character(len=256)    :: devname
  type(c_ptr)           :: dptr
  integer(c_int), target :: hbuf(64), hbuf2(64)
  integer(c_size_t)     :: nbytes

  write(*,"(a)",advance="no") "-- Running test 'hip device_properties' (Fortran 2003 interfaces) - "

  ! must be called before any context is created
  call hipCheck(hipSetDeviceFlags(hipDeviceScheduleAuto))

  call hipCheck(hipSetDevice(0))

  call hipCheck(hipGetDeviceProperties(prop, 0))

  ! extract null-terminated device name
  devname = " "
  namelen = 0
  do i = 1, 256
    if (prop%name(i) == c_null_char) exit
    devname(i:i) = prop%name(i)
    namelen = i
  end do
  if (namelen == 0) then
    write(*,*) "FAILED! device name is empty"
    call exit(1)
  end if

  if (prop%totalGlobalMem <= 0) then
    write(*,*) "FAILED! totalGlobalMem = ", prop%totalGlobalMem
    call exit(1)
  end if

  if (prop%warpSize /= 32 .and. prop%warpSize /= 64) then
    write(*,*) "FAILED! warpSize = ", prop%warpSize, " (expected 32 or 64)"
    call exit(1)
  end if

  if (prop%multiProcessorCount <= 0) then
    write(*,*) "FAILED! multiProcessorCount = ", prop%multiProcessorCount
    call exit(1)
  end if

  if (prop%maxThreadsPerBlock <= 0) then
    write(*,*) "FAILED! maxThreadsPerBlock = ", prop%maxThreadsPerBlock
    call exit(1)
  end if

  if (prop%clockRate <= 0) then
    write(*,*) "FAILED! clockRate = ", prop%clockRate
    call exit(1)
  end if

  ! cross-check via hipDeviceGetAttribute: agreement between the two APIs
  call hipCheck(hipDeviceGetAttribute(warp_attr, hipDeviceAttributeWarpSize, 0))
  if (warp_attr /= prop%warpSize) then
    write(*,*) "FAILED! warpSize mismatch: prop=", prop%warpSize, " attr=", warp_attr
    call exit(1)
  end if

  call hipCheck(hipDeviceGetAttribute(nmp_attr, hipDeviceAttributeMultiprocessorCount, 0))
  if (nmp_attr /= prop%multiProcessorCount) then
    write(*,*) "FAILED! multiProcessorCount mismatch: prop=", prop%multiProcessorCount, " attr=", nmp_attr
    call exit(1)
  end if

  call hipCheck(hipDeviceReset())

  ! prove the runtime is still usable after reset
  nbytes = int(64, c_size_t) * c_sizeof(hbuf(1))
  do i = 1, 64
    hbuf(i) = i
  end do
  hbuf2 = 0

  call hipCheck(hipMalloc(dptr, nbytes))
  call hipCheck(hipMemcpy(dptr, c_loc(hbuf(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(c_loc(hbuf2(1)), dptr, nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dptr))

  do i = 1, 64
    if (hbuf2(i) /= i) then
      write(*,*) "FAILED! post-reset memcpy round-trip at i=", i, " got=", hbuf2(i)
      call exit(1)
    end if
  end do

  write(*,*) "PASSED!"

end program device_properties
