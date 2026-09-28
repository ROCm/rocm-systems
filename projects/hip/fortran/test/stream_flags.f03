!!!!!!!!!!!!!!
! HIP runtime stream flag and priority API
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipDeviceGetStreamPriorityRange, hipStreamCreateWithFlags,
! hipStreamCreateWithPriority, hipStreamGetFlags, hipStreamGetPriority,
! and hipStreamQuery.
!!!!!!!!!!!!!!
!
program stream_flags
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 1048576   ! 4 MB of int32
  integer(c_int), target    :: hsrc(n), hdst(n)
  type(c_ptr)    :: dptr = c_null_ptr
  type(c_ptr)    :: strm_nb = c_null_ptr, strm_pri = c_null_ptr
  integer(c_int) :: least, greatest, got_flags, got_pri
  integer(c_size_t) :: nbytes
  integer :: i
  integer(kind(hipSuccess)) :: qret

  write(*,"(a)",advance="no") &
    "-- Running test 'hip stream_flags' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 4   ! 4 bytes per int32

  ! HIP convention: lower number means higher priority, so greatest <= least.
  call hipCheck(hipSetDevice(0))
  call hipCheck(hipDeviceGetStreamPriorityRange(least, greatest))

  write(*,"(a,i0,a,i0,a)",advance="no") &
    "[priority range: least=", least, " greatest=", greatest, "] "

  if (least < greatest) then
     write(*,*) "FAILED! leastPriority (", least, &
       ") should be >= greatestPriority (", greatest, ")"
     call exit(1)
  end if

  ! Non-blocking stream: the flags must read back as they were set.
  call hipCheck(hipStreamCreateWithFlags(strm_nb, hipStreamNonBlocking))
  if (.not. c_associated(strm_nb)) then
     write(*,*) "FAILED! strm_nb is null after hipStreamCreateWithFlags"
     call exit(1)
  end if

  got_flags = -1
  call hipCheck(hipStreamGetFlags(strm_nb, got_flags))
  if (got_flags /= hipStreamNonBlocking) then
     write(*,*) "FAILED! hipStreamGetFlags returned ", got_flags, &
       " (expected hipStreamNonBlocking =", hipStreamNonBlocking, ")"
     call exit(1)
  end if

  ! Ask for the highest-priority value so the round trip is unambiguous even
  ! when the range spans a single value.
  call hipCheck(hipStreamCreateWithPriority(strm_pri, hipStreamDefault, greatest))
  if (.not. c_associated(strm_pri)) then
     write(*,*) "FAILED! strm_pri is null after hipStreamCreateWithPriority"
     call exit(1)
  end if

  got_pri = -999
  call hipCheck(hipStreamGetPriority(strm_pri, got_pri))
  if (got_pri /= greatest) then
     write(*,*) "FAILED! hipStreamGetPriority returned ", got_pri, &
       " (expected ", greatest, ")"
     call exit(1)
  end if

  ! Real work on the priority stream: after sync the stream must report idle
  ! and the data must be what the copy and memset left behind.
  do i = 1, n
     hsrc(i) = i
  end do
  hdst = 0

  call hipCheck(hipMalloc(dptr, nbytes))

  call hipCheck(hipMemcpyAsync(dptr, c_loc(hsrc(1)), nbytes, hipMemcpyHostToDevice, strm_pri))
  ! overwrite first half with a known pattern so we can detect stale data
  call hipCheck(hipMemsetAsync(dptr, 0, nbytes / 2, strm_pri))

  call hipCheck(hipStreamSynchronize(strm_pri))

  qret = hipStreamQuery(strm_pri)
  if (qret /= hipSuccess) then
     write(*,*) "FAILED! hipStreamQuery returned ", qret, " (expected hipSuccess=0)"
     call exit(1)
  end if

  call hipCheck(hipMemcpy(c_loc(hdst(1)), dptr, nbytes, hipMemcpyDeviceToHost))

  ! first half was memset to 0
  do i = 1, n / 2
     if (hdst(i) /= 0) then
        write(*,*) "FAILED! hdst(", i, ") = ", hdst(i), " (expected 0 from memset)"
        call exit(1)
     end if
  end do
  ! second half carries the original copy
  do i = n / 2 + 1, n
     if (hdst(i) /= i) then
        write(*,*) "FAILED! hdst(", i, ") = ", hdst(i), " (expected ", i, ")"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(strm_nb))
  call hipCheck(hipStreamDestroy(strm_pri))

  write(*,*) "PASSED!"

end program stream_flags
