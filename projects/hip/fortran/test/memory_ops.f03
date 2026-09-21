!!!!!!!!!!!!!!
! HIP runtime 2-D and typed memory operations
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipMemset2D (pitched 2-D byte fill), hipMemsetD32 (32-bit word fill),
! hipMemcpy2DAsync (async D2D pitched copy on a stream), and hipMemGetInfo.
!!!!!!!!!!!!!!
!
program memory_ops
  use iso_c_binding
  use hip

  implicit none

  ! matrix: 8 int32 words per row, 4 rows
  integer, parameter :: NCOLS = 8, NROWS = 4, COL_BYTES = NCOLS * 4

  ! DEADBEEF as signed int32 (two's complement: -559038737)
  integer(c_int), parameter :: DEADBEEF = int(z'DEADBEEF', c_int)

  type(c_ptr) :: dptr_a = c_null_ptr  ! pitched src (memset2D + async test)
  type(c_ptr) :: dptr_b = c_null_ptr  ! flat buffer (memsetD32 test)
  type(c_ptr) :: dptr_c = c_null_ptr  ! pitched dst (async test)
  type(c_ptr) :: stream = c_null_ptr

  integer(c_size_t) :: pitch_a, pitch_c
  integer(c_size_t) :: free_mem, total_mem

  integer(c_int8_t), target :: hbuf8(COL_BYTES * NROWS)
  integer(c_int),    target :: hbuf_i(NCOLS * NROWS)
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip memory_ops' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))

  call hipCheck(hipMemGetInfo(free_mem, total_mem))
  if (total_mem == 0_c_size_t) then
     write(*,*) "FAILED! hipMemGetInfo total = 0"
     call exit(1)
  end if
  if (free_mem > total_mem) then
     write(*,*) "FAILED! hipMemGetInfo free > total:", free_mem, ">", total_mem
     call exit(1)
  end if

  ! Pitched buffer COL_BYTES wide by NROWS rows; the driver picks pitch_a.
  call hipCheck(hipMallocPitch(dptr_a, pitch_a, int(COL_BYTES, c_size_t), int(NROWS, c_size_t)))

  call hipCheck(hipMemset2D(dptr_a, pitch_a, int(z'42', c_int), &
                            int(COL_BYTES, c_size_t), int(NROWS, c_size_t)))

  hbuf8 = 0_c_int8_t
  call hipCheck(hipMemcpy2D(c_loc(hbuf8(1)), int(COL_BYTES, c_size_t), &
                            dptr_a, pitch_a, &
                            int(COL_BYTES, c_size_t), int(NROWS, c_size_t), &
                            hipMemcpyDeviceToHost))

  do i = 1, COL_BYTES * NROWS
     if (hbuf8(i) /= int(z'42', c_int8_t)) then
        write(*,"(a,i0,a,i0,a)") "FAILED! hipMemset2D: hbuf8(", i, ") = ", hbuf8(i), &
                                  " (expected 0x42 = 66)"
        call exit(1)
     end if
  end do

  call hipCheck(hipMalloc(dptr_b, int(NCOLS * NROWS * 4, c_size_t)))
  call hipCheck(hipMemsetD32(dptr_b, DEADBEEF, int(NCOLS * NROWS, c_size_t)))

  hbuf_i = 0
  call hipCheck(hipMemcpy(c_loc(hbuf_i(1)), dptr_b, &
                          int(NCOLS * NROWS * 4, c_size_t), hipMemcpyDeviceToHost))

  do i = 1, NCOLS * NROWS
     if (hbuf_i(i) /= DEADBEEF) then
        write(*,"(a,i0,a,z8.8,a,z8.8,a)") "FAILED! hipMemsetD32: hbuf_i(", i, ") = 0x", &
              hbuf_i(i), " (expected 0x", DEADBEEF, ")"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr_b))
  dptr_b = c_null_ptr

  call hipCheck(hipMallocPitch(dptr_c, pitch_c, int(COL_BYTES, c_size_t), int(NROWS, c_size_t)))
  call hipCheck(hipStreamCreate(stream))

  call hipCheck(hipMemcpy2DAsync(dptr_c, pitch_c, dptr_a, pitch_a, &
                                 int(COL_BYTES, c_size_t), int(NROWS, c_size_t), &
                                 hipMemcpyDeviceToDevice, stream))
  call hipCheck(hipStreamSynchronize(stream))

  hbuf8 = 0_c_int8_t
  call hipCheck(hipMemcpy2D(c_loc(hbuf8(1)), int(COL_BYTES, c_size_t), &
                            dptr_c, pitch_c, &
                            int(COL_BYTES, c_size_t), int(NROWS, c_size_t), &
                            hipMemcpyDeviceToHost))

  do i = 1, COL_BYTES * NROWS
     if (hbuf8(i) /= int(z'42', c_int8_t)) then
        write(*,"(a,i0,a,i0,a)") "FAILED! hipMemcpy2DAsync: hbuf8(", i, ") = ", hbuf8(i), &
                                  " (expected 0x42 = 66)"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr_a))
  call hipCheck(hipFree(dptr_c))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program memory_ops
