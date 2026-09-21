!!!!!!!!!!!!!!
! HIP runtime module / kernel API (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Loads a separately compiled code object, looks up the kernel it contains,
! queries its attributes and occupancy, then runs it as a graph kernel node,
! as an in-place update of the instantiated graph, and cooperatively.
! CTest passes the code object path in HIPFORT_TEST_CODE_OBJECT.
!!!!!!!!!!!!!!
!
program module_kernel
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 1024
  character(len=22), target :: kname = "_Z10vector_addPdS_S_i"//c_null_char

  character(len=4096) :: copath
  character(len=4097), target :: cofile
  integer :: pathlen, i
  real(c_double), target :: ha(n), hb(n), hout(n)
  type(c_ptr), target :: da = c_null_ptr, db = c_null_ptr
  type(c_ptr), target :: dout = c_null_ptr, dout2 = c_null_ptr
  integer(c_int), target :: nn
  type(c_ptr), target :: args(4), args2(4)
  type(c_ptr) :: hmod = c_null_ptr, kfunc = c_null_ptr
  type(c_ptr) :: graph = c_null_ptr, gexec = c_null_ptr, knode = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr
  type(hipKernelNodeParams) :: kparams
  integer(c_size_t) :: nbytes
  integer(c_int), target :: maxthreads
  integer(c_int) :: gridsize, blocksize, numblocks
  integer(c_int) :: coop

  write(*,"(a)",advance="no") "-- Running test 'hip module_kernel' (Fortran 2003 interfaces) - "

  call get_environment_variable("HIPFORT_TEST_CODE_OBJECT", copath, pathlen)
  if (pathlen == 0) then
     write(*,*) "FAILED! HIPFORT_TEST_CODE_OBJECT is not set"
     call exit(1)
  end if
  cofile = copath(1:pathlen)//c_null_char

  nbytes = int(n, c_size_t) * 8
  nn = n
  do i = 1, n
     ha(i) = real(i, c_double)
     hb(i) = real(2*i, c_double)
  end do

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(da, nbytes))
  call hipCheck(hipMalloc(db, nbytes))
  call hipCheck(hipMalloc(dout, nbytes))
  call hipCheck(hipMemcpy(da, c_loc(ha(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(hb(1)), nbytes, hipMemcpyHostToDevice))

  call hipCheck(hipModuleLoad(hmod, c_loc(cofile)))
  call hipCheck(hipModuleGetFunction(kfunc, hmod, c_loc(kname)))

  call hipCheck(hipFuncGetAttribute(c_loc(maxthreads), HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc))
  if (maxthreads <= 0 .or. maxthreads > 1024) then
     write(*,*) "FAILED! max threads per block = ", maxthreads
     call exit(1)
  end if

  call hipCheck(hipModuleOccupancyMaxPotentialBlockSize(gridsize, blocksize, kfunc, 0_c_size_t, 0))
  if (blocksize <= 0 .or. blocksize > maxthreads .or. gridsize <= 0) then
     write(*,*) "FAILED! occupancy block size = ", blocksize, " grid size = ", gridsize
     call exit(1)
  end if

  call hipCheck(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numblocks, kfunc, blocksize, 0_c_size_t))
  if (numblocks <= 0) then
     write(*,*) "FAILED! active blocks per CU = ", numblocks
     call exit(1)
  end if

  ! void* args[] = { &dout, &da, &db, &n }
  args(1) = c_loc(dout)
  args(2) = c_loc(da)
  args(3) = c_loc(db)
  args(4) = c_loc(nn)

  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipGraphCreate(graph, 0))

  ! Zero the output first so a node that never executes cannot pass.
  call hipCheck(hipMemset(dout, 0, nbytes))

  kparams%func = kfunc
  kparams%gridDim = dim3(4, 1, 1)
  kparams%blockDim = dim3(256, 1, 1)
  kparams%sharedMemBytes = 0
  kparams%kernelParams = c_loc(args(1))
  kparams%extra = c_null_ptr

  call hipCheck(hipGraphAddKernelNode(knode, graph, c_null_ptr, 0_c_size_t, kparams))
  call hipCheck(hipGraphInstantiate(gexec, graph, c_null_ptr, c_null_ptr, 0_c_size_t))
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))
  call check_result("hipGraphAddKernelNode", dout)

  ! Retarget the already-instantiated graph at a second output buffer.
  call hipCheck(hipMalloc(dout2, nbytes))
  call hipCheck(hipMemset(dout2, 0, nbytes))
  args2(1) = c_loc(dout2)
  args2(2) = c_loc(da)
  args2(3) = c_loc(db)
  args2(4) = c_loc(nn)
  kparams%kernelParams = c_loc(args2(1))
  call hipCheck(hipGraphExecKernelNodeSetParams(gexec, knode, kparams))
  call hipCheck(hipGraphLaunch(gexec, stream))
  call hipCheck(hipStreamSynchronize(stream))
  call check_result("hipGraphExecKernelNodeSetParams", dout2)

  ! Cooperative launch takes no extra pointer, so the argument array goes as-is.
  ! Guarded on the device attribute, as cooperative_launch.f03 does: a device
  ! that reports 0 here fails the launch with hipErrorLaunchFailure.
  call hipCheck(hipDeviceGetAttribute(coop, hipDeviceAttributeCooperativeLaunch, 0))
  if (coop /= 0) then
     call hipCheck(hipMemset(dout, 0, nbytes))
     call hipCheck(hipModuleLaunchCooperativeKernel(kfunc, 4, 1, 1, 256, 1, 1, 0, stream, args(1)))
     call hipCheck(hipStreamSynchronize(stream))
     call check_result("hipModuleLaunchCooperativeKernel", dout)
  end if

  call hipCheck(hipGraphExecDestroy(gexec))
  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipStreamDestroy(stream))
  call hipCheck(hipModuleUnload(hmod))
  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dout))
  call hipCheck(hipFree(dout2))

  write(*,*) "PASSED!"

contains

  subroutine check_result(what, buf)
    character(len=*), intent(in) :: what
    type(c_ptr), intent(in) :: buf
    integer :: j
    hout = 0.0d0
    call hipCheck(hipMemcpy(c_loc(hout(1)), buf, nbytes, hipMemcpyDeviceToHost))
    do j = 1, n
       if (hout(j) /= ha(j) + hb(j)) then
          write(*,*) "FAILED! ", what, " out(", j, ") = ", hout(j), " expected ", ha(j) + hb(j)
          call exit(1)
       end if
    end do
  end subroutine check_result

end program module_kernel
