!!!!!!!!!!!!!!
! HIP peer access query/enable (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipDeviceCanAccessPeer (and enable/disable when a second device is
! present). On a single-GPU host a device is not its own peer, so the query must
! return 0 without error.
!!!!!!!!!!!!!!
!
program peer_access
  use iso_c_binding
  use hip

  implicit none

  integer(c_int) :: ndev, canAccess

  write(*,"(a)",advance="no") "-- Running test 'hip peer_access' (Fortran 2003 interfaces) - "

  call hipCheck(hipGetDeviceCount(ndev))
  call hipCheck(hipSetDevice(0))

  if (ndev >= 2) then
     call hipCheck(hipDeviceCanAccessPeer(canAccess, 0, 1))
     if (canAccess == 1) then
        ! Enable then disable peer access from device 0 to device 1.
        call hipCheck(hipDeviceEnablePeerAccess(1, 0))
        call hipCheck(hipDeviceDisablePeerAccess(1))
     end if
  else
     ! Single device: a device is not its own peer.
     call hipCheck(hipDeviceCanAccessPeer(canAccess, 0, 0))
     if (canAccess /= 0) then
        write(*,*) "FAILED! canAccessPeer(0,0) = ", canAccess, " (expected 0)"
        call exit(1)
     end if
  end if

  write(*,*) "PASSED!"

end program peer_access
