module kind_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real32, real64
  implicit none
  integer(int32), parameter :: IK = int32
#ifdef SINGLE_PRECISION
  integer(IK), parameter :: RK = real32
#else
  integer(IK), parameter :: RK = real64
#endif
end module kind_mod
