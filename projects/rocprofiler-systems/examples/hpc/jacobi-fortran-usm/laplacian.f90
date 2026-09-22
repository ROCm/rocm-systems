module laplacian_mod
  use kind_mod
  use mesh_mod, only: mesh_t
  implicit none
  !$omp requires unified_shared_memory

  private
  public :: laplacian

contains

  subroutine laplacian(mesh,u,au)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: u(:,:)
    real(RK), intent(inout) :: au(:,:)
    integer(IK) :: i,j
    real(RK) :: invdx2,invdy2

    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    !$omp target teams distribute parallel do collapse(2)
    do j = 2,mesh%n_y-1
      do i = 2,mesh%n_x-1
        au(i,j) = (-u(i-1,j)+2._RK*u(i,j)-u(i+1,j))*invdx2 &
                + (-u(i,j-1)+2._RK*u(i,j)-u(i,j+1))*invdy2
      end do
    end do

  end subroutine laplacian

end module laplacian_mod
