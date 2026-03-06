module update_mod
  use kind_mod
  use mesh_mod, only: mesh_t
  implicit none
  !$omp requires unified_shared_memory

  private
  public :: update

contains

  subroutine update(mesh,rhs,au,u,res)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: rhs(:,:), au(:,:)
    real(RK), intent(inout) :: u(:,:)
    real(RK), intent(inout) :: res(:,:)
    integer(IK) :: i,j
    real(RK) :: temp,factor

    factor = (2._RK/mesh%dx**2+2._RK/mesh%dy**2)**-1

    !$omp target teams distribute parallel do collapse(2) private(temp)
    do j = 1,mesh%n_y
      do i = 1,mesh%n_x
        temp = rhs(i,j) - au(i,j)
        res(i,j) = temp
        u(i,j) = u(i,j) + temp*factor
      end do
    end do
  end subroutine

end module update_mod
