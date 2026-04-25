module jacobi_mod
  use, intrinsic :: ISO_Fortran_env, only : stdout=>output_unit
  use kind_mod
  use input_mod, only: debug
  use mesh_mod, only: mesh_t
  use norm_mod, only: norm
  use laplacian_mod, only: laplacian
  use boundary_mod, only: boundary_conditions
  use update_mod, only: update
  use omp_lib, only: omp_get_wtime
  use iso_c_binding
  use hipfort
  use hipfort_roctx
  implicit none

  private
  public :: jacobi_t, init_jacobi, run_jacobi

  real(RK), parameter :: pi = 4._RK*atan(1._RK)
  real(RK), parameter :: tolerance = 1.e-5_RK
  integer(IK) :: max_iters = 1000

  type :: jacobi_t
    real(RK), allocatable :: u(:,:), rhs(:,:), au(:,:), res(:,:)
    real(RK) :: t_start, t_stop, elapsed
    integer(IK) :: iters
  end type jacobi_t

contains

  subroutine init_jacobi(this,mesh)
    type(jacobi_t), intent(inout) :: this
    type(mesh_t), intent(inout) :: mesh
    integer(IK) :: i,j
    real(RK) :: rhs_bc
    integer :: ret

    ret = roctxRangePush("init"//c_null_char)
    allocate(this%u(mesh%n_x,mesh%n_y))
    allocate(this%au(mesh%n_x,mesh%n_y))
    allocate(this%rhs(mesh%n_x,mesh%n_y))
    allocate(this%res(mesh%n_x,mesh%n_y))

    ! Initialize values
    this%u = 0._RK
    this%rhs = 0._RK
    this%au = 0._RK

     ! Add forcing function to rhs
    do i=1,mesh%n_x
      rhs_bc = cos(pi*mesh%x(i))/mesh%dx**2
      this%rhs(i,1) = this%rhs(i,1) + rhs_bc
      this%rhs(i,mesh%n_y) = this%rhs(i,mesh%n_y) + rhs_bc
    end do
    do j=1,mesh%n_y
      rhs_bc = cos(pi*mesh%y(j))/mesh%dy**2
      this%rhs(1,j) = this%rhs(1,j) + rhs_bc
      this%rhs(mesh%n_x,j) = this%rhs(mesh%n_x,j) + rhs_bc
    end do

    this%res = this%rhs

    !$omp target enter data map(to:this%u,this%rhs,this%au,this%res)

    if (debug) then
      max_iters=10
      write(stdout,'(*(E12.3E2))') mesh%x
      write(stdout,'(*(E12.3E2))') mesh%y
      write(*,*)
      call print_2D(this%rhs)
      write(*,*)
    end if
    ret = roctxRangePop()

  end subroutine init_jacobi

  subroutine run_jacobi(this,mesh)
    type(jacobi_t), intent(inout) :: this
    type(mesh_t), intent(inout) :: mesh

    real(RK) :: resid
    integer :: ret

    write(stdout,'(A)') 'Starting Jacobi run'
    ret = roctxRangePush("run"//c_null_char)
    this%iters = 0

    resid = norm(mesh, this%res)
    write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid

    this%t_start = omp_get_wtime()

    do while (this%iters < max_iters .and. resid > tolerance)
      ! Compute Laplacian
      call laplacian(mesh,this%u,this%au)
      if (debug) then
        !$omp target update from(this%au)
        call print_2D(this%au)
        write(stdout,*)
      end if

      ! Apply boundary conditions
      call boundary_conditions(mesh,this%u,this%au)
      if (debug) then
        !$omp target update from(this%au)
        call print_2D(this%au)
        write(stdout,*)
      end if

      ! Update the solution
      call update(mesh,this%rhs,this%au,this%u,this%res)
      if (debug) then
        !$omp target update from(this%u)
        call print_2D(this%u)
        write(stdout,*)
        !$omp target update from(this%res)
        call print_2D(this%res)
        write(stdout,*)
        write(stdout,*)
      end if

      ! Compute residual = ||U||
      resid = norm(mesh,this%res)

      this%iters = this%iters + 1
      if (debug) write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid
      if (mod(this%iters,100) == 0 .and. .not. debug) write(stdout,'(A,I4,A,ES11.5)') 'Iteration: ',this%iters,' - Residual: ',resid
    end do

    this%t_stop = omp_get_wtime()
    this%elapsed = this%t_stop - this%t_start

    write(stdout,'(A,I4,A,ES11.5)') 'Stopped after ',this%iters,' iterations with residue: ',resid

    call print_results(this,mesh)
    ret = roctxRangePop()

  end subroutine run_jacobi

  subroutine print_2D(array)
    real(RK), intent(in) :: array(:,:)

    integer(IK) :: row, array_shape(2)

    array_shape = shape(array)
    do row = 1, array_shape(1)
      write(stdout,'(*(E12.3E2))') array(row,:)
    end do
  end subroutine print_2D

  subroutine print_results(this,mesh)
    type(jacobi_t), intent(in) :: this
    type(mesh_t), intent(in) :: mesh
    real(RK) :: lattice_updates, flops, bandwidth

    write(stdout,'(A,F0.3,A)') 'Total Jacobi run time: ',this%elapsed,' sec.'

    lattice_updates = real(mesh%n_x,RK)*mesh%n_y*this%iters
    flops = 17._RK*lattice_updates
    bandwidth = 12._RK*lattice_updates*RK

    write(stdout,'(A,F0.3,A)') 'Measured lattice updates: ',lattice_updates/this%elapsed/1.e9_RK,' LU/s'
    write(stdout,'(A,F0.1,A)') 'Effective Flops: ',flops/this%elapsed/1.e9_RK,' GFlops'
    write(stdout,'(A,F0.3,A)') 'Effective device bandwidth: ',bandwidth/this%elapsed/1.e12_RK,' TB/s'
    write(stdout,'(A,F0.3)') 'Effective AI=',flops/bandwidth

  end subroutine print_results

end module jacobi_mod
