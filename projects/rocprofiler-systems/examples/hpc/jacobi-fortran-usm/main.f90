program main
  use input_mod, only: parse_arguments
  use mesh_mod, only: mesh_t
  use jacobi_mod, only: jacobi_t, init_jacobi, run_jacobi
  implicit none
  !$omp requires unified_shared_memory
  type(mesh_t) :: mesh
  type(jacobi_t) :: jacobi

  call parse_arguments(mesh)

  call init_jacobi(jacobi,mesh)
  call run_jacobi(jacobi,mesh)

end program main
