module input_mod
  use, intrinsic :: ISO_Fortran_env, only: stdout=>output_unit, stderr=>error_unit
  use kind_mod
  use mesh_mod, only: mesh_t, init_mesh
  implicit none

  private
  public :: debug, parse_arguments

  logical :: debug
  character(len=4) :: debug_string
  integer(IK), parameter :: default_domain_size = 4096

contains

  subroutine parse_arguments(mesh)
    type(mesh_t), intent(out) :: mesh
    integer(IK) :: argc, n_x, n_y
    character(len=:), allocatable :: arg

    argc = command_argument_count()
    select case(argc)
      case (0)
        call init_mesh(mesh,default_domain_size,default_domain_size)
      case (1)
        call print_help_and_exit()
      case (2)
        select case(get_argument(1))
          case ('-m','--mesh')
            arg = get_argument(2)
            read(arg,*) n_x
            call init_mesh(mesh,n_x,n_x)
          case default
            call print_help_and_exit()
        end select
      case (3)
        select case(get_argument(1))
          case ('-m','--mesh')
            arg = get_argument(2)
            read(arg,*) n_x
            arg = get_argument(3)
            read(arg,*) n_y
            call init_mesh(mesh,n_x,n_y)
          case default
            call print_help_and_exit()
        end select
      case default
        call print_help_and_exit()
    end select

    call get_environment_variable("JACOBI_DEBUG", value=debug_string)
    select case(trim(debug_string))
      case ("TRUE","true","True","1")
        debug = .true.
      case default
        debug = .false.
    end select

  end subroutine parse_arguments

  function get_argument(n) result(arg)
    integer(IK), intent(in) :: n
    character(len=:), allocatable :: arg
    integer(IK) :: len

    call get_command_argument(n, length=len)
    allocate(character(len) :: arg)
    call get_command_argument(n,arg)
    if ( arg.eq.'-h'.or.arg.eq.'--help') call print_help_and_exit()
  end function get_argument

  subroutine print_help_and_exit()
    character(len=:), allocatable :: program_name

    program_name = get_argument(0)
    write(stderr,*) 'Usage: ',program_name,' [-m Mesh.X [Mesh.Y]] [-h | --help]'
    write(stderr,*) ' -m | --mesh Mesh.x [Mesh.y]: set the domain size per node'
    write(stderr,*) '    (if "Mesh.y" is missing, the domain size will default to (Mesh.x, Mesh.x);'
    write(stderr,*) '    Mesh.x and Mesh.y must be positive integers)'
    write(stderr,*) ' -h | --help: print help information'

    error stop 1
  end subroutine print_help_and_exit

end module input_mod
