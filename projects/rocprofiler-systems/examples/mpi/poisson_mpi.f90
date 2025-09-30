program main

!*****************************************************************************80
!
!! POISSON_MPI is a program for solving the Poisson problem using MPI.
!
!  Discussion:
!
!    The Poisson equation
!
!      DEL^2 U(X,Y) = F(X,Y)
!
!    is solved on the unit square [0,1] x [0,1] using a grid of [0,NX+1] by
!    [0,NY+1] evenly spaced points.
!
!    The boundary conditions and F are set so that the exact solution is
!
!      U(X,Y) = sin ( X * Y )
!
!    The Jacobi iteration is repeatedly applied until convergence is detected.
!
!    For parallel execution, the domain is divided up into parallel horizontal
!    strips.  Each strip is assigned to a process.  A process must communicate
!    with the processes "above" and "below" it to get information about the
!    solution value along their common interface.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 October 2002
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    William Gropp, Ewing Lusk, Anthony Skjellum,
!    Using MPI: Portable Parallel Programming with the
!    Message-Passing Interface,
!    Second Edition,
!    MIT Press, 1999,
!    ISBN: 0262571323.
!
  use mpi

  integer, parameter :: ndim = 1
  integer, parameter :: nx = 9
  integer, parameter :: ny = 9

  real, allocatable, dimension ( :, : ) :: a
  real anorm
  real, allocatable, dimension ( :, : ) :: b
  real bnorm
  real bnorm_part
  integer bot
  integer cartesian_comm
  integer cartesian_id
  integer, parameter :: cartesian_master = 0
  logical converged
  real diff
  real diff_part
  integer dim
  integer dims(ndim)
  real dx
  real dy
  integer e
  integer error
  real, allocatable, dimension ( :, : ) :: f
  real fnorm
  integer i
  integer it
  integer, parameter :: it_max = 100
  integer j
  integer num_procs
  logical periodic(ndim)
  logical reorder
  integer s
  integer shift
  integer top
  real, parameter :: tolerance = 0.001E+00
  integer world_id
  integer, parameter :: world_master = 0
!
!  Initialize MPI.
!
  call MPI_Init ( error )
!
!  Get the number of processes.
!
  call MPI_Comm_size ( MPI_COMM_WORLD, num_procs, error )
!
!  Get the individual process ID.
!
  call MPI_Comm_rank ( MPI_COMM_WORLD, world_id, error )
!
!  Print a message.
!
  if ( world_id == world_master ) then
    call timestamp ( )
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'POISSON_MPI - Master process:'
    write ( *, '(a)' ) '  FORTRAN90 version'
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) '  A program to solve the Poisson equation.'
    write ( *, '(a)' ) '  The MPI message passing library is used.'
    write ( *, '(a)' ) ' '
    write ( *, '(a,i8)' ) '  The number of interior X grid lines is ', nx
    write ( *, '(a,i8)' ) '  The number of interior Y grid lines is ', ny
    write ( *, '(a)' ) ' '
    write ( *, '(a,i3)' ) '  The number of processes is ', num_procs
    write ( *, '(a)' ) ' '
  end if

  write ( *, '(a,i3,a)' ) '  Process ', world_id, ' is active'
!
!  Get a communicator for the Cartesian decomposition of the domain.
!
  dims(1) = num_procs
  periodic(1) = .false.
  reorder = .true.

  call MPI_Cart_create ( MPI_COMM_WORLD, ndim, dims, periodic, reorder, &
    cartesian_comm, error )
!
!  Get this processor's rank within the Cartesian communicator.
!
  call MPI_Comm_rank ( cartesian_comm, cartesian_id, error )
!
!  Determine the bottom and top neighbors of each processor.
!
!  Note that MPI_Cart_shift counts dimensions using 0-based arithmetic!
!  We only have one dimension, but to inquire about a shift in dimension 1
!  we actually ask about dimension 0!
!
  dim = 0
  shift = 1

  call MPI_Cart_shift ( cartesian_comm, dim, shift, bot, top, error )
!
!  Compute the rows "S" through "E" of data that are to be assigned to this
!  process.
!
!  (There's an MPE routine MPE_DECOMP1D that does this, but I don't know where
!  MPE is, so I wrote my own!)
!
  call decomp_band ( ny, num_procs, cartesian_id, s, e )
!
!  Once we have the sizes S and E, we can allocate A, B, and F.
!
  allocate ( a(0:nx+1,s-1:e+1) )
  allocate ( b(0:nx+1,s-1:e+1) )
  allocate ( f(0:nx+1,s-1:e+1) )
!
!  Initialize the right hand side and initial solution guess.
!
  call init_band ( nx, ny, s, e, cartesian_id, cartesian_comm, num_procs, &
    dx, dy, f, a, b )

  call get_norm ( nx, s, e, f, cartesian_comm, cartesian_master, fnorm )

  if ( cartesian_id == cartesian_master ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'POISSON_MPI - Master Cartesian process:'
    write ( *, '(a,g14.6)' ) &
      '  Max norm of right hand side F at interior nodes = ', fnorm
  end if
!
!  Do the iteration.
!  We do two steps of iteration each time.
!
  converged = .false.

  if ( cartesian_id == cartesian_master ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'Step    ||U||         ||Unew||     ||Unew-U||'
    write ( *, '(a)' ) ' '
  end if

  do it = 1, it_max
!
!  Share with neighbor processes the value of A at "ghost" points.
!
    call exchange_band ( nx, s, e, cartesian_comm, bot, top, a )
!
!  Perform a Jacobi sweep, computing B from A.
!
    call sweep_band ( nx, s, e, dx, dy, f, a, b )

    call get_norm ( nx, s, e, b, cartesian_comm, cartesian_master, bnorm )
!
!  Share with neighbor processes the value of B at "ghost" points.
!
    call exchange_band ( nx, s, e, cartesian_comm, bot, top, b )
!
!  Perform a Jacobi sweep, computing A from B.
!
    call sweep_band ( nx, s, e, dx, dy, f, b, a )

    call get_norm ( nx, s, e, a, cartesian_comm, cartesian_master, anorm )
!
!  Check for convergence.
!
    diff_part = 0.0E+00
    do j = s, e
      do i = 1, nx
        diff_part = max ( diff_part, abs ( a(i,j) - b(i,j) ) )
      end do
    end do

    call MPI_Allreduce ( diff_part, diff, 1, MPI_REAL, MPI_MAX, &
      cartesian_comm, error )

    if ( cartesian_id == cartesian_master ) then
      write ( *, '(i8,3g14.6)' ) it, bnorm, anorm, diff
    end if

    if ( diff <= tolerance ) then
      converged = .true.
      exit
    end if

  end do

  if ( world_id == world_master ) then

    if ( converged ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'POISSON_MPI - Master process:'
      write ( *, '(a)' ) '  The iteration has converged'
    else
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'POISSON_MPI - Master process:'
      write ( *, '(a)' ) '  The iteration has NOT converged'
    end if

  end if
!
!  Free memory.
!
  deallocate ( a )
  deallocate ( b )
  deallocate ( f )
!
!  Terminate MPI.
!
  call MPI_Finalize ( error )
!
!  Terminate.
!
  if ( world_id == world_master ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'POISSON_MPI:'
    write ( *, '(a)' ) '  Normal end of execution.'
    write ( *, '(a)' ) ' '
    call timestamp ( )
  end if

  stop
end
subroutine get_norm ( nx, s, e, a, cartesian_comm, cartesian_master, anorm )

!*****************************************************************************80
!
!! GET_NORM gets the max norm of a shared quantity, over the grid interior.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 September 2002
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer NX, the X grid dimension.
!
!    Input, integer S, E, indicates the first and last indices of Y data
!    to be controlled by this process.
!
!    Input, real A(0:NX+1,S-1:E+1), the quantity whose max norm is desired.
!
!    Input, integer CARTESIAN_COMM, the id of the Cartesian communicator.
!
!    Input, integer CARTESIAN_MASTER, the id of the master Cartesian process.
!
!    Output, real ANORM, the maximum absolute value of A, over the whole
!    interior grid.
!
  use mpi

  implicit none

  integer e
  integer nx
  integer s

  real a(0:nx+1,s-1:e+1)
  real anorm
  real anorm_part
  integer cartesian_comm
  integer cartesian_master
  integer error
  integer i
  integer j

  anorm_part = 0.0E+00
  do j = s, e
    do i = 1, nx
      anorm_part = max ( anorm_part, abs ( a(i,j) ) )
    end do
  end do

  call MPI_Reduce ( anorm_part, anorm, 1, MPI_REAL, MPI_MAX, cartesian_master, &
    cartesian_comm, error )

  return
end
subroutine decomp_band ( m, n, r, s, e )

!*****************************************************************************80
!
!! DECOMP_BAND determines the part of the work to be done by this process.
!
!  Discussion:
!
!    This routine is a replacement for the routine MPE_Decomp1d, and
!    is an informed guess as to how it works.
!
!    Given M things in a row, divide them up in consecutive chunks among
!    N processes so that the number assigned to each process is nearly
!    equal.
!
!    If M is exactly divisible by N, then each process gets M/N consecutive
!    tasks.
!
!    Otherwise, if there is a remainder of P, then the first P processes get
!    M/N+1 tasks, and the last (N-P) processes get M/N tasks.
!
!  Example:
!
!    M = 27, N = 10
!
!    R    S   E
!   --   --  --
!    0:   1   3
!    1:   4   6
!    2:   7   9
!    3:  10  12
!    4:  13  15
!    5:  16  18
!    6:  19  21
!    7:  22  23
!    8:  24  25
!    9:  26  27
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 September 2002
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer M, the number of tasks to be divided among processes.
!
!    Input, integer N, the number of processes.
!
!    Input, integer R, the rank of a particular process.  Because of
!    peculiarities of MPI, R will be between 0 and N-1.
!
!    Output, integer S, E, the first and last tasks to be given to process R.
!    S will be 1 or greater, E will be M or less.
!
  implicit none

  integer e
  integer m
  integer n
  integer part
  integer r
  integer s
  integer whole

  whole = m / n
  part = m - whole * n

  if ( r + 1 <= part ) then
    s =    r             * ( whole + 1 ) + 1
    e =  ( r + 1 )       * ( whole + 1 )
  else
    s =   r             * whole + 1 + part
    e = ( r + 1 )       * whole     + part
  end if

  return
end
subroutine init_band ( nx, ny, s, e, cartesian_id, cartesian_comm, num_procs, &
  dx, dy, f, a, b )

!*****************************************************************************80
!
!! INIT_BAND initializes the arrays.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 September 2002
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer NX, NY, the X and Y grid dimensions.
!
!    Input, integer S, E, indicates the first and last indices of Y data
!    to be controlled by this process.
!
!    Input, integer CARTESIAN_ID, the rank of this process.
!
!    Input, integer CARTESIAN_COMM, the id of the Cartesian communicator.
!
!    Input, integer NUM_PROCS, the number of processes.
!
!    Output, real DX, DY, the spacing between grid points.
!
!    Output, real F(0:NX+1,S-1:E+1), the right hand side data.
!
!    Output, real A(0:NX+1,S-1:E+1), B(0:NX+1,S-1:E+1), the initial
!    solution estimates.
!
  use mpi

  integer e
  integer nx
  integer s

  real a(0:nx+1,s-1:e+1)
  real b(0:nx+1,s-1:e+1)
  integer cartesian_id
  integer cartesian_comm
  integer, parameter :: cartesian_master = 0
  real dx
  real dy
  integer error
  real f(0:nx+1,s-1:e+1)
  real fnorm
  real fnorm_part
  integer i
  integer j
  integer num_procs
  integer ny
  real x
  real y

  dx = 1.0E+00 / real ( nx + 1 )
  dy = 1.0E+00 / real ( ny + 1 )

  if ( cartesian_id == cartesian_master ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INIT_BAND - Master Cartesian process:'
    write ( *, '(a,g14.6)' ) '  The X grid spacing is ', dx
    write ( *, '(a,g14.6)' ) '  The Y grid spacing is ', dy
  end if
!
!  Set the initial guesses to zero.
!
  a(0:nx+1,s-1:e+1) = 0.0E+00
  b(0:nx+1,s-1:e+1) = 0.0E+00
!
!  The "boundary" entries of F will store the boundary values of the solution.
!
  fnorm_part = 0.0E+00

  x = 0.0E+00
  do j = s, e
    y = real ( j ) / real ( ny + 1 )
    f(0,j) = sin ( x * y )
    a(0,j) = f(0,j)
    b(0,j) = f(0,j)
    fnorm_part = max ( fnorm_part, abs ( f(0,j) ) )
  end do

  x = 1.0E+00
  do j = s, e
    y = real ( j ) / real ( ny + 1 )
    f(nx+1,j) = sin ( x * y )
    a(nx+1,j) = f(nx+1,j)
    b(nx+1,j) = f(nx+1,j)
    fnorm_part = max ( fnorm_part, abs ( f(nx+1,j) ) )
  end do

  if ( cartesian_id == 0 ) then
    y = 0.0E+00
    do i = 0, nx+1
      x = real ( i ) / real ( nx + 1 )
      f(i,0) = sin ( x * y )
      a(i,0) = f(i,0)
      b(i,0) = f(i,0)
      fnorm_part = max ( fnorm_part, abs ( f(i,0) ) )
    end do
  end if

  if ( cartesian_id == num_procs-1 ) then
    y = 1.0E+00
    do i = 0, nx+1
      x = real ( i ) / real ( nx + 1 )
      f(i,ny+1) = sin ( x * y )
      a(i,ny+1) = f(i,ny+1)
      b(i,ny+1) = f(i,ny+1)
      fnorm_part = max ( fnorm_part, abs ( f(i,ny+1) ) )
    end do
  end if

  call MPI_Reduce ( fnorm_part, fnorm, 1, MPI_REAL, MPI_MAX, cartesian_master, &
    cartesian_comm, error )

  if ( cartesian_id == cartesian_master ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INIT_BAND - Master Cartesian process:'
    write ( *, '(a,g14.6)' ) '  Max norm of boundary values = ', fnorm
  end if
!
!  The "interior" entries of F will store the right hand sides of the
!  Poisson equation.
!
  do j = s, e
    y = real ( j ) / real ( ny + 1 )
    do i = 1, nx
      x = real ( i ) / real ( nx + 1 )
      f(i,j) = - ( x**2 + y**2 ) * sin ( x * y )
    end do
  end do

  return
end
subroutine exchange_band ( nx, s, e, cartesian_comm, bot, top, a )

!*****************************************************************************80
!
!! EXCHANGE_BAND swaps interface information between neighboring processes.
!
!  Discussion:
!
!    The standard MPI send and receive routines are used.  These are
!    blocking routines, which means that control will not return from
!    a call to MPI_Send until the data has been buffered or has reached
!    the receiving process.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    18 September 2002
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer NX, the number of (interior) grid lines in the X direction.
!
!    Input, integer S, E, indicates the first and last indices of Y data
!    to be controlled by this process.
!
!    Input, integer CARTESIAN_COMM, the identifier for the Cartesian
!    communicator.
!
!    Input, integer BOT, TOP, the identifiers of the processes "below" and
!    "above" this process, with which it must exchange information.
!
!    Input/output, real A(0:NX+1,S-1:E+1), an array of data.  Column
!    S must be sent to the "bottom" process and "E" to the top process.
!    Values for S-1 and E+1 will be received from the bottom and top
!    processes.
!
  use mpi

  integer e
  integer nx
  integer s

  real a(0:nx+1,s-1:e+1)
  integer bot
  integer cartesian_comm
  integer error
  integer status(MPI_STATUS_SIZE)
  integer tag
  integer top

  tag = 0
  call MPI_Send ( a(1,e),   nx, MPI_REAL, top, tag, cartesian_comm, error )

  call MPI_Recv ( a(1,s-1), nx, MPI_REAL, bot, tag, cartesian_comm, status, &
    error )

  tag = 1
  call MPI_Send ( a(1,s),   nx, MPI_REAL, bot, tag, cartesian_comm, error )
  call MPI_Recv ( a(1,e+1), nx, MPI_REAL, top, tag, cartesian_comm, status, &
    error )

  return
end
subroutine sweep_band ( nx, s, e, dx, dy, f, u, unew )

!*****************************************************************************80
!
!! SWEEP_BAND carries out one step of the Jacobi iteration.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    16 September 2002
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer NX, the X grid dimension.
!
!    Input, integer S, E, indicates the first and last indices of Y data
!    to be controlled by this process.
!
!    Input, real DX, DY, the spacing between grid points.
!
!    Input, real F(0:NX+1,S-1:E+1), the right hand side data.
!
!    Input, real U(0:NX+1,S-1:E+1), the previous solution estimate.
!
!    Output, real UNEW(0:NX+1,S-1:E+1), the updated solution estimate.
!
  implicit none

  integer e
  integer nx
  integer s

  real dx
  real dy
  real f(0:nx+1,s-1:e+1)
  integer i
  integer j
  real u(0:nx+1,s-1:e+1)
  real unew(0:nx+1,s-1:e+1)

  do j = s, e
    do i = 1, nx

      unew(i,j) = ( u(i-1,j) + u(i,j+1) + u(i,j-1) + u(i+1,j) ) / 4.0E+00 &
        - f(i,j) * dx * dy

    end do
  end do

  return
end
subroutine timestamp ( )

!*****************************************************************************80
!
!! TIMESTAMP prints the current YMDHMS date as a time stamp.
!
!  Example:
!
!    31 May 2001   9:45:54.872 AM
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    18 May 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    None
!
  implicit none

  character ( len = 8 ) ampm
  integer ( kind = 4 ) d
  integer ( kind = 4 ) h
  integer ( kind = 4 ) m
  integer ( kind = 4 ) mm
  character ( len = 9 ), parameter, dimension(12) :: month = (/ &
    'January  ', 'February ', 'March    ', 'April    ', &
    'May      ', 'June     ', 'July     ', 'August   ', &
    'September', 'October  ', 'November ', 'December ' /)
  integer ( kind = 4 ) n
  integer ( kind = 4 ) s
  integer ( kind = 4 ) values(8)
  integer ( kind = 4 ) y

  call date_and_time ( values = values )

  y = values(1)
  m = values(2)
  d = values(3)
  h = values(5)
  n = values(6)
  s = values(7)
  mm = values(8)

  if ( h < 12 ) then
    ampm = 'AM'
  else if ( h == 12 ) then
    if ( n == 0 .and. s == 0 ) then
      ampm = 'Noon'
    else
      ampm = 'PM'
    end if
  else
    h = h - 12
    if ( h < 12 ) then
      ampm = 'PM'
    else if ( h == 12 ) then
      if ( n == 0 .and. s == 0 ) then
        ampm = 'Midnight'
      else
        ampm = 'AM'
      end if
    end if
  end if

  write ( *, '(i2.2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
    d, trim ( month(m) ), y, h, ':', n, ':', s, '.', mm, trim ( ampm )

  return
end
