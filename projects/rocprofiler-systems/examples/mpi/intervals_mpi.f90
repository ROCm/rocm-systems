program main

!*****************************************************************************80
!
!! MAIN is the main program for INTERVALS.
!
!  Discussion:
!
!    INTERVALS uses MPI routines to multiprocess a computational task.
!
!    We have a function F(X), an interval [XMIN,XMAX], 
!    and a value N.
!
!    We define N equally spaced points in the interval,
!
!      X(I) = ( ( N - I     ) * XMIN 
!             + (     I - 1 ) * XMAX ) 
!             / ( N     - 1 )
!
!    We thus have N-1 subintervals.
!
!    We assume we have N processors available.
!
!    Processor 0 is designated the master processor, assigned
!    to estimating the integral of F(X) over the entire
!    interval [ X(1), X(N) ].
!
!    For I = 1 to N-1, processor I is assigned the subinterval
!
!      [ X(I), X(I+1) ]
!
!    and then estimates the integral Q(I) of F(X) over that
!    subinterval.
!
!    COMMUNICATION:
!
!    Processor 0 communicates to processor I the endpoints of 
!    the interval it is assigned, and the number of sample points
!    to use in that interval.
!
!    Processor I communicates to processor 0 the computed value of
!    Q(I).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    17 March 2006
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
!    Snir, Otto, Huss-Lederman, Walker, Dongarra,
!    MPI - The Complete Reference,
!    Volume 1, The MPI Core,
!    second edition,
!    MIT Press, 1998.
!
  use mpi
 
  real ( kind = 8 ) f
  real ( kind = 8 ) h
  integer ( kind = 4 ) i
  integer ( kind = 4 ) id
  integer ( kind = 4 ) ierr
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  real ( kind = 8 ), parameter :: pi = 3.141592653589793238462643D+00
  integer ( kind = 4 ) process
  integer ( kind = 4 ) process_num
  real ( kind = 8 ) q_global
  real ( kind = 8 ) q_local
  integer ( kind = 4 ) received
  integer ( kind = 4 ) source
  integer ( kind = 4 ) status(MPI_Status_size)
  integer ( kind = 4 ) tag
  integer ( kind = 4 ) target
  real ( kind = 8 ) wtime
  real ( kind = 8 ) x
  real ( kind = 8 ) xb(2)
  real ( kind = 8 ) :: x_max = 1.0D+00
  real ( kind = 8 ) :: x_min = 0.0D+00
!
!  Establish the MPI environment.
!
  call MPI_Init ( ierr )
!
!  Get this process's ID.
!
  call MPI_Comm_rank ( MPI_COMM_WORLD, id, ierr )
!
!  Find out how many processes are available.
!
  call MPI_Comm_size ( MPI_COMM_WORLD, process_num, ierr )
!
!  Say hello (once), and shut down right away unless we
!  have at least 2 processes available.
!
  if ( id == 0 ) then
    call timestamp ( )
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INTERVALS - Master process:'
    write ( *, '(a)' ) '  FORTRAN90 version'
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) '  An MPI example program.'
    write ( *, '(a)' ) '  A quadrature over an interval is done by'
    write ( *, '(a)' ) '  assigning subintervals to processes.'
    write ( *, '(a,i8)' ) '  The number of processes is ', process_num

    wtime = MPI_Wtime ( )

    if ( process_num <= 1 ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'INTERVALS - Master process:'
      write ( *, '(a)' ) '  Need at least 2 processes!'
      call MPI_Finalize ( ierr )
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'INTERVALS - Master process:'
      write ( *, '(a)' ) '  Abnormal end of execution.'
      stop
    end if

  end if

  write ( *, '(a)' ) ' '
  write ( *, '(a,i8,a)' ) 'Process ', id, ': Active!'
!
!  Every process could figure out the endpoints of its interval
!  on its own.  But we want to demonstrate communication.  So we
!  assume that the assignment of processes to intervals is done
!  only by the master process, which then tells each process
!  what job it is to do.
!
  if ( id == 0 ) then

    do process = 1, process_num-1

      xb(1) = ( real ( process_num - process,     kind = 8 ) * x_min   &
              + real (               process - 1, kind = 8 ) * x_max ) &
              / real ( process_num           - 1, kind = 8 )

      xb(2) = ( real ( process_num - process - 1, kind = 8 ) * x_min   &
              + real (               process,     kind = 8 ) * x_max ) &
              / real ( process_num           - 1, kind = 8 )
 
      target = process
      tag = 1

      call MPI_Send ( xb, 2, MPI_DOUBLE_PRECISION, target, tag, &
        MPI_COMM_WORLD, ierr )

    end do

  else

    tag = 1

    call MPI_Recv ( xb, 2, MPI_DOUBLE_PRECISION, 0, tag, &
      MPI_COMM_WORLD, status, ierr )
    
  end if
!
!  Wait here until everyone has gotten their assignment.
!
  call MPI_Barrier ( MPI_COMM_WORLD, ierr )

  if ( id == 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INTERVALS - Master process:'
    write ( *, '(a)' ) '  Subintervals have been assigned.'
  end if
!
!  Every process needs to be told the number of points to use.
!  Since this is the same value for everybody, we use a broadcast.
!  Again, we are doing it in this roundabout way to emphasize that
!  the choice for M could really be made at runtime, by processor 0,
!  and then sent out to the others.
!
  m = 100

  call MPI_Bcast ( m, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr )
!
!  Now, every process EXCEPT 0 computes its estimate of the 
!  integral over its subinterval, and sends the result back
!  to process 0.
!
  if ( id /= 0 ) then

    q_local = 0.0D+00

    do i = 1, m

      x = ( real ( 2 * m - 2 * i + 1, kind = 8 ) * xb(1)   &
          + real (         2 * i - 1, kind = 8 ) * xb(2) ) &
          / real ( 2 * m,             kind = 8 )

      q_local = q_local + f ( x )

    end do

    q_local = q_local * ( xb(2) - xb(1) ) / real ( m, kind = 8 )

    tag = 2

    call MPI_Send ( q_local, 1, MPI_DOUBLE_PRECISION, 0, tag, &
      MPI_COMM_WORLD, ierr )
!
!  Process 0 expects to receive N-1 partial results.
!
  else

    received = 0
    q_global = 0.0D+00

    do while ( received < process_num - 1 )

      source = MPI_ANY_SOURCE
      tag = 2

      call MPI_Recv ( q_local, 1, MPI_DOUBLE_PRECISION, source, tag, &
        MPI_COMM_WORLD, status, ierr )

      q_global = q_global + q_local
      received = received + 1

    end do

  end if
!
!  The master process prints the answer.
!
  if ( id == 0 ) then

    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INTERVALS - Master process:'
    write ( *, '(a,g14.6)' ) '  Estimate for PI is ', q_global
    write ( *, '(a,g14.6)' ) '  Error is           ', q_global - pi

    wtime = MPI_Wtime ( ) - wtime

    write ( *, '(a)' ) ' '
    write ( *, '(a,f14.6)' ) '  Elapsed wall clock seconds = ', wtime

  end if
!
!  Terminate MPI.
!
  call MPI_Finalize ( ierr )
!
!  Terminate.
!
  if ( id == 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'INTERVALS:'
    write ( *, '(a)' ) '  Normal end of execution.'
    write ( *, '(a)' ) ' '
    call timestamp ( )
  end if

  stop
end
function f ( x )

!*****************************************************************************80
!
!! F is the function we are integrating.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 February 2000
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the function.
!
!    Output, real ( kind = 8 ) F, the value of the function.
!
  implicit none

  real ( kind = 8 ) f
  real ( kind = 8 ) x

  f = 4.0D+00 / ( 1.0D+00 + x * x )

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
