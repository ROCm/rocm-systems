C******************************************************************************
C FILE: mpi_mm.f
C DESCRIPTION:
C   MPI Matrix Multiply - Fortran Version
C   In this code, the master task distributes a matrix multiply
C   operation to numtasks-1 worker tasks.
C   NOTE1:  C and Fortran versions of this code differ because of the way
C   arrays are stored/passed.  C arrays are row-major order but Fortran
C   arrays are column-major order.
C AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
C   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
C LAST REVISED: 04/02/05
C******************************************************************************

      program mm
      include 'mpif.h'

      parameter (NRA = 62)
      parameter (NCA = 15)
      parameter (NCB = 7)
      parameter (MASTER = 0)
      parameter (FROM_MASTER = 1)
      parameter (FROM_WORKER = 2)

      integer 	numtasks,taskid,numworkers,source,dest,mtype,
     &          cols,avecol,extra, offset,i,j,k,ierr
      integer status(MPI_STATUS_SIZE)
      real*8	a(NRA,NCA), b(NCA,NCB), c(NRA,NCB)

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, taskid, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numtasks, ierr )
      numworkers = numtasks-1
      print *, 'task ID= ',taskid

C *************************** master task *************************************
      if (taskid .eq. MASTER) then

C     Initialize A and B
        do 30 i=1, NRA
          do 30 j=1, NCA
          a(i,j) = (i-1)+(j-1)
 30     continue
        do 40 i=1, NCA
          do 40 j=1, NCB
	    b(i,j) = (i-1)*(j-1)
 40     continue

C     Send matrix data to the worker tasks
        avecol = NCB/numworkers
        extra = mod(NCB, numworkers)
        offset = 1
        mtype = FROM_MASTER
        do 50 dest=1, numworkers
          if (dest .le. extra) then
            cols = avecol + 1
          else
            cols = avecol
          endif
          write(*,*)'   sending',cols,' cols to task',dest
          call MPI_SEND( offset, 1, MPI_INTEGER, dest, mtype,
     &                   MPI_COMM_WORLD, ierr )
          call MPI_SEND( cols, 1, MPI_INTEGER, dest, mtype,
     &                   MPI_COMM_WORLD, ierr )
          call MPI_SEND( a, NRA*NCA, MPI_DOUBLE_PRECISION, dest, mtype,
     &                   MPI_COMM_WORLD, ierr )
          call MPI_SEND( b(:,offset), cols*NCA, MPI_DOUBLE_PRECISION,
     &                   dest, mtype, MPI_COMM_WORLD, ierr )
          offset = offset + cols
 50     continue

C     Receive results from worker tasks
        mtype = FROM_WORKER
        do 60 i=1, numworkers
          source = i
          call MPI_RECV( offset, 1, MPI_INTEGER, source,
     &                   mtype, MPI_COMM_WORLD, status, ierr )
          call MPI_RECV( cols, 1, MPI_INTEGER, source,
     &                   mtype, MPI_COMM_WORLD, status, ierr )
          call MPI_RECV( c(:,offset), cols*NRA, MPI_DOUBLE_PRECISION,
     &                   source, mtype, MPI_COMM_WORLD, status, ierr )
 60     continue

C     Print results
        do 90 i=1, NRA
          do 80 j = 1, NCB
            write(*,70)c(i,j)
  70        format(2x,f8.2,$)
  80      continue
          print *, ' '
  90    continue
      endif

C *************************** worker task *************************************
      if (taskid > MASTER) then
C     Receive matrix data from master task
        mtype = FROM_MASTER
        call MPI_RECV( offset, 1, MPI_INTEGER, MASTER,
     &                 mtype, MPI_COMM_WORLD, status, ierr )
        call MPI_RECV( cols, 1, MPI_INTEGER, MASTER,
     &                 mtype, MPI_COMM_WORLD, status, ierr )
        call MPI_RECV( a, NRA*NCA, MPI_DOUBLE_PRECISION, MASTER,
     &                 mtype, MPI_COMM_WORLD, status, ierr )
        call MPI_RECV( b, cols*NCA, MPI_DOUBLE_PRECISION, MASTER,
     &                 mtype, MPI_COMM_WORLD, status, ierr )

C     Do matrix multiply
        do 100 k=1, cols
          do 100 i=1, NRA
            c(i,k) = 0.0
            do 100 j=1, NCA
              c(i,k) = c(i,k) + a(i,j) * b(j,k)
  100   continue

C     Send results back to master task
        mtype = FROM_WORKER
        call MPI_SEND( offset, 1, MPI_INTEGER, MASTER, mtype,
     &                 MPI_COMM_WORLD, ierr )
        call MPI_SEND( cols, 1, MPI_INTEGER, MASTER, mtype,
     &                 MPI_COMM_WORLD, ierr )
        call MPI_SEND( c, cols*NRA, MPI_DOUBLE_PRECISION, MASTER,
     &                  mtype, MPI_COMM_WORLD, ierr )
      endif
      call MPI_FINALIZE(ierr)
      end
