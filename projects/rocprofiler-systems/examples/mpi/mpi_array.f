C *****************************************************************************
C FILE: mpi_array.f
C DESCRIPTION:
C   MPI Example - Array Assignment - Fortran Version
C   This program demonstrates a simple data decomposition. The master task
C   first initializes an array and then distributes an equal portion that
C   array to the other tasks. After the other tasks receive their portion
C   of the array, they perform an addition operation to each array element.
C   They also maintain a sum for their portion of the array. The master task
C   does likewise with its portion of the array and any leftover elements.
C   As each of the non-master tasks finish, they send their updated portion
C   of the array to the master. An MPI collective communication call is used
C   to collect the sums maintained by each task.  Finally, the master
C   task displays selected parts of the final array and the global sum of
C   all array elements.
C AUTHOR: Blaise Barney
C LAST REVISED: 07/03/19
C **************************************************************************

      program array
      include 'mpif.h'

      integer   ARRAYSIZE, MASTER
      parameter (ARRAYSIZE = 20000000)
      parameter (MASTER = 0)

      integer  numtasks, taskid, ierr, dest, offset, i, tag1,
     &         tag2, source, chunksize, leftover
      real*8   mysum, sum, data(ARRAYSIZE)
      integer  status(MPI_STATUS_SIZE)
      common   /a/ data

C ***** Initializations *****
      call MPI_INIT(ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, numtasks, ierr)

c     i = MOD(numtasks, 4)
c     if (i .ne. 0) then
c       call MPI_Abort(MPI_COMM_WORLD,ierr)
c       stop
c     end if

      call MPI_COMM_RANK(MPI_COMM_WORLD, taskid, ierr)
      write(*,10) taskid
  10  format('MPI task ',I4,' has started...  '$)
      chunksize = (ARRAYSIZE / numtasks)
      leftover = MOD(ARRAYSIZE, numtasks)
      tag2 = 1
      tag1 = 2

C***** Master task only ******
      if (taskid .eq. MASTER) then

C       Initialize the array
        sum = 0.0
        do i=1, ARRAYSIZE
          data(i) = i * 1.0
          sum = sum + data(i)
        end do
        write(*,20) sum
        write(*,21) numtasks, chunksize, leftover
  20    format('Initialized array sum = ',E14.6)
  21    format('numtasks= ',I4,'  chunksize= ',I8,'  leftover= ',I4)

C       Send each task its portion of the array - master keeps 1st part
C       plus leftover elements
        offset = chunksize + leftover + 1
        do dest=1, numtasks-1
          call MPI_SEND(offset, 1, MPI_INTEGER, dest, tag1,
     &      MPI_COMM_WORLD, ierr)
          call MPI_SEND(data(offset), chunksize, MPI_REAL8, dest,
     &      tag2, MPI_COMM_WORLD, ierr)
          write(*,22) chunksize, dest, offset
  22      format('Sent ',I8,' elements to task ',I4,' offset= ',I10)
          offset = offset + chunksize
        end do

C       Master does its part of the work
        offset = 1
        call update(offset, chunksize + leftover, taskid, mysum)

C       Wait to receive results from each task
        do i=1, numtasks-1
          source = i
          call MPI_RECV(offset, 1, MPI_INTEGER, source, tag1,
     &      MPI_COMM_WORLD, status, ierr)
          call MPI_RECV(data(offset), chunksize, MPI_REAL8,
     &      source, tag2, MPI_COMM_WORLD, status, ierr)
        end do

C       Get final sum and print sample results
        call MPI_Reduce(mysum, sum, 1, MPI_REAL8, MPI_SUM, MASTER,
     &    MPI_COMM_WORLD, ierr)
        print *, 'Sample results:'
        offset = 1
        do i=1, numtasks
          write (*,30) data(offset:offset+4)
  30      format(5E14.6)
          offset = offset + chunksize
        end do
        write(*,40) sum
  40    format('*** Final sum= ',E14.6,' ***')

      end if


C***** Non-master tasks only *****

      if (taskid .gt. MASTER) then

C       Receive my portion of array from the master task */
        call MPI_RECV(offset, 1, MPI_INTEGER, MASTER, tag1,
     &    MPI_COMM_WORLD, status, ierr)
        call MPI_RECV(data(offset), chunksize, MPI_REAL8, MASTER,
     &    tag2, MPI_COMM_WORLD, status, ierr)

C       Do my part of the work
        call update(offset, chunksize, taskid, mysum)

C       Send my results back to the master
        call MPI_SEND(offset, 1, MPI_INTEGER, MASTER, tag1,
     &    MPI_COMM_WORLD, ierr)
        call MPI_SEND(data(offset), chunksize, MPI_REAL8, MASTER,
     &    tag2, MPI_COMM_WORLD, ierr)

C       Use sum reduction operation to obtain final sum
        call MPI_Reduce(mysum, sum, 1, MPI_REAL8, MPI_SUM, MASTER,
     &    MPI_COMM_WORLD, ierr)

      endif


      call MPI_FINALIZE(ierr)


      end


      subroutine update(myoffset, chunksize, myid, mysum)
        integer   ARRAYSIZE, myoffset, chunksize, myid, i
        parameter (ARRAYSIZE = 20000000)
        real*8 mysum, data(ARRAYSIZE)
        common /a/ data
C       Perform addition to each of my array elements and keep my sum
        mysum = 0
        do i=myoffset, myoffset + chunksize-1
          data(i) = data(i) + (i * 1.0)
          mysum = mysum + data(i)
        end do
        write(*,50) myid,mysum
  50    format('Task',I4,' mysum = ',E14.6)
      end subroutine update
