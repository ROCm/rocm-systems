!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!
! HIP runtime stream-capture introspection (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises the three capture-status queries -- hipStreamIsCapturing,
! hipStreamGetCaptureInfo and hipStreamGetCaptureInfo_v2 -- plus
! hipGraphNodeGetType, all of which return an enum through a by-reference dummy
! argument.
!
! graph.f03 already covers hipStreamIsCapturing around a capture; this test adds
! the two GetCaptureInfo flavours and the node-type query, and is written so a
! getter that never writes to its output cannot pass:
!
!   * every query is primed with the sentinel -1 before the call;
!   * the status is checked both OUTSIDE a capture (must read None) and INSIDE
!     one (must read Active), so a getter stuck on a constant fails one of them;
!   * _v2 must additionally hand back a non-null graph and a capture id matching
!     the one the v1 call reported;
!   * hipGraphNodeGetType is asked about two nodes of DIFFERENT kinds in the
!     same graph (the captured memset and an explicitly added empty node), so it
!     has to discriminate rather than return a fixed value.
!!!!!!!!!!!!!!
!
program stream_capture_info
  use iso_c_binding
  use hip

  implicit none

  integer(c_size_t), parameter :: nbytes = 256_c_size_t

  type(c_ptr) :: stream = c_null_ptr
  type(c_ptr) :: dptr = c_null_ptr
  type(c_ptr) :: graph = c_null_ptr
  type(c_ptr) :: graph_out = c_null_ptr
  type(c_ptr) :: deps_out = c_null_ptr
  type(c_ptr) :: emptynode = c_null_ptr
  type(c_ptr) :: nodes_out(8)
  integer(kind(hipStreamCaptureStatusNone)) :: capstat
  integer(kind(hipGraphNodeTypeKernel)) :: nodetype
  integer(kind(hipGraphNodeTypeKernel)) :: nodetypes(2)
  integer(c_long_long) :: capid, capid_v2
  integer(c_size_t) :: ndeps, numnodes

  write(*,"(a)",advance="no") &
    "-- Running test 'hip stream_capture_info' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMalloc(dptr, nbytes))

  ! 1. An idle stream is not capturing, as seen by both v1 queries.
  capstat = -1
  call hipCheck(hipStreamIsCapturing(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusNone, "hipStreamIsCapturing before capture")

  capstat = -1
  call hipCheck(hipStreamGetCaptureInfo(stream, capstat))   ! pId omitted: NULL
  call expect_status(capstat, hipStreamCaptureStatusNone, "hipStreamGetCaptureInfo before capture")

  ! 2. Inside a capture all three queries must report Active, and the two
  !    GetCaptureInfo flavours must agree on the capture id.
  call hipCheck(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal))
  call hipCheck(hipMemsetAsync(dptr, 5, nbytes, stream))

  capstat = -1
  call hipCheck(hipStreamIsCapturing(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusActive, "hipStreamIsCapturing during capture")

  capstat = -1
  capid = -1
  call hipCheck(hipStreamGetCaptureInfo(stream, capstat, capid))
  call expect_status(capstat, hipStreamCaptureStatusActive, "hipStreamGetCaptureInfo during capture")
  if (capid <= 0) then
     write(*,*) "FAILED! hipStreamGetCaptureInfo returned capture id ", capid, &
                " (expected a positive id while capturing)"
     call exit(1)
  end if

  capstat = -1
  capid_v2 = -1
  ndeps = 0
  call hipCheck(hipStreamGetCaptureInfo_v2(stream, capstat, capid_v2, &
                                           graph_out, deps_out, ndeps))
  call expect_status(capstat, hipStreamCaptureStatusActive, "hipStreamGetCaptureInfo_v2 during capture")
  if (capid_v2 /= capid) then
     write(*,*) "FAILED! hipStreamGetCaptureInfo_v2 capture id ", capid_v2, &
                " differs from hipStreamGetCaptureInfo id ", capid
     call exit(1)
  end if
  if (.not. c_associated(graph_out)) then
     write(*,*) "FAILED! hipStreamGetCaptureInfo_v2 returned a null graph while capturing"
     call exit(1)
  end if
  if (ndeps /= 1) then
     write(*,*) "FAILED! hipStreamGetCaptureInfo_v2 reported ", ndeps, &
                " dependencies (expected 1 after a single memset)"
     call exit(1)
  end if

  call hipCheck(hipStreamEndCapture(stream, graph))
  if (.not. c_associated(graph)) then
     write(*,*) "FAILED! hipStreamEndCapture produced a null graph"
     call exit(1)
  end if

  ! 3. The stream is idle again.
  capstat = -1
  call hipCheck(hipStreamIsCapturing(stream, capstat))
  call expect_status(capstat, hipStreamCaptureStatusNone, "hipStreamIsCapturing after capture")

  ! 4. The captured graph holds exactly the memset node; add an empty node so
  !    hipGraphNodeGetType has two different kinds to tell apart.
  call hipCheck(hipGraphAddEmptyNode(emptynode, graph, c_null_ptr, 0_c_size_t))

  numnodes = size(nodes_out, kind=c_size_t)
  call hipCheck(hipGraphGetNodes(graph, nodes_out(1), numnodes))
  if (numnodes /= 2) then
     write(*,*) "FAILED! captured graph holds ", numnodes, " nodes (expected 2)"
     call exit(1)
  end if

  ! hipGraphGetNodes returns the nodes in no specified order, so accept the
  ! set {Memset, Empty} either way round.
  nodetypes = -1
  call hipCheck(hipGraphNodeGetType(nodes_out(1), nodetypes(1)))
  call hipCheck(hipGraphNodeGetType(nodes_out(2), nodetypes(2)))
  if (.not. ((nodetypes(1) == hipGraphNodeTypeMemset .and. nodetypes(2) == hipGraphNodeTypeEmpty) .or. &
             (nodetypes(1) == hipGraphNodeTypeEmpty .and. nodetypes(2) == hipGraphNodeTypeMemset))) then
     write(*,*) "FAILED! hipGraphNodeGetType on the graph nodes returned ", nodetypes, &
                " (expected hipGraphNodeTypeMemset = ", hipGraphNodeTypeMemset, &
                " and hipGraphNodeTypeEmpty = ", hipGraphNodeTypeEmpty, " in either order)"
     call exit(1)
  end if

  nodetype = -1
  call hipCheck(hipGraphNodeGetType(emptynode, nodetype))
  if (nodetype /= hipGraphNodeTypeEmpty) then
     write(*,*) "FAILED! hipGraphNodeGetType on the empty node returned ", nodetype, &
                " (expected hipGraphNodeTypeEmpty = ", hipGraphNodeTypeEmpty, ")"
     call exit(1)
  end if

  call hipCheck(hipGraphDestroy(graph))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED! capture id = ", capid

contains

  subroutine expect_status(got, want, what)
    integer(kind(hipStreamCaptureStatusNone)), intent(in) :: got, want
    character(len=*), intent(in) :: what
    if (got /= want) then
       write(*,*) "FAILED! ", what, " = ", got, " (expected ", want, ")"
       call exit(1)
    end if
  end subroutine expect_status

end program stream_capture_info
