#!/usr/bin/gawk -f
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.# Usage:

BEGIN {
  max_rank=0
  rings[""]=0
  max_ring=0
  treedns[""]=0
  max_treedn=0
  conn[""]=0
  has_collnet=0
  max_collnet=0
  max_collnet_rank=0
  max_collnet_channel=0
  collnet[""]=0
  collnet_conn[""]=0
  collnet_conn_type[""]=0
  col_start=2
  col_p1=col_start+1
  col_p2=col_start+2
  col_p3=col_start+3
  col_p4=col_start+4
  col_p5=col_start+5
  col_p6=col_start+6
  col_p7=col_start+7
  col_p8=col_start+8
}

{
  if($3=="NCCL" && $4=="INFO" && col_start==2) {
    col_start=5
    col_p1=col_start+1
    col_p2=col_start+2
    col_p3=col_start+3
    col_p4=col_start+4
    col_p5=col_start+5
    col_p6=col_start+6
    col_p7=col_start+7
    col_p8=col_start+8
  }

  if($5=="NCCL" && $6=="INFO" && col_start==2) {
    col_start=7
    col_p1=col_start+1
    col_p2=col_start+2
    col_p3=col_start+3
    col_p4=col_start+4
    col_p5=col_start+5
    col_p6=col_start+6
    col_p7=col_start+7
    col_p8=col_start+8
  }

  if($col_start=="Ring" && $col_p4=="->" && $col_p6=="->") {
    chan=strtonum($col_p1)
    rank=strtonum($col_p5)
    next_rank=strtonum($col_p7)
    rings[rank "," next_rank "," chan]="1"
    if(chan>max_ring)
      max_ring=chan
    if(rank>max_rank)
      max_rank=rank
  }

  if($col_start=="Trees") {
    col_1=col_start+1
    col_2=col_start+2
    do {
      match($col_1, /\[([0-9]+)\]/, ary)
      chan=strtonum(ary[1])
      where = match($col_2, /(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\|(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)/, ary)
      if(where != 0) {
        if(ary[8]!="-1")
          treedns[ary[7] "," ary[8] "," chan]="1"
        if(ary[9]!="-1")
          treedns[ary[7] "," ary[9] "," chan]="1"
        if(ary[10]!="-1")
          treedns[ary[7] "," ary[10] "," chan]="1"
      } else {
        where = match($col_2, /(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)/, ary)
        if(where != 0) {
          if(ary[1]!="-1")
            treedns[ary[4] "," ary[1] "," chan]="1"
          if(ary[2]!="-1")
            treedns[ary[4] "," ary[2] "," chan]="1"
          if(ary[3]!="-1")
            treedns[ary[4] "," ary[3] "," chan]="1"
        }
      }
      if(chan>max_treedn)
        max_treedn=chan
      col_1=col_1+2
      col_2=col_2+2
    } while ($col_1!="")
  }

  if($col_start=="CollNet" && $col_p1=="channel" && $col_p5=="down") {
    channel=strtonum($col_p2)
    up_rank=strtonum($col_p4)
    if(up_rank>max_collnet_rank)
      max_collnet_rank=up_rank
    for(s=col_p6;s<=NF;s++) {
      if($s=="nDown") break;
      rank=$s
      collnet[up_rank "," rank]="1"
      if(rank>max_collnet_rank)
        max_collnet_rank=rank
    }
    if(has_collnet==0)
      has_collnet=1
    if(channel>max_collnet_channel)
      max_collnet_channel=channel
  }

  if($col_start=="Coll" && $col_p2==":") {
    chan=strtonum($col_p1)
    rank=strtonum($col_p3)
    if($col_p4=="[receive]")
      collnet_conn[rank "," chan]=0
    else if($col_p4=="[send]")
      collnet_conn[rank "," chan]=1
    else
      printf "Error!\n"
    collnet_conn_type[rank "," chan]=$col_p6
    if(chan>max_collnet)
      max_collnet=chan
  }

  if($col_p6=="via") {
    match($col_p1, /([0-9]+)/, ary)
    chan=strtonum(ary[1])
    match($col_p3, /([0-9]+)\[.*\]/, ary)
    s=ary[1]
    match($col_p5, /([0-9]+)\[.*\]/, ary)
    d=ary[1]
    if(!((s "," d "," chan) in conn) || match($col_p7,"NET"))
      conn[s "," d "," chan]=$col_p7
  }

  if($col_p6=="[receive]" && $col_p7=="via") {
    match($col_p1, /([0-9]+)/, ary)
    chan=strtonum(ary[1])
    match($col_p3, /([0-9]+)\[.*\]/, ary)
    s=ary[1]
    match($col_p5, /([0-9]+)\[.*\]/, ary)
    d=ary[1]
    if(!((s "," d "," chan) in conn) || match($col_p8,"NET"))
      conn[s "," d "," chan]=$col_p8
  }
}

END {
  printf "digraph RCCL {\n"
  for(r=0;r<max_treedn+1;r++) {
    printf "  subgraph tree_%d {\n", r
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in treedns) {
          val=conn[s "," d "," r]
          style="solid"
          color="red"
          if (match(val,"NET")) {
            style="dashed"
            if (match(val,"GDRDMA"))
              color="green"
          }
          if (match(val,"P2P")) {
            color="green"
          }
          printf "    t%d_%d -> t%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, s, r, d, val, color, style
        }
      }
    }
    printf "\n"
    for(s=0;s<=max_rank;s++) {
      printf "    t%d_%d [label=\"%d\",fontsize=\"28\"];\n", r, s, s
    }
    printf "  }\n\n"
  }

  for(r=0;r<max_ring+1;r++) {
    remove_ring=0
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in rings && !((s "," d "," r) in conn)) {
          remove_ring=1
          break;
        }
      }
      if(d<=max_rank)
        break;
    }
    if (remove_ring!=0)
      continue
    printf "  subgraph ring_%d {\n", r
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in rings) {
          val=conn[s "," d "," r]
          style="solid"
          color="red"
          if (match(val,"NET")) {
            style="dashed"
            if (match(val,"GDRDMA"))
              color="green"
          }
          if (match(val,"P2P")) {
            color="green"
          }
          printf "    r%d_%d -> r%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, s, r, d, val, color, style
        }
      }
    }
    printf "\n"
    for(s=0;s<=max_rank;s++) {
      printf "    r%d_%d [label=\"%d\",fontsize=\"28\"];\n", r, s, s
    }
    printf "  }\n\n"
  }

  for(r=0; has_collnet && r<=max_collnet; r++) {
    printf "  subgraph collnet_%d {\n", r
    num_top_ranks=0
    rank_switch=max_collnet_rank+1
    for(s=0;s<=max_collnet_rank;s++) {
      if((s "," r) in collnet_conn_type)
        top_ranks[num_top_ranks++]=s
    }
    for(d=0; d<num_top_ranks; d++) {
      rank=top_ranks[d]
      send=collnet_conn[rank "," r]
      val=collnet_conn_type[rank "," r]
      style="solid"
      color="red"
      if (match(val,"COLLNET")) {
        style="dashed"
        if (match(val,"GDRDMA"))
          color="green"
      }
      printf "    c%d_%d -> c%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, rank_switch, r, rank, val, color, style
      for(s=0;s<=max_collnet_rank;s++) {
        if((rank "," s) in collnet) {
          style="solid"
          color="green"
          printf "    c%d_%d -> c%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, rank, r, s, "", color, style
        }
      }
    }
    printf "\n"
    for(s=0;s<=max_collnet_rank;s++) {
      printf "    c%d_%d [label=\"%d\",fontsize=\"28\"];\n", r, s, s
    }
    printf "    c%d_%d [label=\"SHARP:%d\",fontsize=\"28\"];\n", r, rank_switch, r
    printf "  }\n\n"
  }
  printf "}\n"
}
