#!/usr/bin/gawk -f
# Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
# Parallel version: Generates separate DOT files per channel for parallel rendering

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

  # Output directory from environment or default
  outdir = ENVIRON["TOPO_OUTDIR"]
  if (outdir == "") outdir = "/tmp/topo_split"
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
    if(chan>max_ring) max_ring=chan
    if(rank>max_rank) max_rank=rank
  }

  if($col_start=="Trees") {
    col_1=col_start+1
    col_2=col_start+2
    do {
      match($col_1, /\[([0-9]+)\]/, ary)
      chan=strtonum(ary[1])
      where = match($col_2, /(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\|(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)/, ary)
      if(where != 0) {
        if(ary[8]!="-1") treedns[ary[7] "," ary[8] "," chan]="1"
        if(ary[9]!="-1") treedns[ary[7] "," ary[9] "," chan]="1"
        if(ary[10]!="-1") treedns[ary[7] "," ary[10] "," chan]="1"
      } else {
        where = match($col_2, /(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)/, ary)
        if(where != 0) {
          if(ary[1]!="-1") treedns[ary[4] "," ary[1] "," chan]="1"
          if(ary[2]!="-1") treedns[ary[4] "," ary[2] "," chan]="1"
          if(ary[3]!="-1") treedns[ary[4] "," ary[3] "," chan]="1"
        }
      }
      if(chan>max_treedn) max_treedn=chan
      col_1=col_1+2
      col_2=col_2+2
    } while ($col_1!="")
  }

  if($col_start=="CollNet" && $col_p1=="channel" && $col_p5=="down") {
    channel=strtonum($col_p2)
    up_rank=strtonum($col_p4)
    if(up_rank>max_collnet_rank) max_collnet_rank=up_rank
    for(s=col_p6;s<=NF;s++) {
      if($s=="nDown") break;
      rank=$s
      collnet[up_rank "," rank]="1"
      if(rank>max_collnet_rank) max_collnet_rank=rank
    }
    if(has_collnet==0) has_collnet=1
    if(channel>max_collnet_channel) max_collnet_channel=channel
  }

  if($col_start=="Coll" && $col_p2==":") {
    chan=strtonum($col_p1)
    rank=strtonum($col_p3)
    if($col_p4=="[receive]") collnet_conn[rank "," chan]=0
    else if($col_p4=="[send]") collnet_conn[rank "," chan]=1
    collnet_conn_type[rank "," chan]=$col_p6
    if(chan>max_collnet) max_collnet=chan
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

function get_fontsize() {
  if (max_rank >= 64) return 18
  else if (max_rank >= 32) return 22
  else return 28
}

function write_header(file) {
  printf "digraph RCCL {\n" > file
  printf "  graph [fontname=\"Helvetica\", fontsize=12, compound=true];\n" > file
  printf "  node [fontname=\"Helvetica\", shape=circle, fixedsize=true, width=0.6, height=0.6];\n" > file
  printf "  edge [fontname=\"Helvetica\", fontsize=10];\n" > file
  if (max_rank >= 64) {
    printf "  graph [overlap=false, splines=true, sep=\"+10\", esep=\"+5\"];\n" > file
    printf "  node [width=0.5, height=0.5, fontsize=20];\n" > file
  } else if (max_rank >= 32) {
    printf "  node [width=0.55, height=0.55, fontsize=24];\n" > file
  }
}

END {
  fs = get_fontsize()

  # Write manifest file listing all channels
  manifest = outdir "/manifest.txt"
  printf "" > manifest

  # Generate separate DOT file for each tree channel
  for(r=0; r<max_treedn+1; r++) {
    dotfile = outdir "/tree_" r ".dot"
    printf "tree_%d\n", r >> manifest

    write_header(dotfile)
    printf "  subgraph cluster_tree_%d {\n", r > dotfile
    printf "    label=\"Tree Channel %d\";\n", r > dotfile
    printf "    style=dashed; color=gray;\n" > dotfile

    for(s=0; s<=max_rank; s++) {
      for(d=0; d<=max_rank; d++) {
        if ((s "," d "," r) in treedns) {
          val=conn[s "," d "," r]
          style="solid"; color="red"; penwidth="1.5"
          if (match(val,"NET")) { style="dashed"; if (match(val,"GDRDMA")) color="green" }
          if (match(val,"P2P")) color="green"
          printf "    t%d_%d -> t%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",penwidth=\"%s\"];\n", r, s, r, d, val, color, style, penwidth > dotfile
        }
      }
    }
    for(s=0; s<=max_rank; s++) {
      printf "    t%d_%d [label=\"%d\",fontsize=\"%d\"];\n", r, s, s, fs > dotfile
    }
    printf "  }\n}\n" > dotfile
    close(dotfile)
  }

  # Generate separate DOT file for each ring channel
  for(r=0; r<max_ring+1; r++) {
    remove_ring=0
    for(s=0; s<=max_rank; s++) {
      for(d=0; d<=max_rank; d++) {
        if ((s "," d "," r) in rings && !((s "," d "," r) in conn)) {
          remove_ring=1; break
        }
      }
      if(d<=max_rank) break
    }
    if (remove_ring!=0) continue

    dotfile = outdir "/ring_" r ".dot"
    printf "ring_%d\n", r >> manifest

    write_header(dotfile)
    printf "  subgraph cluster_ring_%d {\n", r > dotfile
    printf "    label=\"Ring Channel %d\";\n", r > dotfile
    printf "    style=dashed; color=gray;\n" > dotfile

    for(s=0; s<=max_rank; s++) {
      for(d=0; d<=max_rank; d++) {
        if ((s "," d "," r) in rings) {
          val=conn[s "," d "," r]
          style="solid"; color="red"; penwidth="1.5"
          if (match(val,"NET")) { style="dashed"; if (match(val,"GDRDMA")) color="green" }
          if (match(val,"P2P")) color="green"
          printf "    r%d_%d -> r%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",penwidth=\"%s\"];\n", r, s, r, d, val, color, style, penwidth > dotfile
        }
      }
    }
    for(s=0; s<=max_rank; s++) {
      printf "    r%d_%d [label=\"%d\",fontsize=\"%d\"];\n", r, s, s, fs > dotfile
    }
    printf "  }\n}\n" > dotfile
    close(dotfile)
  }

  # Generate separate DOT file for each collnet channel
  for(r=0; has_collnet && r<=max_collnet; r++) {
    dotfile = outdir "/collnet_" r ".dot"
    printf "collnet_%d\n", r >> manifest

    write_header(dotfile)
    printf "  subgraph cluster_collnet_%d {\n", r > dotfile
    printf "    label=\"CollNet Channel %d\";\n", r > dotfile
    printf "    style=dashed; color=gray;\n" > dotfile

    num_top_ranks=0
    rank_switch=max_collnet_rank+1
    for(s=0; s<=max_collnet_rank; s++) {
      if((s "," r) in collnet_conn_type) top_ranks[num_top_ranks++]=s
    }
    for(d=0; d<num_top_ranks; d++) {
      rank=top_ranks[d]
      val=collnet_conn_type[rank "," r]
      style="solid"; color="red"; penwidth="1.5"
      if (match(val,"COLLNET")) { style="dashed"; if (match(val,"GDRDMA")) color="green" }
      printf "    c%d_%d -> c%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",penwidth=\"%s\"];\n", r, rank_switch, r, rank, val, color, style, penwidth > dotfile
      for(s=0; s<=max_collnet_rank; s++) {
        if((rank "," s) in collnet) {
          printf "    c%d_%d -> c%d_%d [label=\"\",color=\"green\",style=\"solid\",penwidth=\"1.5\"];\n", r, rank, r, s > dotfile
        }
      }
    }
    for(s=0; s<=max_collnet_rank; s++) {
      printf "    c%d_%d [label=\"%d\",fontsize=\"%d\"];\n", r, s, s, fs > dotfile
    }
    printf "    c%d_%d [label=\"SHARP:%d\",fontsize=\"%d\",shape=box];\n", r, rank_switch, r, fs > dotfile
    printf "  }\n}\n" > dotfile
    close(dotfile)
  }

  close(manifest)
  printf "Generated %d DOT files in %s\n", (max_treedn+1) + (max_ring+1) + (has_collnet ? max_collnet+1 : 0), outdir > "/dev/stderr"
}
