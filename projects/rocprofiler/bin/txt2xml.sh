#!/bin/bash
timestamp=`date +%y%m%d_%H%M%S`

if [ $# = 0 ] ; then
  echo "Usage: $0 <input text file> [output dir]"
  exit -1
fi

input=$1
outdir=$2
if [ -z "$outdir" ] ; then
  outdir="."
fi

range=""
kernel=""
gpu_index=""

parse() {
  scan="$1"
  index=0
  while read -r line ; do
    line=`echo $line | sed "s/\s*#.*$//"`
    if [ -z "$line" ] ; then
      continue
    fi
  
    feature=`echo $line | sed -n "s/^\s*\([a-z]*\)\s*:.*$/\1/p"`
    line=`echo $line | sed "s/^[^:]*:\s*//"`
    line=`echo "$line" | sed -e "s/\s*=\s*/=/g" -e "s/\s*:\s*/:/g" -e "s/,\{1,\}/ /g" -e "s/\s\{1,\}/ /g" -e "s/\s*$//"`

    if [ "$scan" = 0 ] ; then
      line=`echo "$line" | sed -e "s/ /,/g"`
      if [ "$feature" == "range" ] ; then
        range=$line
      fi
      if [ "$feature" == "kernel" ] ; then
        kernel=$line
      fi
      if [ "$feature" == "gpu" ] ; then
        gpu_index=$line
      fi
    else
      output=$outdir/input${index}.xml
      header="# $timestamp '$output' generated with '$0 $*'"
    
      if [ "$feature" == "pmc" ] ; then
        line=`echo "$line" | sed -e "s/ /,/g"`
        cat >> $output <<EOF
$header
<metric range="$range" kernel="$kernel" gpu_index="$gpu_index"></metric>
<metric name=$line ></metric>
EOF
      fi
    
      if [ "$feature" == "sqtt" ] ; then
        cat >> $output <<EOF
$header
<metric range="$range" kernel="$kernel" gpu_index="$gpu_index"></metric>
<trace name="SQTT"><parameters $line ></parameters></trace>
EOF
      fi
    fi
  
    index=$((index + 1))
  done < $input
}

parse 0
parse 1

exit 0
