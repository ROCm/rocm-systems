#!/bin/bash

#===------------------------------------------------------------------------===#
#
# Create source tarballs for the rocm-systems release.
#
#===------------------------------------------------------------------------===#

# Adapted from
# https://github.com/llvm/llvm-project/blob/llvmorg-21.1.5/llvm/utils/release/export.sh
# 
# The original license header
#
#===-- tag.sh - Tag the LLVM release candidates ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# Changes for rocm-systems made under this license
#
# MIT License
# 
# Copyright (C) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e

# These are just standins for the usage printout
projects="aqlprofile clr hip hipother hip-tests rdc rocm-core rocminfo rocm-smi-lib \
          rocprofiler rocprofiler-compute rocprofiler-register rocprofiler-sdk \
          rocprofiler-systems rocr-runtime roctracer"
shared="amdgpu-windows-interop"

release=""
rc=""
yyyymmdd=$(date +'%Y%m%d')
snapshot=""
template='${PROJECT}-${RELEASE}${RC}.tar.xz'

usage() {
cat <<EOF
Export the Git sources and build tarballs from them.

Usage: $(basename $0) [-release|--release <major>.<minor>.<patch>]
                      [-rc|--rc <num>]
                      [-final|--final]
                      [-git-ref|--git-ref <git-ref>]
                      [-template|--template <template>]

Flags:

  -release  | --release <major>.<minor>.<patch>    The version number of the release
  -rc       | --rc <num>                           The release candidate number
  -final    | --final                              When provided, this option will disable the rc flag
  -git-ref  | --git-ref <git-ref>                  (optional) Use <git-ref> to determine the release
  -template | --template <template>                (optional) Possible placeholders: \$PROJECT \$YYYYMMDD \$GIT_REF \$RELEASE \$RC.
                                                   Defaults to '${template}'.

The following list shows the filenames (with <placeholders>) for the artifacts
that are being generated (given that you don't touch --template).

$(echo "$projects "| sed 's/\([a-z-]\+\) /  * \1-<RELEASE><RC>.tar.xz \n/g')
$(echo "$shared "| sed 's/\([a-z-]\+\) /  * \1-<RELEASE><RC>.tar.xz \n/g')

Additional files being generated:

  * rocm-systems-<RELEASE><RC>.tar.xz    (the complete source project)

To ease the creation of snapshot builds, we also provide these files

  * rocm-release-<YYYYMMDD>.txt        (contains the <RELEASE> as a text)
  * rocm-rc-<YYYYMMDD>.txt             (contains the rc version passed to the invocation of $(basename $0))
  * rocm-git-revision-<YYYYMMDD>.txt   (contains the current git revision sha1)

Example values for the placeholders:

  * <RELEASE>  -> 7.1.0
  * <YYYYMMDD> -> 20251105   (the date when executing this script)
  * <RC>       -> rc4        (will be empty when using --git-ref)

In order to generate snapshots of the upstream main branch you could do this for example:

  $(basename $0) --git-ref develop --template '\${PROJECT}-\${YYYYMMDD}.tar.xz'

EOF
}

template_file() {
    export PROJECT=$1 YYYYMMDD=$yyyymmdd RC=$rc RELEASE=$release GIT_REF=$git_rev
    basename $(echo $template | envsubst '$PROJECT $RELEASE $RC $YYYYMMDD $GIT_REF')
    unset PROJECT YYYYMMDD RC RELEASE GIT_REF
}

export_sources() {
    local tag="rocm-$release"

    # Assuming this script is in $src_dir/utils/release/export.sh, set the src_dir
    src_dir=`realpath "$(dirname $0)"/../..`
    [ -d $src_dir/.git ] || ( echo "No git repository at $src_dir" ; exit 1 )

    if [ -n "$snapshot" ]; then
        pushd $src_dir
        cdate=`git log -1 --date=format:%Y%m%d --format=%cd $snapshot`
        cshort_hash=${snapshot:0:7}
        release=${cdate}-${cshort_hash}
        popd
    fi

    tag="rocm-$release"

    if [ "$rc" = "final" ]; then
        rc=""
    else
        tag="$tag-$rc"
    fi

    target_dir=$(pwd)

    echo "Creating tarball for rocm-systems ..."
    pushd $src_dir/
    tree_id=$tag
    [ -n "$snapshot" ] && tree_id="$snapshot"
    echo "Tree ID to archive: $tree_id"

    git_rev=$(git rev-parse $tree_id)
    echo "git revision: $git_rev"
    echo "$release" > $target_dir/rocm-release-$yyyymmdd.txt
    echo "$rc" > $target_dir/rocm-rc-$yyyymmdd.txt
    echo "$git_rev" > $target_dir/rocm-git-revision-$yyyymmdd.txt

    git archive --prefix=rocm-systems-$release$rc/ $tree_id . | xz -T0 >$target_dir/$(template_file rocm-systems)

    # progammatically determine the list of projects
    # expect the output to be in this form
    # 040000 tree 826d30d68a55d8cff0986f1caa2c52849a68827a	composablekernel
    projects=`git ls-tree -d ${tree_id}:projects | awk '{ print $4 }' `
    for proj in $projects; do
        echo "Creating tarball for $proj ..."
        git archive --prefix=$proj-$release$rc/ $tree_id:projects/$proj | xz -T0 >$target_dir/$(template_file $proj)
    done

    # Similarly there should be a number of projects in the shared/ dir
    shared=`git ls-tree -d ${tree_id}:shared | awk '{ print $4 }' `
    for proj in $shared; do
        echo "Creating tarball for $proj ..."
        git archive --prefix=$proj-$release$rc/ $tree_id:shared/$proj | xz -T0 >$target_dir/$(template_file $proj)
    done

    popd
}

while [ $# -gt 0 ]; do
    case $1 in
        -release )
            shift
            release=$1
            ;;
        -rc )
            shift
            rc="rc$1"
            ;;
        -final )
            rc="final"
            ;;
        -git-ref )
            shift
            snapshot="$1"
            ;;
        -template )
            shift
            template="$1"
            ;;
        -h | -help | --help )
            usage
            exit 0
            ;;
        * )
            echo "unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -n "$snapshot" ]; then
    if [[ "$rc" != "" || "$release" != "" ]]; then
        echo "error: must not specify -rc or -release when creating a snapshot"
        exit 1
    fi
elif [ -z "$release" ]; then
    echo "error: need to specify a release version"
    exit 1
fi

# Make sure umask is not overly restrictive.
umask 0022

export_sources
exit 0
