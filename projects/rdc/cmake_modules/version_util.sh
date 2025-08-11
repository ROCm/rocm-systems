#!/bin/bash

# Handle commandline args
while [ "$1" != "" ]; do
    case $1 in
        -c )  # Commits since prevous tag
            TARGET="count" ;;
         * )
            TARGET="count"
            break ;;
    esac
    shift 1
done
TAG_PREFIX=$1
reg_ex="${TAG_PREFIX}*"

commits_since_last_tag() {
  TAG_ARR=(`git tag --sort=committerdate -l ${reg_ex} | tail -2`)
  # if we don't have 2 tags, just say there were 0 commits since
  # last tag
  if [ ${#TAG_ARR[@]} != 2 ]; then
     echo 0
     exit 0
  fi

  PREVIOUS_TAG=${TAG_ARR[0]}
  CURRENT_TAG=${TAG_ARR[1]}

  PREV_CMT_NUM=`git rev-list --count $PREVIOUS_TAG`
  CURR_CMT_NUM=`git rev-list --count $CURRENT_TAG`

  # Commits since prevous tag:
  let NUM_COMMITS="${CURR_CMT_NUM}-${PREV_CMT_NUM}"
  echo $NUM_COMMITS
}

case $TARGET in
    count) commits_since_last_tag ;;
    *) die "Invalid target $target" ;;
esac

exit 0

