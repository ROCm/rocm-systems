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
  PREVIOUS_TAG=${TAG_ARR[0]}
  CURRENT_TAG=${TAG_ARR[1]}

  PREV_CMT_NUM=`git rev-list --count $PREVIOUS_TAG`
  CURR_CMT_NUM=`git rev-list --count $CURRENT_TAG`

  # Commits since prevous tag:
  if [[ -z $PREV_CMT_NUM || -z $CURR_CMT_NUM ]]; then
    let NUM_COMMITS="0"
  else
    let NUM_COMMITS="${CURR_CMT_NUM}-${PREV_CMT_NUM}"
  fi
  echo $NUM_COMMITS
}

case $TARGET in
    count) commits_since_last_tag ;;
    *) die "Invalid target $target" ;;
esac

exit 0

