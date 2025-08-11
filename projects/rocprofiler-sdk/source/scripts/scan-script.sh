#!/bin/bash

find-keyword() 
{
	echo -e "Searching for $1..."
	if [ "$(grep -r $1 .)" ]; 
	then
		exit 1
	fi
}
for i in $KEYWORDS;
do
	find-keyword $i
done
