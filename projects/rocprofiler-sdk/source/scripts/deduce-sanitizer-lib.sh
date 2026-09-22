#!/bin/bash -e

TEST_SOURCE=${PWD}/deduce-sanitizer-lib.cpp
TEST_BINARY=${PWD}/deduce-sanitizer-lib.out
TEST_RESULT=${PWD}/deduce-sanitizer-lib.txt

LIBNAME=${1}
shift

cat << EOF > ${TEST_SOURCE}
#include <string>

int
main(int argc, char** argv)
{
    auto ret = 0;
    if(argc > 1) ret = std::stoi(argv[1]);
    return ret;
}
EOF

${@} ${TEST_SOURCE} -o ${TEST_BINARY} &> /dev/stderr

ldd ${TEST_BINARY} | grep ${LIBNAME} | sed -E 's/.* => //g' | awk '{print $1}' > ${TEST_RESULT}
cat ${TEST_RESULT}
