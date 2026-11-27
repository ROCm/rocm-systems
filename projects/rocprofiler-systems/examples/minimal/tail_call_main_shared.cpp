// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

extern "C" int
tail_call_shared_workload(int argc);

// main() ends with a tail call into a function of a shared library, which is a jmp
// through the PLT, not a ret.
int
main(int argc, char** /*argv*/)
{
    return tail_call_shared_workload(argc);
}
