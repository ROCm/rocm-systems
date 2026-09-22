# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

        .text
.globl _StackContext_setjmp
        .type _StackContext_setjmp, @function
_StackContext_setjmp:

#if defined(_LP64)
        movq (%rsp), %rsi
        movq %rbx, (%rdi)
        lea  8(%rsp), %rax
        movq %rax, 8(%rdi)
        movq %rbp, 16(%rdi)
        movq %r12, 24(%rdi)
        movq %r13, 32(%rdi)
        movq %r14, 40(%rdi)
        movq %r15, 48(%rdi)
        movq %rsi, 56(%rdi)
#else // _LP64
        movl (%esp), %ecx
        movl 4(%esp), %edx
        movl %ebx, (%edx)
        lea  4(%esp), %eax
        movl %eax, 4(%edx)
        movl %ebp, 8(%edx)
        movl %edi, 12(%edx)
        movl %esi, 16(%edx)
        movl %ecx, 20(%edx)
#endif // _LP64
        xor %eax, %eax
        ret

.globl _StackContext_longjmp
        .type _StackContext_longjmp, @function
_StackContext_longjmp:

#if defined(_LP64)
        mov %rsi, %rax
        movq (%rdi), %rbx
        movq 8(%rdi), %rsp
        movq 16(%rdi), %rbp
        movq 24(%rdi), %r12
        movq 32(%rdi), %r13
        movq 40(%rdi), %r14
        movq 48(%rdi), %r15
        movq 56(%rdi), %r8
        jmp *%r8
#else // !_LP64
        movl 4(%esp), %edx
        movl 8(%esp), %eax
        movl (%edx), %ebx
        movl 4(%edx), %esp
        movl 8(%edx), %ebp
        movl 12(%edx), %edi
        movl 16(%edx), %esi
        movl 20(%edx), %ecx
        jmp *%ecx
#endif // !_LP64

.section .note.GNU-stack,"",%progbits

