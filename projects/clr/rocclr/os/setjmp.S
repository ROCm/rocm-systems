 # Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 #Permission is hereby granted, free of charge, to any person obtaining a copy
 #of this software and associated documentation files (the "Software"), to deal
 #in the Software without restriction, including without limitation the rights
 #to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 #copies of the Software, and to permit persons to whom the Software is
 #furnished to do so, subject to the following conditions:

 #The above copyright notice and this permission notice shall be included in
 #all copies or substantial portions of the Software.

 #THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 #IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 #FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 #AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 #LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 #OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 #THE SOFTWARE. 


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

