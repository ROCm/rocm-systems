 ; Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 ;Permission is hereby granted, free of charge, to any person obtaining a copy
 ;of this software and associated documentation files (the "Software"), to deal
 ;in the Software without restriction, including without limitation the rights
 ;to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 ;copies of the Software, and to permit persons to whom the Software is
 ;furnished to do so, subject to the following conditions:

 ;The above copyright notice and this permission notice shall be included in
 ;all copies or substantial portions of the Software.

 ;THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 ;IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 ;FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 ;AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 ;LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 ;OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 ;THE SOFTWARE. 


ifndef _WIN64
    .386
    .model flat, c
endif ; !_WIN64

OPTION PROLOGUE:NONE
OPTION EPILOGUE:NONE
.code

ifndef _WIN64

_StackContext_setjmp proc
    mov ecx,[esp]
    mov edx,4[esp]
    mov [edx],ebx
    lea eax,4[esp]
    mov 4[edx],eax
    mov 8[edx],ebp
    mov 0Ch[edx],edi
    mov 10h[edx],esi
    mov 14h[edx],ecx
    xor eax,eax
    ret
_StackContext_setjmp endp

_StackContext_longjmp proc
    mov edx,4[esp]
    mov eax,8[esp]
    mov ebx,[edx]
    mov esp,4[edx]
    mov ebp,8[edx]
    mov edi,0Ch[edx]
    mov esi,10h[edx]
    mov ecx,14h[edx]
    jmp ecx
_StackContext_longjmp endp

else ; _WIN64

_Os_setCurrentStackPtr proc
    pop r8
    mov rsp,rcx
    push r8
    ret
_Os_setCurrentStackPtr endp

_StackContext_setjmp proc
    mov r8,[rsp]
    mov [rcx],rbx
    lea r9,8[rsp]
    mov 8[rcx],r9
    mov 10h[rcx],rbp
    mov 18h[rcx],rsi
    mov 20h[rcx],rdi
    mov 28h[rcx],r12
    mov 30h[rcx],r13
    mov 38h[rcx],r14
    mov 40h[rcx],r15
    mov 48h[rcx],r8
    stmxcsr 50h[rcx]
    fnstcw 54h[rcx]
    movdqa 60h[rcx],xmm6
    movdqa 70h[rcx],xmm7
    movdqa 80h[rcx],xmm8
    movdqa 90h[rcx],xmm9
    movdqa 0A0h[rcx],xmm10
    movdqa 0B0h[rcx],xmm11
    movdqa 0C0h[rcx],xmm12
    movdqa 0D0h[rcx],xmm13
    movdqa 0E0h[rcx],xmm14
    movdqa 0F0h[rcx],xmm15
    xor rax,rax
    ret
_StackContext_setjmp endp

_StackContext_longjmp proc
    mov rax,rdx
    mov rbx,[rcx]
    mov rsp,8[rcx]
    mov rbp,10h[rcx]
    mov rsi,18h[rcx]
    mov rdi,20h[rcx]
    mov r12,28h[rcx]
    mov r13,30h[rcx]
    mov r14,38h[rcx]
    mov r15,40h[rcx]
    mov rdx,48h[rcx]
    ldmxcsr 50h[rcx]
    fnclex
    fldcw 54h[rcx]
    movdqa xmm6,60h[rcx]
    movdqa xmm7,70h[rcx]
    movdqa xmm8,80h[rcx]
    movdqa xmm9,90h[rcx]
    movdqa xmm10,0A0h[rcx]
    movdqa xmm11,0B0h[rcx]
    movdqa xmm12,0C0h[rcx]
    movdqa xmm13,0D0h[rcx]
    movdqa xmm14,0E0h[rcx]
    movdqa xmm15,0F0h[rcx]
    jmp rdx
_StackContext_longjmp endp

endif ; _WIN64

end
