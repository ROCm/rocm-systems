/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

GLPREFIX(void, glBindBuffer, (GLenum target, GLuint buffer))
GLPREFIX(void, glBindRenderbuffer, (GLenum target, GLuint renderbuffer))
GLPREFIX(void, glBindTexture, (GLenum target, GLuint texture))

// 10
GLPREFIX(void, glGetBufferParameteriv, (GLenum target, GLenum pname, GLint* params))
GLPREFIX(GLenum, glGetError, (void))
GLPREFIX(void, glFinish, (void))
GLPREFIX(void, glFlush, (void))
GLPREFIX(GLenum, glClientWaitSync, (GLsync sync, GLbitfield flags, GLuint64 timeout))
GLPREFIX(void, glGetIntegerv, (GLenum pname, GLint* params))
GLPREFIX(void, glGetRenderbufferParameterivEXT, (GLenum target, GLenum pname, GLint* params))
GLPREFIX(void, glGetTexLevelParameteriv, (GLenum target, GLint level, GLenum pname, GLint* params))
GLPREFIX(void, glGetTexParameteriv, (GLenum target, GLenum pname, GLint* params))

GLPREFIX(GLboolean, glIsBuffer, (GLuint buffer))
GLPREFIX(GLboolean, glIsRenderbufferEXT, (GLuint renderbuffer))
GLPREFIX(GLboolean, glIsTexture, (GLuint texture))
#undef GLPREFIX
