/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
