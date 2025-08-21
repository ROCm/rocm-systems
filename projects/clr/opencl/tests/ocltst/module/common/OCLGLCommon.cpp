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

#include "OCLGLCommon.h"

#include <cmath>
#include <cstring>

void OCLGLCommon::open(unsigned int test, char* units, double& conversion, unsigned int& deviceId) {
  // OpenCL Initialization
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test (%d)", error_);

  char name[1024] = {0};
  size_t size = 0;

  if (deviceId >= deviceCount_) {
    _errorFlag = true;
    return;
  }

  // Check that the device supports CL/GL interop extension
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024, name, &size);
  if (!strstr(name, "cl_khr_gl_sharing")) {
    printf("KHR GL sharing extension is required for this test!\n");
    _errorFlag = true;
    return;
  }

  // OpenGL Initialization
  bool retVal = initializeGLContext(hGL_);
  CHECK_RESULT((retVal == CL_SUCCESS), "Error opening test (%d)", error_);

  createCLContextFromGLContext(hGL_);
  // Device id may change to match the GL context. Return the id to caller
  deviceId = _deviceId;
}

bool OCLGLCommon::IsGLEnabled(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  bool bResult = initializeGLContext(hGL_);
  if (bResult) {
    deleteGLContext(hGL_);
  }
  OCLTestImp::close();
  return bResult;
}

void OCLGLCommon::gluPerspective(double fovy, double aspect, double zNear, double zFar) {
  double xmin, xmax, ymin, ymax;
  ymax = zNear * tan(fovy * 3.149 / 360.0);
  ymin = -ymax;
  xmin = ymin * aspect;
  xmax = ymax * aspect;
  glFrustum(xmin, xmax, ymin, ymax, zNear, zFar);
}

unsigned int OCLGLCommon::close(void) {
  makeCurrent(hGL_);
  unsigned int retVal = OCLTestImp::close();
  deleteGLContext(hGL_);
  return retVal;
}

void OCLGLCommon::dumpBuffer(float* pBuffer, const char fileName[], unsigned int dimSize) {
  if (pBuffer) {
    FILE* f = fopen(fileName, "w");
    if (NULL != f) {
      unsigned int i, j;
      for (i = 0; i < dimSize; i++) {
        for (j = 0; j < dimSize; j++) {
          fprintf(f, "%e,\t", pBuffer[i * (dimSize) + j]);
        }
        fprintf(f, "\n");
      }
      fclose(f);
    }
  }
}

bool OCLGLCommon::createGLFragmentProgramFromSource(const char* source, GLuint& shader,
                                                    GLuint& program) {
  shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);
  printShaderInfoLog(shader);
  program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  printProgramInfoLog(program);

  return program != 0;
}

int OCLGLCommon::printOglError(char* file, int line) {
  //
  // Returns 1 if an OpenGL error occurred, 0 otherwise.
  //
  GLenum glErr;
  int retCode = 0;

  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    printf("glError in file %s @ line %d: %d\n", file, line, glErr);
    retCode = 1;
  }
  return retCode;
}

//
// Print out the information log for a shader object
//
void OCLGLCommon::printShaderInfoLog(GLuint shader) {
  int infologLength = 0;
  int charsWritten = 0;
  GLchar* infoLog;

  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

  if (infologLength > 0) {
    infoLog = (GLchar*)malloc(infologLength);
    if (infoLog == NULL) {
      printf("ERROR: Could not allocate InfoLog buffer\n");
      return;
    }
    glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
    printf("Shader InfoLog:\n%s\n\n", infoLog);
    free(infoLog);
  }
}

void OCLGLCommon::printProgramInfoLog(GLuint program) {
  int infologLength = 0;
  int charsWritten = 0;
  GLchar* infoLog;

  // printOpenGLError();  // Check for OpenGL errors

  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

  // printOpenGLError();  // Check for OpenGL errors

  if (infologLength > 0) {
    infoLog = (GLchar*)malloc(infologLength);
    if (infoLog == NULL) {
      printf("ERROR: Could not allocate InfoLog buffer\n");
      exit(1);
    }
    glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
    printf("Program InfoLog:\n%s\n\n", infoLog);
    free(infoLog);
  }
  //  printOpenGLError();  // Check for OpenGL errors
}
