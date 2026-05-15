/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef GL_INTEROP_H_
#define GL_INTEROP_H_

#define GL_RESOURCE_ATTACH_TEXTURE_AMD 0x12a000
#define GL_RESOURCE_ATTACH_FRAMEBUFFER_AMD 0x12a001
#define GL_RESOURCE_ATTACH_RENDERBUFFER_AMD 0x12a002
#define GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD 0x12a003

#ifndef GLuintp
typedef uintptr_t GLuintp;
#endif

#ifndef GLlonglong
typedef long long GLlonglong;
#endif

typedef struct GLResourceRec {
  GLenum type;
  GLuint name;
  GLuint flags;
  GLuintp mbResHandle;
  GLuint level;      ///< Start level to attach
  GLuint numLevels;  ///< Number of levels to attach (can be set to GL_TEXTURE_ALL_LEVELS_AMD)
  GLuint layer;      ///< Start layer to attach
  GLuint numLayers;  ///< Number of layers to attach (can be set to GL_TEXTURE_ALL_LAYERS_AMD)
} GLResource;

typedef struct GLResourceDimRec {
  GLuint width;
  GLuint height;
  GLuint depth;
} GLResourceDim;
#define GLRDATA_MAX_LAYERS 8192
#define GL_RESOURCE_DATA_VERSION 7
typedef struct GLResourceDataRec {
  GLuint size;
  GLuint version;
  GLuint surfaceSize;  ///< Size of the base surface.
  GLuint pad;
  GLuintp offset;  ///< Offset pointing to the sub resource's surface.
  GLuintp mbResHandle;
  GLuint format;
  GLuint flags;
  GLuint tilingMode;
  GLuint swizzles[GLRDATA_MAX_LAYERS];
  GLResourceDim paddedDimensions;
  GLResourceDim rawDimensions;
  GLlonglong cardAddr;  ///< Address of the base surface (add offset to get the actual address of
                        ///< the sub surface)
  GLlonglong p2pAddr;
  GLlonglong mc_size;
  GLuint cpuAccess;
  GLuintp handle;
  GLuint perSurfTileInfo;
  GLuint objectAttribType;
  GLuint sharedBufferID;
  GLuint levels;
  GLuint swizzlesMip[GLRDATA_MAX_LAYERS];
  GLuint textureSRDSize;
  GLuint samplerSRDSize;
  GLuint textureSRD[8];
  GLuint samplerSRD[8];
  GLboolean isDoppDesktopTexture;
  GLboolean isDoppPresentTexture;
  GLuint isTilingRotated;
  GLuint vidpnSourceId;
  GLboolean isDisplayable;

} GLResourceData;

#ifdef _WIN32
typedef BOOL(WINAPI* PFNWGLBEGINCLINTEROPAMD)(HGLRC hglrc, GLuint flags);
typedef BOOL(WINAPI* PFNWGLENDCLINTEROPAMD)(HGLRC hglrc, GLuint flags);
typedef BOOL(WINAPI* PFNWGLRESOURCEATTACHAMD)(HGLRC hglrc, GLvoid* resource, GLvoid* pResourceData);
typedef BOOL(WINAPI* PFNWGLRESOURCEDETACHAMD)(HGLRC hglrc, GLvoid* resource);
typedef BOOL(WINAPI* PFNWGLGETCONTEXTGPUINFOAMD)(HGLRC hglrc, LUID* adapterLUID,
                                                 UINT* chainBitmask);
typedef BOOL(WINAPI* PFNWGLMULTIRESOURCEACQUIREAMD)(HGLRC hglrc, GLvoid** resource,
                                                    GLuint numResources);
typedef BOOL(WINAPI* PFNWGLMULTIRESOURCERELEASEAMD)(HGLRC hglrc, GLvoid** resource,
                                                    GLuint numResources);

#endif /* _WIN32 */

#endif /* GL_INTEROP_H_*/
