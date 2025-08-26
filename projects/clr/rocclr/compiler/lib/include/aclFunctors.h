/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

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

#ifndef _ACL_FUNCTORS_0_8_H_
#define _ACL_FUNCTORS_0_8_H_

//! Callback for the log function function pointer that many
// API calls take to have the calling application receive
// information on what errors occur.
typedef void (*aclLogFunction_0_8)(const char* msg, size_t size);

typedef bool (*aclJITSymbolCallback)(const char*, const void*, void*);
typedef void* aclJITObjectImage;
typedef const void* constAclJITObjectImage;

typedef acl_error(ACL_API_ENTRY* InsertSec_0_8)(aclCompiler* cl, aclBinary* binary,
                                                const void* data, size_t data_size,
                                                aclSections id) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* InsertSym_0_8)(aclCompiler* cl, aclBinary* binary,
                                                const void* data, size_t data_size, aclSections id,
                                                const char* symbol) ACL_API_0_8;

typedef const void*(ACL_API_ENTRY* ExtractSec_0_8)(aclCompiler* cl, const aclBinary* binary,
                                                   size_t* size, aclSections id,
                                                   acl_error* error_code)ACL_API_0_8;

typedef const void*(ACL_API_ENTRY* ExtractSym_0_8)(aclCompiler* cl, const aclBinary* binary,
                                                   size_t* size, aclSections id, const char* symbol,
                                                   acl_error* error_code)ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* RemoveSec_0_8)(aclCompiler* cl, aclBinary* binary,
                                                aclSections id) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* RemoveSym_0_8)(aclCompiler* cl, aclBinary* binary, aclSections id,
                                                const char* symbol) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* QueryInfo_0_8)(aclCompiler* cl, const aclBinary* binary,
                                                aclQueryType query, const char* kernel,
                                                void* data_ptr, size_t* ptr_size) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* AddDbgArg_0_8)(aclCompiler* cl, aclBinary* bin, const char* kernel,
                                                const char* name, bool byVal) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* RemoveDbgArg_0_8)(aclCompiler* cl, aclBinary* bin,
                                                   const char* kernel,
                                                   const char* name) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* Compile_0_8)(aclCompiler* cl, aclBinary* bin, const char* options,
                                              aclType from, aclType to,
                                              aclLogFunction_0_8 compile_callback) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* Link_0_8)(aclCompiler* cl, aclBinary* src_bin,
                                           unsigned int num_libs, aclBinary** libs,
                                           aclType link_mode, const char* options,
                                           aclLogFunction_0_8 link_callback) ACL_API_0_8;

typedef const char*(ACL_API_ENTRY* CompLog_0_8)(aclCompiler* cl)ACL_API_0_8;

typedef const void*(ACL_API_ENTRY* RetrieveType_0_8)(aclCompiler* cl, const aclBinary* bin,
                                                     const char* name, size_t* data_size,
                                                     aclType type,
                                                     acl_error* error_code)ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* SetType_0_8)(aclCompiler* cl, aclBinary* bin, const char* name,
                                              aclType type, const void* data,
                                              size_t size) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* ConvertType_0_8)(aclCompiler* cl, aclBinary* bin, const char* name,
                                                  aclType type) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* Disassemble_0_8)(aclCompiler* cl, aclBinary* bin,
                                                  const char* kernel,
                                                  aclLogFunction_0_8 disasm_callback) ACL_API_0_8;

typedef const void*(ACL_API_ENTRY* GetDevBinary_0_8)(aclCompiler* cl, const aclBinary* bin,
                                                     const char* kernel, size_t* size,
                                                     acl_error* error_code)ACL_API_0_8;

typedef aclLoaderData*(ACL_API_ENTRY* LoaderInit_0_8)(aclCompiler* cl, aclBinary* bin,
                                                      aclLogFunction_0_8 callback,
                                                      acl_error* error);

typedef acl_error(ACL_API_ENTRY* LoaderFini_0_8)(aclLoaderData* data);

typedef aclModule*(ACL_API_ENTRY* FEToIR_0_8)(aclLoaderData* ald, const char* source,
                                              size_t data_size, aclContext* ctx,
                                              acl_error* error)ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* SourceToISA_0_8)(aclLoaderData* ald, const char* source,
                                                  size_t data_size) ACL_API_0_8;

typedef aclModule*(ACL_API_ENTRY* IRPhase_0_8)(aclLoaderData* data, aclModule* ir, aclContext* ctx,
                                               acl_error* error)ACL_API_0_8;

typedef aclModule*(ACL_API_ENTRY* LinkPhase_0_8)(aclLoaderData* data, aclModule* ir,
                                                 unsigned int num_libs, aclModule** libs,
                                                 aclContext* ctx, acl_error* error)ACL_API_0_8;

typedef const void*(ACL_API_ENTRY* CGPhase_0_8)(aclLoaderData* data, aclModule* ir, aclContext* ctx,
                                                acl_error* error)ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* DisasmISA_0_8)(aclLoaderData* data, const char* kernel,
                                                const void* isa_code, size_t isa_size) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* SetupLoaderObject_0_8)(aclCompiler* cl) ACL_API_0_8;

typedef aclJITObjectImage(ACL_API_ENTRY* JITObjectImageCreate_0_8)(
    const void* buffer, size_t length, aclBinary* bin, acl_error* error_code) ACL_API_0_8;

typedef aclJITObjectImage(ACL_API_ENTRY* JITObjectImageCopy_0_8)(const void* buffer, size_t length,
                                                                 acl_error* error_code) ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* JITObjectImageDestroy_0_8)(aclJITObjectImage image) ACL_API_0_8;

typedef size_t(ACL_API_ENTRY* JITObjectImageSize_0_8)(aclJITObjectImage image,
                                                      acl_error* error_code) ACL_API_0_8;

typedef const char*(ACL_API_ENTRY* JITObjectImageData_0_8)(aclJITObjectImage image,
                                                           acl_error* error_code)ACL_API_0_8;

typedef acl_error(ACL_API_ENTRY* JITObjectImageFinalize_0_8)(aclJITObjectImage image) ACL_API_0_8;

typedef size_t(ACL_API_ENTRY* JITObjectImageGetGlobalsSize_0_8)(aclJITObjectImage image,
                                                                acl_error* error_code) ACL_API_0_8;

typedef bool (*JITSymbolCallback_0_8)(const char*, const void*, void*);

typedef acl_error(ACL_API_ENTRY* JITObjectImageIterateSymbols_0_8)(
    aclJITObjectImage image, JITSymbolCallback_0_8 jit_callback, void* data) ACL_API_0_8;

typedef char*(ACL_API_ENTRY* JITObjectImageDisassembleKernel_0_8)(constAclJITObjectImage image,
                                                                  const char* kernel,
                                                                  acl_error* error_code)ACL_API_0_8;

typedef void* (*AllocFunc_0_8)(size_t size)ACL_API_0_8;

typedef void (*FreeFunc_0_8)(void* ptr) ACL_API_0_8;

#endif  // _ACL_FUNCTORS_0_8_H_
