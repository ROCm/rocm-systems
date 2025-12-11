/*
 * Copyright (c) 2017-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#include "icd.h"
#include "icd_windows_dxgk.h"

#if defined(OPENCL_ICD_LOADER_REQUIRE_WDK)
#include <windows.h>

#ifndef NTSTATUS
typedef LONG NTSTATUS;
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
#define STATUS_BUFFER_TOO_SMALL ((NTSTATUS)0xC0000023)
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif

#include <d3dkmthk.h>
#endif

bool khrIcdOsVendorsEnumerateDXGK(void) {
  bool ret = false;
  int result = 0;
#if defined(OPENCL_ICD_LOADER_REQUIRE_WDK)
#if defined(DXGKDDI_INTERFACE_VERSION_WDDM2_4) &&                                                  \
    (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WDDM2_4)
  // Get handle to GDI Runtime
  HMODULE h = LoadLibrary("gdi32.dll");
  if (h == NULL) return ret;

  if (GetProcAddress((HMODULE)h, "D3DKMTSubmitPresentBltToHwQueue"))  // OS Version check
  {
    D3DKMT_ADAPTERINFO* pAdapterInfo = NULL;
    D3DKMT_ENUMADAPTERS2 EnumAdapters;
    NTSTATUS Status = STATUS_SUCCESS;

    char cszLibraryName[MAX_PATH] = {0};
    EnumAdapters.NumAdapters = 0;
    EnumAdapters.pAdapters = NULL;
    PFND3DKMT_ENUMADAPTERS2 pEnumAdapters2 =
        (PFND3DKMT_ENUMADAPTERS2)GetProcAddress((HMODULE)h, "D3DKMTEnumAdapters2");
    if (!pEnumAdapters2) {
      KHR_ICD_TRACE("GetProcAddress failed for D3DKMT_ENUMADAPTERS2\n");
      goto out;
    }
    while (1) {
      EnumAdapters.NumAdapters = 0;
      EnumAdapters.pAdapters = NULL;
      Status = pEnumAdapters2(&EnumAdapters);
      if (Status == STATUS_BUFFER_TOO_SMALL) {
        // Number of Adapters increased between calls, retry;
        continue;
      } else if (!NT_SUCCESS(Status)) {
        KHR_ICD_TRACE("D3DKMT_ENUMADAPTERS2 status != SUCCESS\n");
        goto out;
      }
      break;
    }
    pAdapterInfo =
        (D3DKMT_ADAPTERINFO*)malloc(sizeof(D3DKMT_ADAPTERINFO) * (EnumAdapters.NumAdapters));
    if (pAdapterInfo == NULL) {
      KHR_ICD_TRACE("Allocation failure for AdapterInfo buffer\n");
      goto out;
    }
    EnumAdapters.pAdapters = pAdapterInfo;
    Status = pEnumAdapters2(&EnumAdapters);
    if (!NT_SUCCESS(Status)) {
      KHR_ICD_TRACE("D3DKMT_ENUMADAPTERS2 status != SUCCESS\n");
      goto out;
    }
    const char* cszOpenCLRegKeyName = getOpenCLRegKeyName();
    const int szOpenCLRegKeyName =
        (int)(strlen(cszOpenCLRegKeyName) + 1) * sizeof(cszOpenCLRegKeyName[0]);
    for (UINT AdapterIndex = 0; AdapterIndex < EnumAdapters.NumAdapters; AdapterIndex++) {
      D3DDDI_QUERYREGISTRY_INFO queryArgs = {0};
      D3DDDI_QUERYREGISTRY_INFO* pQueryArgs = &queryArgs;
      D3DDDI_QUERYREGISTRY_INFO* pQueryBuffer = NULL;
      queryArgs.QueryType = D3DDDI_QUERYREGISTRY_ADAPTERKEY;
      queryArgs.QueryFlags.TranslatePath = TRUE;
      queryArgs.ValueType = REG_SZ;
      result = MultiByteToWideChar(CP_ACP, 0, cszOpenCLRegKeyName, szOpenCLRegKeyName,
                                   queryArgs.ValueName, ARRAYSIZE(queryArgs.ValueName));
      if (!result) {
        KHR_ICD_TRACE("MultiByteToWideChar status != SUCCESS\n");
        continue;
      }
      D3DKMT_QUERYADAPTERINFO queryAdapterInfo = {0};
      queryAdapterInfo.hAdapter = pAdapterInfo[AdapterIndex].hAdapter;
      queryAdapterInfo.Type = KMTQAITYPE_QUERYREGISTRY;
      queryAdapterInfo.pPrivateDriverData = &queryArgs;
      queryAdapterInfo.PrivateDriverDataSize = sizeof(queryArgs);
      Status = D3DKMTQueryAdapterInfo(&queryAdapterInfo);
      if (!NT_SUCCESS(Status)) {
        // Try a different value type.  Some vendors write the key as a multi-string type.
        queryArgs.ValueType = REG_MULTI_SZ;
        Status = D3DKMTQueryAdapterInfo(&queryAdapterInfo);
        if (NT_SUCCESS(Status)) {
          KHR_ICD_TRACE("Accepting multi-string registry key type\n");
        } else {
          // Continue trying to get as much info on each adapter as possible.
          // It's too late to return FALSE and claim WDDM2_4 enumeration is not available here.
          continue;
        }
      }
      if (NT_SUCCESS(Status) && pQueryArgs->Status == D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW) {
        ULONG queryBufferSize = sizeof(D3DDDI_QUERYREGISTRY_INFO) + queryArgs.OutputValueSize;
        pQueryBuffer = (D3DDDI_QUERYREGISTRY_INFO*)malloc(queryBufferSize);
        if (pQueryBuffer == NULL) continue;
        memcpy(pQueryBuffer, &queryArgs, sizeof(D3DDDI_QUERYREGISTRY_INFO));
        queryAdapterInfo.pPrivateDriverData = pQueryBuffer;
        queryAdapterInfo.PrivateDriverDataSize = queryBufferSize;
        Status = D3DKMTQueryAdapterInfo(&queryAdapterInfo);
        pQueryArgs = pQueryBuffer;
      }
      if (NT_SUCCESS(Status) && pQueryArgs->Status == D3DDDI_QUERYREGISTRY_STATUS_SUCCESS) {
        wchar_t* pWchar = pQueryArgs->OutputString;
        memset(cszLibraryName, 0, sizeof(cszLibraryName));
        {
          size_t len = wcstombs(cszLibraryName, pWchar, sizeof(cszLibraryName));
          KHR_ICD_ASSERT(len == (sizeof(cszLibraryName) - 1));
          ret |= adapterAdd(cszLibraryName, pAdapterInfo[AdapterIndex].AdapterLuid);
        }
      } else if (Status == STATUS_INVALID_PARAMETER &&
                 pQueryArgs->Status == D3DDDI_QUERYREGISTRY_STATUS_FAIL) {
        free(pQueryBuffer);
        goto out;
      }
      free(pQueryBuffer);
    }
  out:
    free(pAdapterInfo);
  }

  FreeLibrary(h);

#endif
#endif
  return ret;
}
