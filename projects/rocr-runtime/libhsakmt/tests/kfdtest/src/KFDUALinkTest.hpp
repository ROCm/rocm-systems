// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef __KFD_UALINK_TEST__H__
#define __KFD_UALINK_TEST__H__

#include <gtest/gtest.h>

#include "PM4Queue.hpp"
#include "KFDBaseComponentTest.hpp"
#include "KFDMultiNodeP2PTest.hpp"

class KFDUALinkTest : public KFDMultiNodeP2PTest,
                      public ::testing::WithParamInterface<uint64_t> {
   public:
      KFDUALinkTest() {}
      ~KFDUALinkTest() {}

   protected:
      virtual void SetUp();
      virtual void TearDown();

      int SetupExporter(int exporterGPU, amdgpu_bo_handle& expBo_handle,
                        amdgpu_va_handle& vahandle, uint64_t& expBoAddr,
                        void*& bo_cpu_addr, uint64_t size,
                        uint64_t (&ualink_handle)[2]);
      void CleanupExporter(amdgpu_bo_handle expBo_handle,
                           amdgpu_va_handle vaHandle,
                           uint64_t expBoAddr, uint64_t size);
      int SetupImporter(int importerGPU, amdgpu_bo_handle& imp_app_bo,
                        amdgpu_va_handle& va_handle, uint64_t (&ualink_handle)[2],
                        uint64_t size, uint64_t& imp_app_addr);
      void CleanupImporter(amdgpu_bo_handle imp_app_bo,
                           amdgpu_va_handle va_handle,
                           uint64_t imp_app_addr, uint64_t size);
      void CopyBuffer(int gpuNode, void *dst, void *src, uint64_t size);
      void AddressFaultImporter(uint64_t kSize, uint32_t exporterWritePattern);

      /* Handshake timeout for a step that fills, copies or checks size bytes
       * on the other side. Allows a floor of 10 MB/s on top of the base
       * handshake timeout.
       */
      static int DataTimeoutMs(uint64_t size) {
          return HANDSHAKE_TIMEOUT_MS + static_cast<int>(size / (10 * 1024));
      }

      /* The exporter revokes freed memory asynchronously, so the importer's
       * mapping can still be valid when EXPORT_MEMORY_FREED arrives.
       */
      static constexpr int REVOKE_SETTLE_MS = 100;

   protected:  // Members
      /* A BO set up by SetupExporter()/SetupImporter() and not yet released by
       * the matching Cleanup call. TearDown() frees it if an ASSERT ended the
       * test early.
       */
      struct UALinkBo {
          amdgpu_bo_handle bo = nullptr;
          amdgpu_va_handle va = nullptr;
          uint64_t addr = 0;
          uint64_t size = 0;
      };
      UALinkBo m_ExporterBo;
      UALinkBo m_ImporterBo;
      HSAKMT_STATUS m_ChildStatus;
};

#endif  // __KFD_UALINK_TEST__H__
