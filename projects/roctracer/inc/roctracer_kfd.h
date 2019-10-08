/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_ROCTRACER_KFD_H_
#define INC_ROCTRACER_KFD_H_
#include <iostream>
#include <mutex>

#include <hsa.h>

#include "roctracer.h"
#include "hsakmt.h"

namespace roctracer {
namespace kfd_support {
template <typename T>
struct output_streamer {
  inline static std::ostream& put(std::ostream& out, const T& v) { return out; }
};
template<>
struct output_streamer<bool> {
  inline static std::ostream& put(std::ostream& out, bool v) { out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint8_t> {
  inline static std::ostream& put(std::ostream& out, uint8_t v) { out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint16_t> {
  inline static std::ostream& put(std::ostream& out, uint16_t v) { out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint32_t> {
  inline static std::ostream& put(std::ostream& out, uint32_t v) { out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint64_t> {
  inline static std::ostream& put(std::ostream& out, uint64_t v) { out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }
};

template<>
struct output_streamer<bool*> {
  inline static std::ostream& put(std::ostream& out, bool* v) { out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint8_t*> {
  inline static std::ostream& put(std::ostream& out, uint8_t* v) { out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint16_t*> {
  inline static std::ostream& put(std::ostream& out, uint16_t* v) { out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint32_t*> {
  inline static std::ostream& put(std::ostream& out, uint32_t* v) { out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint64_t*> {
  inline static std::ostream& put(std::ostream& out, uint64_t* v) { out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }
};

template<>
struct output_streamer<hsa_queue_t*> {
  inline static std::ostream& put(std::ostream& out, hsa_queue_t* v) { out << "<queue " << v << ">"; return out; }
};
template<>
struct output_streamer<hsa_queue_t**> {
  inline static std::ostream& put(std::ostream& out, hsa_queue_t** v) { out << "<queue " << *v << ">"; return out; }
};
// begin ostream ops for KFD
template<>
struct output_streamer<HsaVersionInfo&> {
  inline static std::ostream& put(std::ostream& out, HsaVersionInfo& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.KernelInterfaceMajorVersion);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.KernelInterfaceMinorVersion);
    return out;
}
};
template<>
struct output_streamer<HsaSystemProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaSystemProperties& v) {
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumNodes);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.PlatformOem);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.PlatformId);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.PlatformRev);
    return out;
}
};
template<>
struct output_streamer<HSA_CAPABILITY&> {
  inline static std::ostream& put(std::ostream& out, HSA_CAPABILITY& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Value);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.HotPluggable);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.HSAMMUPresent);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.SharedWithGraphics);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.QueueSizePowerOfTwo);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.QueueSize32bit);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.QueueIdleEvent);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.VALimit);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.WatchPointsSupported);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.WatchPointsTotalBits);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.DoorbellType);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    return out;
}
};
template<>
struct output_streamer<HsaNodeProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaNodeProperties& v)
{
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumCPUCores);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumFComputeCores);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumMemoryBanks);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumCaches);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumIOLinks);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.CComputeIdLo);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.FComputeIdLo);
    roctracer::kfd_support::output_streamer<HSA_CAPABILITY>::put(out,v.Capability);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.MaxWavesPerSIMD);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.LDSSizeInKB);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.GDSSizeInKB);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.WaveFrontSize);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumShaderBanks);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumArrays);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumCUPerArray);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumSIMDPerCU);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.MaxSlotsScratchCU);
    roctracer::kfd_support::output_streamer<HSA_ENGINE_ID>::put(out,v.EngineId);
    roctracer::kfd_support::output_streamer<HSAuint16>::put(out,v.VendorId);
    roctracer::kfd_support::output_streamer<HSAuint16>::put(out,v.DeviceId);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.LocationId);
    roctracer::kfd_support::output_streamer<HSAuint64>::put(out,v.LocalMemSize);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.MaxEngineClockMhzFCompute);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.MaxEngineClockMhzCCompute);
    roctracer::kfd_support::output_streamer<HSAint32>::put(out,v.DrmRenderMinor);
    roctracer::kfd_support::output_streamer<HSAuint16[64]>::put(out,v.MarketingName);
    roctracer::kfd_support::output_streamer<HSAuint8[64]>::put(out,v.AMDName);
    roctracer::kfd_support::output_streamer<HSA_ENGINE_VERSION>::put(out,v.uCodeEngineVersions);
    roctracer::kfd_support::output_streamer<HSA_DEBUG_PROPERTIES>::put(out,v.DebugProperties);
    roctracer::kfd_support::output_streamer<HSAuint64>::put(out,v.HiveID);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumSdmaEngines);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumSdmaXgmiEngines);
    roctracer::kfd_support::output_streamer<HSAuint32>::put(out,v.NumGws);
    roctracer::kfd_support::output_streamer<HSAuint8[32]>::put(out,v.Reserved);
    return out;
}
};
template<>
struct output_streamer<HSA_MEMORYPROPERTY&> {
  inline static std::ostream& put(std::ostream& out, HSA_MEMORYPROPERTY& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MemoryProperty);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.HotPluggable);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NonVolatile);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    return out;
}
};
template<>
struct output_streamer<HsaMemoryProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaMemoryProperties& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.HeapType);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.SizeInBytes);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.SizeInBytesLow);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.SizeInBytesHigh);
    roctracer::kfd_support::output_streamer<HSA_MEMORYPROPERTY>::put(out,v.Flags);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Width);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MemoryClockMax);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.VirtualBaseAddress);
    return out;
}
};
template<>
struct output_streamer<HsaCacheType&> {
  inline static std::ostream& put(std::ostream& out, HsaCacheType& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Value);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Data);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Instruction);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.CPU);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.HSACU);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    return out;
}
};
template<>
struct output_streamer<HsaCacheProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaCacheProperties& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ProcessorIdLow);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheLevel);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheSize);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheLineSize);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheLinesPerTag);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheAssociativity);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CacheLatency);
    roctracer::kfd_support::output_streamer<HsaCacheType>::put(out,v.CacheType);
    roctracer::kfd_support::output_streamer<uint32_t[256]>::put(out,v.SiblingMap);
    return out;
}
};
template<>
struct output_streamer<HsaCComputeProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaCComputeProperties& v)
{
    roctracer::kfd_support::output_streamer<uint32_t[256]>::put(out,v.SiblingMap);
    return out;
}
};
template<>
struct output_streamer<HSA_LINKPROPERTY&> {
  inline static std::ostream& put(std::ostream& out, HSA_LINKPROPERTY& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.LinkProperty);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Override);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NonCoherent);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NoAtomics32bit);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NoAtomics64bit);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    return out;
}
};
template<>
struct output_streamer<HsaIoLinkProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaIoLinkProperties& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.IoLinkType);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.VersionMajor);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.VersionMinor);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NodeFrom);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NodeTo);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Weight);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MinimumLatency);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MaximumLatency);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MinimumBandwidth);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.MaximumBandwidth);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.RecTransferSize);
    roctracer::kfd_support::output_streamer<HSA_LINKPROPERTY&>::put(out,v.Flags);
    return out;
}
};
template<>
struct output_streamer<HsaMemFlags&> {
  inline static std::ostream& put(std::ostream& out, HsaMemFlags& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NonPaged);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.CachePolicy);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.ReadOnly);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.PageSize);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.HostAccess);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.NoSubstitute);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.GDSMemory);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Scratch);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.AtomicAccessFull);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.AtomicAccessPartial);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.ExecuteAccess);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Value);
    return out;
}
};
template<>
struct output_streamer<HsaQueueResource&> {
  inline static std::ostream& put(std::ostream& out, HsaQueueResource& v)
{
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.QueueId);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,*(v.Queue_DoorBell));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,*(v.Queue_DoorBell_aql));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.QueueDoorBell);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,*(v.Queue_write_ptr));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,*(v.Queue_write_ptr_aql));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.QueueWptrValue);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,*(v.Queue_read_ptr));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,*(v.Queue_read_ptr_aql));
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.QueueRptrValue);
    return out;
}
};
template<>
struct output_streamer<HsaQueueReport&> {
  inline static std::ostream& put(std::ostream& out, HsaQueueReport& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.VMID);
    out << "<void *" << v.QueueAddress << ">";
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.QueueSize);
    return out;
}
};
template<>
struct output_streamer<HsaDbgWaveMsgAMDGen2&> {
  inline static std::ostream& put(std::ostream& out, HsaDbgWaveMsgAMDGen2& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out, v.Value);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out, v.Reserved2);
    return out;
}
};
template<>
struct output_streamer<HsaDbgWaveMessageAMD&> {
  inline static std::ostream& put(std::ostream& out, HsaDbgWaveMessageAMD& v)
{
    roctracer::kfd_support::output_streamer<HsaDbgWaveMsgAMDGen2>::put(out,v.WaveMsgInfoGen2);
    return out;
}
};
template<>
struct output_streamer<HsaDbgWaveMessage&> {
  inline static std::ostream& put(std::ostream& out, HsaDbgWaveMessage& v)
{
    out << "<void* " << v.MemoryVA << ">";
    roctracer::kfd_support::output_streamer<HsaDbgWaveMessageAMD>::put(out,v.DbgWaveMsg);
    return out;
}
};
template<>
struct output_streamer<HsaSyncVar&> {
  inline static std::ostream& put(std::ostream& out, HsaSyncVar& v)
{
    out << "<void * " << v.SyncVar.UserData << ">";
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.SyncVar.UserDataPtrValue);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.SyncVarSize);
    return out;
}
};
template<>
struct output_streamer<HsaNodeChange&> {
  inline static std::ostream& put(std::ostream& out, HsaNodeChange& v)
{
  roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Flags);
  return out;
}
};
template<>
struct output_streamer<HsaDeviceStateChange&> {
  inline static std::ostream& put(std::ostream& out, HsaDeviceStateChange& v)
{
  roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NodeId);
  roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Device);
  roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Flags);
  return out;
}
};
template<>
struct output_streamer<HsaAccessAttributeFailure&> {
  inline static std::ostream& put(std::ostream& out, HsaAccessAttributeFailure& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NotPresent);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ReadOnly);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NoExecute);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.GpuAccess);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ECC);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Reserved);
    return out;
}
};
template<>
struct output_streamer<HsaMemoryAccessFault&> {
  inline static std::ostream& put(std::ostream& out, HsaMemoryAccessFault& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NodeId);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.VirtualAddress);
    roctracer::kfd_support::output_streamer<HsaAccessAttributeFailure>::put(out,v. Failure);
    roctracer::kfd_support::output_streamer<int>::put(out,v.Flags);
    return out;
}
};
template<>
struct output_streamer<HsaEventData&> {
  inline static std::ostream& put(std::ostream& out, HsaEventData& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.EventType);
    roctracer::kfd_support::output_streamer<HsaSyncVar>::put(out,v.EventData.SyncVar);
    roctracer::kfd_support::output_streamer<HsaNodeChange>::put(out,v.EventData.NodeChangeState);
    roctracer::kfd_support::output_streamer<HsaDeviceStateChange>::put(out,v.EventData.DeviceState);
    roctracer::kfd_support::output_streamer<HsaMemoryAccessFault>::put(out,v.EventData.MemoryAccessFault);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.HWData1);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.HWData2);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.HWData3);
    return out;
}
};
template<>
struct output_streamer<HsaEventDescriptor&> {
  inline static std::ostream& put(std::ostream& out, HsaEventDescriptor& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.EventType);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NodeId);
    roctracer::kfd_support::output_streamer<HsaSyncVar>::put(out,v.SyncVar);
    return out;
}
};
template<>
struct output_streamer<HsaEvent&> {
  inline static std::ostream& put(std::ostream& out, HsaEvent& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.EventId);
    roctracer::kfd_support::output_streamer<HsaEventData>::put(out,v.EventData);
    return out;
}
};
template<>
struct output_streamer<HsaClockCounters&> {
  inline static std::ostream& put(std::ostream& out, HsaClockCounters& v)
{
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.GPUClockCounter);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.CPUClockCounter);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.SystemClockCounter);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.SystemClockFrequencyHz);
    return out;
}
};
template<>
struct output_streamer<HSA_UUID&> {
  inline static std::ostream& put(std::ostream& out, HSA_UUID& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Data1);
    roctracer::kfd_support::output_streamer<uint16_t>::put(out,v.Data2);
    roctracer::kfd_support::output_streamer<uint16_t>::put(out,v.Data3);
    roctracer::kfd_support::output_streamer<uint8_t[8]>::put(out,v.Data4);
    return out;
}
};
template<>
struct output_streamer<HsaCounterFlags&> {
  inline static std::ostream& put(std::ostream& out, HsaCounterFlags& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Global);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Resettable);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.ReadOnly);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Stream);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.ui32.Reserved);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out, v.Value);
    return out;
}
};
template<>
struct output_streamer<HsaCounter&> {
  inline static std::ostream& put(std::ostream& out, HsaCounter& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.Type);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.CounterId);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.CounterSizeInBits);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.CounterMask);
    roctracer::kfd_support::output_streamer<HsaCounterFlags>::put(out,v.Flags);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.BlockIndex);
    return out;
}
};
template<>
struct output_streamer<HsaCounterBlockProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaCounterBlockProperties& v)
{
    roctracer::kfd_support::output_streamer<HSA_UUID>::put(out,v.BlockId);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumCounters);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumConcurrent);
    roctracer::kfd_support::output_streamer<HsaCounter>::put(out,v.Counters[1]);
    return out;
}
};
template<>
struct output_streamer<HsaCounterProperties&> {
  inline static std::ostream& put(std::ostream& out, HsaCounterProperties& v)
{
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumBlocks);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumConcurrent);
    roctracer::kfd_support::output_streamer<HsaCounterBlockProperties>::put(out,v.Blocks[1]);
    return out;
}
};
template<>
struct output_streamer<HsaPmcTraceRoot&> {
  inline static std::ostream& put(std::ostream& out, HsaPmcTraceRoot& v)
{
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.TraceBufferMinSizeBytes);
    roctracer::kfd_support::output_streamer<uint32_t>::put(out,v.NumberOfPasses);
    roctracer::kfd_support::output_streamer<uint64_t>::put(out,v.TraceId);
    return out;
}
};
// end ostream ops for KFD
};};

#include <inc/kfd_prof_str.h>

#endif // INC_ROCTRACER_KFD_H_
