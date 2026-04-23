/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_graph_internal.hpp"
#include "hip_global.hpp"

#include "device/devprogram.hpp"
#include "device/devkernel.hpp"
#include "platform/kernel.hpp"
#include "platform/program.hpp"

#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// In-memory dump structure (full definition; hip_runtime_api.h has the typedef)
// ---------------------------------------------------------------------------
struct hipExtGraphExecDump_st {
  std::string jsonContent;
  // Insertion-ordered list of hashes (for stable indexed access)
  std::vector<std::string> codeObjectOrder;
  // Hash -> binary blob (for O(1) dedup lookup)
  std::unordered_map<std::string, std::vector<uint8_t>> codeObjects;
};

namespace hip {

// ---------------------------------------------------------------------------
// JSON helper — lightweight JSON writer (no external dependency)
// ---------------------------------------------------------------------------
class JsonWriter {
 public:
  JsonWriter() : indent_(0), needsComma_(false) {}

  void beginObject() {
    maybeComma();
    out_ << "{\n";
    indent_ += 2;
    needsComma_ = false;
  }

  void endObject() {
    out_ << "\n";
    indent_ -= 2;
    writeIndent();
    out_ << "}";
    needsComma_ = true;
  }

  void beginArray(const char* key) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": [\n";
    indent_ += 2;
    needsComma_ = false;
  }

  void beginArrayItem() {
    maybeComma();
    writeIndent();
    out_ << "{\n";
    indent_ += 2;
    needsComma_ = false;
  }

  void endArrayItem() {
    out_ << "\n";
    indent_ -= 2;
    writeIndent();
    out_ << "}";
    needsComma_ = true;
  }

  void endArray() {
    out_ << "\n";
    indent_ -= 2;
    writeIndent();
    out_ << "]";
    needsComma_ = true;
  }

  void writeInt(const char* key, int64_t value) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": " << value;
    needsComma_ = true;
  }

  void writeUint(const char* key, uint64_t value) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": " << value;
    needsComma_ = true;
  }

  void writeString(const char* key, const std::string& value) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": \"" << escapeJson(value) << "\"";
    needsComma_ = true;
  }

  void writeBool(const char* key, bool value) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": " << (value ? "true" : "false");
    needsComma_ = true;
  }

  // Write an inline object for dim3
  void writeDim3(const char* key, uint32_t x, uint32_t y, uint32_t z) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": { \"x\": " << x << ", \"y\": " << y << ", \"z\": " << z << " }";
    needsComma_ = true;
  }

  // Write an array of ints inline
  void writeIntArray(const char* key, const std::vector<int>& values) {
    maybeComma();
    writeIndent();
    out_ << "\"" << key << "\": [";
    for (size_t i = 0; i < values.size(); i++) {
      if (i > 0) out_ << ", ";
      out_ << values[i];
    }
    out_ << "]";
    needsComma_ = true;
  }

  std::string str() const { return out_.str(); }

 private:
  std::ostringstream out_;
  int indent_;
  bool needsComma_;

  void writeIndent() {
    for (int i = 0; i < indent_; i++) out_ << ' ';
  }

  void maybeComma() {
    if (needsComma_) out_ << ",\n";
  }

  static std::string escapeJson(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
      switch (c) {
        case '"':  result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n";  break;
        case '\r': result += "\\r";  break;
        case '\t': result += "\\t";  break;
        default:   result += c;      break;
      }
    }
    return result;
  }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string nodeTypeToString(hipGraphNodeType type) {
  switch (type) {
    case hipGraphNodeTypeKernel:             return "kernel";
    case hipGraphNodeTypeMemcpy:             return "memcpy";
    case hipGraphNodeTypeMemset:             return "memset";
    case hipGraphNodeTypeHost:               return "host";
    case hipGraphNodeTypeGraph:              return "child_graph";
    case hipGraphNodeTypeEmpty:              return "empty";
    case hipGraphNodeTypeWaitEvent:          return "wait_event";
    case hipGraphNodeTypeEventRecord:        return "event_record";
    case hipGraphNodeTypeExtSemaphoreSignal: return "ext_semaphore_signal";
    case hipGraphNodeTypeExtSemaphoreWait:   return "ext_semaphore_wait";
    case hipGraphNodeTypeMemAlloc:           return "mem_alloc";
    case hipGraphNodeTypeMemFree:            return "mem_free";
    default:                                 return "unknown";
  }
}

static std::string bytesToHex(const void* data, size_t size) {
  std::ostringstream oss;
  oss << "0x";
  // Little-endian: dump bytes from high to low for natural numeric representation
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = size; i > 0; i--) {
    oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(bytes[i - 1]);
  }
  return oss.str();
}

// Simple hash for code object deduplication (FNV-1a 64-bit)
static std::string hashCodeObject(const void* data, size_t size) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; i++) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL;
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

// ---------------------------------------------------------------------------
// GraphExec::BuildDump() — build in-memory dump, no file I/O
// ---------------------------------------------------------------------------

hipExtGraphExecDump_t GraphExec::BuildDump(unsigned int flags) const {
  hipExtGraphExecDump_st* dump = new (std::nothrow) hipExtGraphExecDump_st();
  if (dump == nullptr) {
    return nullptr;
  }

  JsonWriter json;
  json.beginObject();
  json.writeInt("format_version", 1);

  // Count kernel nodes
  size_t kernelNodeCount = 0;
  for (const auto& node : topoOrder_) {
    if (node->GetType() == hipGraphNodeTypeKernel) {
      kernelNodeCount++;
    }
  }

  json.writeUint("node_count", topoOrder_.size());
  json.writeUint("kernel_node_count", kernelNodeCount);

  json.beginArray("nodes");

  for (const auto& node : topoOrder_) {
    json.beginArrayItem();
    json.writeInt("id", node->GetID());
    json.writeString("type", nodeTypeToString(node->GetType()));
    json.writeBool("enabled", node->GetEnabled() != 0);

    // Kernel node: dump dispatch dims, arguments, code objects
    if (node->GetType() == hipGraphNodeTypeKernel) {
      GraphKernelNode* kernelNode = static_cast<GraphKernelNode*>(node);

      // Get kernel params via public accessor
      hipKernelNodeParams kparams = {};
      kernelNode->GetParams(&kparams);

      // Resolve the kernel function via public static method
      hipFunction_t func = GraphKernelNode::getFunc(kparams, node->GetDeviceId());

      if (func) {
        amd::Kernel* kernel = hip::asKernel(func);
        // Trim trailing whitespace and null bytes from kernel name
        const std::string& rawName = kernel->name();
        size_t len = rawName.size();
        while (len > 0 && (rawName[len - 1] == ' ' || rawName[len - 1] == '\0')) len--;
        if (len == rawName.size()) {
          json.writeString("kernel_name", rawName);
        } else {
          json.writeString("kernel_name", rawName.substr(0, len));
        }

        // Dispatch dimensions
        if (flags & hipExtGraphExecDumpFlagsDispatch) {
          json.writeDim3("gridDim",  kparams.gridDim.x,  kparams.gridDim.y,  kparams.gridDim.z);
          json.writeDim3("blockDim", kparams.blockDim.x, kparams.blockDim.y, kparams.blockDim.z);
          json.writeUint("sharedMemBytes", kparams.sharedMemBytes);
        }

        // Kernel arguments
        if (flags & hipExtGraphExecDumpFlagsKernelArgs) {
          const amd::KernelSignature& sig = kernel->signature();
          json.beginArray("arguments");

          for (uint32_t i = 0; i < sig.numParameters(); i++) {
            const auto& desc = sig.at(i);

            json.beginArrayItem();
            json.writeUint("index", i);
            json.writeString("name", desc.name_);
            json.writeString("typeName", desc.typeName_);
            json.writeUint("size", desc.size_);
            json.writeUint("offset", desc.offset_);

            bool isPointer = (desc.type_ == T_POINTER) ||
                             (desc.info_.oclObject_ ==
                              amd::KernelParameterDescriptor::MemoryObject);
            bool isImage = (desc.info_.oclObject_ ==
                            amd::KernelParameterDescriptor::ImageObject);
            bool isSampler = (desc.info_.oclObject_ ==
                              amd::KernelParameterDescriptor::SamplerObject);

            if (isPointer) {
              json.writeString("kind", "pointer");
              json.writeString("value", "PTR");
            } else if (isImage) {
              json.writeString("kind", "image");
              json.writeString("value", "IMG");
            } else if (isSampler) {
              json.writeString("kind", "sampler");
              json.writeString("value", "SAMPLER");
            } else {
              json.writeString("kind", "value");
              // Read actual value from captured kernel params
              std::string hexValue = "unknown";
              if (kparams.kernelParams != nullptr && kparams.kernelParams[i] != nullptr) {
                hexValue = bytesToHex(kparams.kernelParams[i], desc.size_);
              } else if (kparams.extra != nullptr && kparams.extra[1] != nullptr) {
                // Extra path: flat buffer at extra[1], read at desc.offset_
                const uint8_t* extraBuf = static_cast<const uint8_t*>(kparams.extra[1]);
                hexValue = bytesToHex(extraBuf + desc.offset_, desc.size_);
              }
              json.writeString("value", hexValue);
            }

            json.endArrayItem();
          }

          json.endArray();  // arguments
        }

        // Code objects
        if (flags & hipExtGraphExecDumpFlagsCodeObjects) {
          amd::Program& prog = kernel->program();
          const amd::Device& device = *g_devices[node->GetDeviceId()]->devices()[0];
          device::Program* devProg = prog.getDeviceProgram(device);

          if (devProg != nullptr) {
            auto bin = devProg->binary();
            const void* image = bin.first;
            size_t imageSize = bin.second;

            if (image != nullptr && imageSize > 0) {
              std::string hash = hashCodeObject(image, imageSize);

              // Store binary in-memory, deduplicated by hash
              if (dump->codeObjects.find(hash) == dump->codeObjects.end()) {
                const uint8_t* imgBytes = static_cast<const uint8_t*>(image);
                dump->codeObjects[hash] = std::vector<uint8_t>(imgBytes, imgBytes + imageSize);
                dump->codeObjectOrder.push_back(hash);
              }

              // JSON references the hash so callers know which binary to associate
              json.writeString("code_object_hash", hash);
              json.writeUint("code_object_size", imageSize);
            }
          }
        }
      } else {
        json.writeString("kernel_name", "unresolved");
      }
    }

    // Dependencies
    if (flags & hipExtGraphExecDumpFlagsDeps) {
      const auto& deps = node->GetDependencies();
      std::vector<int> depIds;
      depIds.reserve(deps.size());
      for (const auto& dep : deps) {
        depIds.push_back(dep->GetID());
      }
      json.writeIntArray("dependencies", depIds);

      const auto& edges = node->GetEdges();
      std::vector<int> edgeIds;
      edgeIds.reserve(edges.size());
      for (const auto& edge : edges) {
        edgeIds.push_back(edge->GetID());
      }
      json.writeIntArray("dependents", edgeIds);
    }

    json.endArrayItem();
  }

  json.endArray();  // nodes
  // Write code object count
  json.writeUint("code_object_count", dump->codeObjects.size());
  json.endObject();

  dump->jsonContent = json.str();
  return dump;
}

}  // namespace hip

// ---------------------------------------------------------------------------
// Public HIP APIs
// ---------------------------------------------------------------------------

hipError_t hipExtGraphExecDumpCreate(hipGraphExec_t graphExec, unsigned int flags,
                                     hipExtGraphExecDump_t* dumpOut) {
  HIP_INIT_API(hipExtGraphExecDumpCreate, graphExec, flags, dumpOut);

  if (graphExec == nullptr || dumpOut == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphExec* exec = reinterpret_cast<hip::GraphExec*>(graphExec);
  if (!hip::GraphExec::isGraphExecValid(exec)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *dumpOut = exec->BuildDump(flags);
  HIP_RETURN(*dumpOut != nullptr ? hipSuccess : hipErrorOutOfMemory);
}

hipError_t hipExtGraphExecDumpGet(hipExtGraphExecDump_t dump,
                                   hipExtGraphExecDumpQuery_t query,
                                   size_t index,
                                   hipExtGraphExecDumpResult_t* result) {
  HIP_INIT_API(hipExtGraphExecDumpGet, dump, query, index, result);

  if (dump == nullptr || result == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  switch (query) {
    case hipExtGraphExecDumpQueryJson:
      result->json.ptr = dump->jsonContent.c_str();
      result->json.len = dump->jsonContent.size();
      break;

    case hipExtGraphExecDumpQueryCodeObjectCount:
      result->count = dump->codeObjectOrder.size();
      break;

    case hipExtGraphExecDumpQueryCodeObject:
      if (index >= dump->codeObjectOrder.size()) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      {
        const std::string& h = dump->codeObjectOrder[index];
        const auto& blob = dump->codeObjects.at(h);
        result->codeObject.hash = h.c_str();
        result->codeObject.data = blob.data();
        result->codeObject.size = blob.size();
      }
      break;

    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipExtGraphExecDumpDestroy(hipExtGraphExecDump_t dump) {
  HIP_INIT_API(hipExtGraphExecDumpDestroy, dump);

  if (dump == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  delete dump;
  HIP_RETURN(hipSuccess);
}
