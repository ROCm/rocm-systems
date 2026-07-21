// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/ip_discovery_blob.h"
#include "rocjitsu/vfu/mmio_registers.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace rocjitsu::vfu {

namespace {

// ---------------------------------------------------------------------------
// Constants matching amdgpu_discovery.h / discovery.h
// ---------------------------------------------------------------------------
constexpr uint32_t kDiscoveryTmrSize   = 10 * 1024; // DISCOVERY_TMR_SIZE
constexpr uint32_t kBinarySignature    = 0x28211407;
constexpr uint32_t kDiscoveryTableSig  = 0x53445049; // "IPDS"
constexpr uint32_t kHarvestTableSig    = 0x56524148; // "HARV"
constexpr uint16_t kDiscoveryVersion   = 2;
constexpr int      kTotalTables        = 6;

// Hardware IDs (from soc15_hw_ip.h in the DKMS source)
constexpr uint16_t kHwIdMp1    =  1;
constexpr uint16_t kHwIdThm    =  3;
constexpr uint16_t kHwIdSmuio  =  4;
constexpr uint16_t kHwIdGc     = 11;
constexpr uint16_t kHwIdMmhub  = 34;
constexpr uint16_t kHwIdAthub  = 35;
constexpr uint16_t kHwIdOsssys = 40;
constexpr uint16_t kHwIdHdp    = 41;
constexpr uint16_t kHwIdSdma0  = 42;  // instances 0-4 for MI350P
constexpr uint16_t kHwIdDf     = 46;
constexpr uint16_t kHwIdUmc    = 196; // actually 12 for VCN/UMC context - see below
constexpr uint16_t kHwIdXgmi   = 200;
constexpr uint16_t kHwIdNbif   = 192;
constexpr uint16_t kHwIdMp0    = 202;
constexpr uint16_t kHwIdPcie   = 187;
constexpr uint16_t kHwIdLsdma  = 91;  // LSDMA_HWID

// ---------------------------------------------------------------------------
// Low-level serialisation helpers
// ---------------------------------------------------------------------------

struct Writer {
  std::vector<uint8_t> &buf;
  size_t pos = 0;

  void seek(size_t p) { assert(p <= buf.size()); pos = p; }
  size_t tell() const { return pos; }

  void write_u8(uint8_t v)  { assert(pos + 1 <= buf.size()); buf[pos++] = v; }
  void write_u16(uint16_t v) {
    assert(pos + 2 <= buf.size());
    buf[pos] = v & 0xff; buf[pos+1] = v >> 8; pos += 2;
  }
  void write_u32(uint32_t v) {
    assert(pos + 4 <= buf.size());
    buf[pos]   =  v        & 0xff;
    buf[pos+1] = (v >>  8) & 0xff;
    buf[pos+2] = (v >> 16) & 0xff;
    buf[pos+3] = (v >> 24) & 0xff;
    pos += 4;
  }
  void pad(size_t n) { assert(pos + n <= buf.size()); pos += n; }
};

uint16_t checksum_range(const std::vector<uint8_t> &buf, size_t off, size_t len) {
  uint16_t s = 0;
  for (size_t i = 0; i < len; ++i)
    s += buf[off + i];
  return s;
}

// ---------------------------------------------------------------------------
// Emit one 'struct ip' (v2 format, 32-bit base addresses)
//   hw_id u16, num_instance u8, num_base_address u8,
//   major u8, minor u8, revision u8, harvest:4|reserved:4 u8,
//   base_address[num_base] u32[]
// ---------------------------------------------------------------------------
void emit_ip(Writer &w, uint16_t hw_id, uint8_t instance,
             uint8_t major, uint8_t minor, uint8_t revision,
             const std::initializer_list<uint32_t> bases) {
  w.write_u16(hw_id);
  w.write_u8(instance);
  w.write_u8(static_cast<uint8_t>(bases.size()));
  w.write_u8(major);
  w.write_u8(minor);
  w.write_u8(revision);
  w.write_u8(0);  // harvest=0, reserved=0
  for (uint32_t b : bases)
    w.write_u32(b);
}

} // namespace

// ---------------------------------------------------------------------------
// build_gfx944_discovery_blob
//
// IP versions are taken from the GFX9.4.3 (MI300X) aldebaran reference and
// bumped to .4 for GC/MMHUB/SDMA where amdgpu_discovery has explicit
// IP_VERSION(9, 4, 4) cases. The key requirement for the driver is that
// GC_HWIP reports version 9.4.4, which triggers:
//   - aqua_vanjaram_init_soc_config()
//   - gfx_v9_4_3_ip_block
//   - gmc_v9_0_ip_block
//   - vega10_common_ip_block
// ---------------------------------------------------------------------------
std::vector<uint8_t> build_gfx944_discovery_blob() {
  std::vector<uint8_t> blob(kDiscoveryTmrSize, 0);
  Writer w{blob};

  // ----------------------------------------------------------------
  // 1. binary_header  (offset 0, size = 4+2+2+2+2 + 6*8 = 60 bytes)
  //    Layout:
  //      uint32 binary_signature
  //      uint16 version_major
  //      uint16 version_minor
  //      uint16 binary_checksum   <-- filled in last
  //      uint16 binary_size
  //      table_info[6]: { uint16 offset, uint16 checksum, uint16 size, uint16 padding }
  // ----------------------------------------------------------------
  static constexpr size_t kBinaryHdrSize  = 12 + kTotalTables * 8; // 60
  static constexpr size_t kIpDiscoveryOff = kBinaryHdrSize;         // 60

  // Write binary_header fields (leave checksum as 0 for now)
  w.write_u32(kBinarySignature);
  w.write_u16(1); // version_major
  w.write_u16(0); // version_minor
  size_t binary_checksum_pos = w.tell();
  w.write_u16(0); // binary_checksum placeholder
  w.write_u16(0); // binary_size placeholder — filled below

  // table_info[6] — all zeroed first, then IP_DISCOVERY and HARVEST filled in
  size_t table_info_pos = w.tell();
  w.pad(kTotalTables * 8); // 48 bytes

  // ----------------------------------------------------------------
  // 2. ip_discovery_header at offset 60
  //    Layout:
  //      uint32 signature
  //      uint16 version
  //      uint16 size          <-- filled in after we know ip table size
  //      uint32 id
  //      uint16 num_dies
  //      die_info[16]: { uint16 die_id, uint16 die_offset }
  //      uint16 padding[1]
  //    Total: 4+2+2+4+2 + 16*4 + 2 = 80 bytes
  // ----------------------------------------------------------------
  static constexpr size_t kIpHdrInnerSize = 4+2+2+4+2 + 16*4 + 2; // 80
  size_t ip_hdr_pos = w.tell(); // == kIpDiscoveryOff
  w.write_u32(kDiscoveryTableSig);
  w.write_u16(kDiscoveryVersion);
  size_t ip_size_pos = w.tell();
  w.write_u16(0); // size placeholder
  w.write_u32(0x00000001); // id (arbitrary, non-zero)
  w.write_u16(1);           // num_dies = 1

  // die_info[0]: die_id=1, die_offset = (offset of die_header FROM ip_hdr_pos)
  // die_header starts immediately after the die_info array + padding
  uint16_t die_hdr_rel_off = static_cast<uint16_t>(kIpHdrInnerSize);
  w.write_u16(1);                // die_id
  w.write_u16(die_hdr_rel_off);  // die_offset (relative to ip_discovery_header start)
  // die_info[1..15]: zeroed
  for (int i = 1; i < 16; ++i) { w.write_u16(0); w.write_u16(0); }
  w.write_u16(0); // padding

  // ----------------------------------------------------------------
  // 3. die_header + ip[] array
  //    die_header: uint16 die_id, uint16 num_ips
  // ----------------------------------------------------------------
  size_t die_hdr_pos = w.tell(); // == ip_hdr_pos + kIpHdrInnerSize
  w.write_u16(1); // die_id = 1
  size_t num_ips_pos = w.tell();
  w.write_u16(0); // num_ips placeholder

  // Emit IP blocks.  Each entry: 8 bytes header + 4*num_base bytes.
  // The GC IP version (9, 4, 4) is the key one that triggers the correct
  // driver code paths. Other versions match GFX9.4.3 (aldebaran reference).
  size_t ip_array_start = w.tell();
  int num_ips = 0;

  // GC (shader engine) — version 9.4.4 → activates gfx_v9_4_3_ip_block
  emit_ip(w, kHwIdGc,    0, 9,  4, 4, {0x2000, 0xa000, 0x2402c00});  ++num_ips;

  // MMHUB — version 9.4.4
  emit_ip(w, kHwIdMmhub, 0, 9,  4, 4, {0x1a000, 0x2408800});          ++num_ips;

  // ATHUB — version 9.4.2 (unchanged from aldebaran)
  emit_ip(w, kHwIdAthub, 0, 9,  4, 2, {0xc20, 0x2408c00});            ++num_ips;

  // OSSSYS (IH) — version 4.4.0
  emit_ip(w, kHwIdOsssys,0, 4,  4, 0, {0x10a0, 0x240a000});           ++num_ips;

  // HDP — version 4.4.0
  emit_ip(w, kHwIdHdp,   0, 4,  4, 0, {0xf20, 0x240a400});            ++num_ips;

  // DF — version 3.6.2
  emit_ip(w, kHwIdDf,    0, 3,  6, 2, {0x7000, 0x240b800, 0x7c00000}); ++num_ips;

  // NBIF — version 6.3.1 (MI300X reference)
  emit_ip(w, kHwIdNbif,  0, 6,  3, 1, {0x0, 0x10000, 0x2410000});      ++num_ips;

  // MP0 — version 13.0.14 (GFX9.4.4 PSP version)
  emit_ip(w, kHwIdMp0,   0, 13, 0, 14,{0x15200, 0x243d000});           ++num_ips;

  // MP1 (SMU) — version 13.0.14
  emit_ip(w, kHwIdMp1,   0, 13, 0, 14,{0x16000, 0xdc0000, 0xe00000, 0x243fc00}); ++num_ips;

  // THM — version 13.0.14
  emit_ip(w, kHwIdThm,   0, 13, 0, 14,{0x16600, 0x2400c00});           ++num_ips;

  // SMUIO — version 13.0.14
  emit_ip(w, kHwIdSmuio, 0, 13, 0, 14,{0x16800, 0x16a00, 0x2401000, 0x3440000}); ++num_ips;

  // SDMA — 5 instances (SDMA 4.4.5 for MI350P)
  for (uint8_t i = 0; i < 5; ++i) {
    uint32_t base0 = 0x1260 + i * 0x600;
    uint32_t base1 = 0x12540 + i * 0x20;
    uint32_t base2 = 0x40a800 + i * 0x400;
    emit_ip(w, kHwIdSdma0, i, 4, 4, 5, {base0, base1, base2});
    ++num_ips;
  }

  // XGMI — version 6.1.0 (no external links on MI350P but driver checks for it)
  emit_ip(w, kHwIdXgmi,  0, 6,  1, 0, {0x78000});                      ++num_ips;

  // PCIE — version 11.0.5
  emit_ip(w, kHwIdPcie,  0, 11, 0, 5, {0x11180000});                   ++num_ips;

  // UMC — version 12.0.0 (GFX9.4.4 memory controller) -- hw_id 196
  emit_ip(w, kHwIdUmc,   0, 12, 0, 0, {0x50000, 0x52000, 0x54000, 0x56000}); ++num_ips;

  // Fill in num_ips now that we know it
  blob[num_ips_pos]     = static_cast<uint8_t>(num_ips & 0xff);
  blob[num_ips_pos + 1] = static_cast<uint8_t>(num_ips >> 8);

  size_t ip_table_end = w.tell();

  // ip_discovery_header.size = total bytes from ip_hdr_pos to ip_table_end
  uint16_t ip_table_size = static_cast<uint16_t>(ip_table_end - ip_hdr_pos);
  blob[ip_size_pos]     = ip_table_size & 0xff;
  blob[ip_size_pos + 1] = ip_table_size >> 8;

  // ----------------------------------------------------------------
  // 4. Harvest info table (empty — no harvested blocks on MI350P base config)
  //    harvest_info_header: uint32 signature, uint32 version
  //    harvest_table: harvest_info[32] — all zero
  // ----------------------------------------------------------------
  size_t harvest_off = w.tell();
  w.write_u32(kHarvestTableSig);
  w.write_u32(1); // version
  // 32 * (uint16 hw_id, uint8 number_instance, uint8 reserved)
  for (int i = 0; i < 32; ++i) {
    w.write_u16(0); w.write_u8(0); w.write_u8(0);
  }
  size_t harvest_end = w.tell();

  // ----------------------------------------------------------------
  // 5. Fill in binary_header.binary_size and table_info entries
  // ----------------------------------------------------------------
  uint16_t binary_size = static_cast<uint16_t>(harvest_end);
  blob[binary_checksum_pos + 2] = binary_size & 0xff;
  blob[binary_checksum_pos + 3] = binary_size >> 8;

  // table_info[IP_DISCOVERY] (index 0): offset=60, checksum, size
  uint16_t ip_size16   = ip_table_size;
  uint16_t ip_checksum = checksum_range(blob, kIpDiscoveryOff, ip_table_size);
  size_t t0 = table_info_pos + 0 * 8;
  blob[t0]   = kIpDiscoveryOff & 0xff;
  blob[t0+1] = kIpDiscoveryOff >> 8;
  blob[t0+2] = ip_checksum & 0xff;
  blob[t0+3] = ip_checksum >> 8;
  blob[t0+4] = ip_size16 & 0xff;
  blob[t0+5] = ip_size16 >> 8;

  // table_info[HARVEST_INFO] (index 2): offset, checksum, size
  uint16_t harv_size     = static_cast<uint16_t>(harvest_end - harvest_off);
  uint16_t harv_off16    = static_cast<uint16_t>(harvest_off);
  uint16_t harv_checksum = checksum_range(blob, harvest_off, harv_size);
  size_t t2 = table_info_pos + 2 * 8;
  blob[t2]   = harv_off16 & 0xff;
  blob[t2+1] = harv_off16 >> 8;
  blob[t2+2] = harv_checksum & 0xff;
  blob[t2+3] = harv_checksum >> 8;
  blob[t2+4] = harv_size & 0xff;
  blob[t2+5] = harv_size >> 8;

  // binary_checksum: sum of all bytes from after the checksum field to end
  size_t chk_start = binary_checksum_pos + 4; // after binary_checksum + binary_size
  uint16_t bin_checksum = checksum_range(blob, chk_start, binary_size - chk_start);
  blob[binary_checksum_pos]     = bin_checksum & 0xff;
  blob[binary_checksum_pos + 1] = bin_checksum >> 8;

  return blob;
}

} // namespace rocjitsu::vfu
