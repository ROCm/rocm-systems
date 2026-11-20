// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/instruction.h"

namespace rocjitsu::amdgpu {

enum class ImageAddressKind { Sample, ResourceInfo, Bvh, Bvh64 };

struct ImageAddressForm {
  ImageAddressKind kind = ImageAddressKind::Sample;
  bool offset = false;
  bool bias = false;
  bool compare = false;
  bool lod = false; // LOD or clamp is packed with the coordinate body.
  bool gradients = false;
  bool g16 = false;
  bool sample_result = false; // Gather and MSAA always return four components.
};

/// Address operand of sampled images, resource queries and ray instructions.
/// Each of these has VADDR as its first source. The base encoding still owns
/// instruction decoding, modifier printing and the execution callback.
template <typename Encoding, bool gfx10, ImageAddressForm Form>
class ImageAddressInstruction : public Encoding {
  static constexpr bool ray =
      Form.kind == ImageAddressKind::Bvh || Form.kind == ImageAddressKind::Bvh64;
  static constexpr bool gfx12 = requires(typename Encoding::OpEncoding e) { e.vaddr0; };

  bool reads_address_inputs() const {
    if constexpr (Form.kind == ImageAddressKind::ResourceInfo && requires { this->inst_.dim; })
      return this->inst_.dmask != 0;
    return true;
  }

  bool scattered() const {
    if constexpr (requires { this->inst_.nsa; })
      return this->inst_.nsa != 0;
    return gfx12;
  }

  ImageAddressRegisters selectors() const {
    const auto &e = this->inst_;
    if constexpr (requires { e.vaddr0; })
      return gfx12_image_address_registers(e);
    else if constexpr (requires { e.nsa; })
      return mimg_image_address_registers(e.nsa, !gfx10, e.vaddr, this->raw_encoding());
    else
      return contiguous_image_address_registers(e.vaddr);
  }

  // BVH explicitly names the base of each vector (node, extent, origin,
  // direction and inverse direction). A16 combines the two direction groups.
  unsigned ray_group_width(unsigned group) const {
    return gfx11_bvh_group_widths(Form.kind == ImageAddressKind::Bvh64, this->inst_.a16)[group];
  }

  unsigned print_group_width(unsigned group, unsigned words) const {
    if (!scattered())
      return words;
    if constexpr (gfx10)
      return 1;
    if constexpr (ray)
      return ray_group_width(group);
    constexpr unsigned last = gfx12 && !requires { this->inst_.vaddr4; } ? 3 : 4;
    return group == last ? words - group : 1;
  }

  unsigned address(RegisterModifiers &m) const {
    const auto &e = this->inst_;
    m.image_address = this->src_operand(0);
    m.image_coordinates = selectors();
    if constexpr (ray) {
      const auto bases = m.image_coordinates;
      unsigned words = 0;
      for (unsigned group = 0; group < 5; ++group) {
        const auto width = ray_group_width(group);
        if (!gfx10 && scattered())
          for (unsigned i = 0; i < width; ++i)
            m.image_coordinates[words + i] = bases[group] + i;
        words += width;
      }
      m.image_prefix_words = words;
      return words;
    } else {
      m.image_a16 = e.a16;
      constexpr unsigned control_words = Form.offset + Form.bias + Form.compare;
      m.image_prefix_words = control_words;
      if constexpr (Form.kind == ImageAddressKind::ResourceInfo) {
        m.image_coordinate_count = 1; // Only the LOD, regardless of DIM.
      } else {
        m.image_coordinate_count = image_coordinate_count(e.dim) + Form.lod;
        const unsigned dimensions = Form.gradients ? image_derivative_component_count(e.dim) : 0;
        m.image_prefix_words += 2 * image_gradient_vector_words(dimensions, Form.g16);
        m.image_prefix_low16_words = image_prefix_low16_words(
            control_words, Form.g16 ? dimensions : 0, e.a16, Form.bias, Form.offset);
      }
      return m.image_prefix_words +
             (m.image_a16 ? (m.image_coordinate_count + 1) / 2 : m.image_coordinate_count);
    }
  }

  std::optional<RegisterRef> preserved_half() const {
    const unsigned components = image_result_components(this->inst_.dmask, !gfx10);
    if (this->inst_.d16 && (components & 1u))
      if (auto reg = this->dst_operand(0)->to_register_ref())
        return RegisterRef{reg->cls, static_cast<uint16_t>(reg->index + components / 2), 1};
    return std::nullopt;
  }

public:
  using Encoding::Encoding;
  void amdgpu_register_modifiers(RegisterModifiers &m) const override {
    Encoding::amdgpu_register_modifiers(m);
    address(m);
    if constexpr (Form.sample_result)
      if (this->inst_.d16)
        m.set_packed_image_result(image_result_components(this->inst_.dmask, !gfx10));
  }
  bool tracks_explicit_source(uint8_t index) const override {
    return index != 0 && reads_address_inputs() && Encoding::tracks_explicit_source(index);
  }
  void implicit_uses(RegisterSet &uses) const override {
    Encoding::implicit_uses(uses);
    if (!reads_address_inputs())
      return;
    RegisterModifiers m;
    const auto words = address(m);
    for (unsigned i = 0; i < words; ++i)
      if (m.image_coordinates[i] < 256)
        uses.expand({RegClass::VGPR, static_cast<uint16_t>(m.image_coordinates[i]), 1});
    if constexpr (Form.sample_result)
      if (const auto preserved = preserved_half())
        uses.expand(*preserved);
  }
  bool implicit_preservation_uses(RegisterSet &uses) const override {
    if constexpr (!Form.sample_result)
      return Encoding::implicit_preservation_uses(uses);
    if (const auto preserved = preserved_half()) {
      RegisterModifiers m;
      const auto words = address(m);
      for (unsigned i = 0; i < words; ++i)
        if (m.image_coordinates[i] == preserved->index)
          return true; // The address still reads this register.
      uses.expand(*preserved);
    }
    return true;
  }
  bool should_print_src_operand(uint8_t index) const override {
    return index == 0 || Encoding::should_print_src_operand(index);
  }
  void append_src_operand(std::string &out, uint8_t index) const override {
    if (index != 0) {
      Encoding::append_src_operand(out, index);
      return;
    }
    RegisterModifiers m;
    const auto words = address(m);
    const bool list = scattered() && !(gfx12 && words == 1);
    if (list)
      out += '[';
    for (unsigned i = 0, group = 0; i < words; ++group) {
      const unsigned width = print_group_width(group, words);
      if (i)
        out += ", ";
      const auto first = m.image_coordinates[i];
      out += width == 1
                 ? "v" + std::to_string(first)
                 : "v[" + std::to_string(first) + ":" + std::to_string(first + width - 1) + "]";
      i += width;
    }
    if (list)
      out += ']';
  }
};

} // namespace rocjitsu::amdgpu
