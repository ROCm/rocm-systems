//! Strongly-typed builtin [`AgentDef`]s.
//!
//! Each function returns the same value the corresponding legacy
//! `agents/<NAME>.json` produced once parsed into an [`AgentDef`].
//! The `MI300X` / `MI350X` agents differ only in their device
//! identity, the shader-engine fan-out, and the per-CU LDS size, so
//! they share builders. `MI450X` mirrors the rocjitsu
//! `amdgpu_gfx1250.json` config and uses a dedicated builder for its
//! larger XCD/IOD fabric.

use mirage_core::agent::{
    AgentDef, AgentTopologyDef, AmdgpuConfig, ComponentDef, ConfigEntry, ForRange, KfdDeviceInfo,
    LinkDef, VirtualMachineConfig,
};

/// All builtin agents, keyed by the name written to disk.
pub fn agents() -> Vec<(&'static str, AgentDef)> {
    vec![
        ("MI300X", mi300x()),
        ("MI350X", mi350x()),
        ("MI450X", mi450x()),
    ]
}

/// `MI300X` builtin agent (registry key `MI300X`).
///
/// The embedded device identity reports `arch = cdna4` /
/// marketing name "AMD Instinct MI350X"; this is preserved verbatim
/// from the original `MI300X.json`.
pub fn mi300x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "cdna4".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                device: KfdDeviceInfo {
                    gpu_id: 38144,
                    gfx_target_version: 90500,
                    vendor_id: 4098,
                    device_id: 5892,
                    family_id: 160,
                    unique_id: 5929628898254127105,
                    marketing_name: "AMD Instinct MI350X".to_string(),
                    drm_render_minor: 128,
                    simd_count: 1024,
                    max_waves_per_simd: 8,
                    num_shader_engines: 4,
                    num_shader_arrays_per_engine: 2,
                    num_cu_per_sh: 4,
                    simd_per_cu: 4,
                    wave_front_size: 64,
                    max_slots_scratch_cu: 32,
                    local_mem_size: 309237645312,
                    lds_size_kb: 160,
                    mem_width: 8192,
                    mem_clk_max: 1600,
                    l1_size_kb: 32,
                    l1_line_size: 128,
                    l1_assoc: 4,
                    l2_size_kb: 4096,
                    l2_line_size: 128,
                    l2_assoc: 16,
                    num_sdma_engines: 5,
                    num_sdma_xgmi_engines: 12,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2700,
                },
            },
        },
        topology: topology("se[0:7]", "160"),
    }
}

/// `MI350X` builtin agent (registry key `MI350X`).
///
/// The embedded device identity reports `arch = cdna3` /
/// marketing name "AMD Instinct MI300X"; this is preserved verbatim
/// from the original `MI350X.json`.
pub fn mi350x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "cdna3".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                device: KfdDeviceInfo {
                    gpu_id: 50148,
                    gfx_target_version: 90402,
                    vendor_id: 4098,
                    device_id: 29856,
                    family_id: 146,
                    unique_id: 0,
                    marketing_name: "AMD Instinct MI300X".to_string(),
                    drm_render_minor: 128,
                    simd_count: 16,
                    max_waves_per_simd: 8,
                    num_shader_engines: 1,
                    num_shader_arrays_per_engine: 1,
                    num_cu_per_sh: 4,
                    simd_per_cu: 4,
                    wave_front_size: 64,
                    max_slots_scratch_cu: 32,
                    local_mem_size: 206158430208,
                    lds_size_kb: 64,
                    mem_width: 8192,
                    mem_clk_max: 1300,
                    l1_size_kb: 32,
                    l1_line_size: 128,
                    l1_assoc: 4,
                    l2_size_kb: 4096,
                    l2_line_size: 128,
                    l2_assoc: 16,
                    num_sdma_engines: 4,
                    num_sdma_xgmi_engines: 6,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2100,
                },
            },
        },
        topology: topology("se[0:1]", "64"),
    }
}

/// `MI450X` builtin agent (registry key `MI450X`).
///
/// The embedded device identity and component tree mirror the
/// rocjitsu `amdgpu_gfx1250.json` config: `arch = gfx1250`, an
/// 8-XCD / 4-SE / 8-CU shader fabric and a 2-IOD memory tier.
pub fn mi450x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "gfx1250".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                device: KfdDeviceInfo {
                    gpu_id: 1250,
                    gfx_target_version: 120500,
                    vendor_id: 4098,
                    device_id: 1250,
                    family_id: 0,
                    unique_id: 1250,
                    marketing_name: "gfx1250".to_string(),
                    drm_render_minor: 0,
                    simd_count: 1024,
                    max_waves_per_simd: 8,
                    num_shader_engines: 4,
                    num_shader_arrays_per_engine: 2,
                    num_cu_per_sh: 4,
                    simd_per_cu: 4,
                    wave_front_size: 32,
                    max_slots_scratch_cu: 0,
                    local_mem_size: 309237645312,
                    lds_size_kb: 160,
                    mem_width: 8192,
                    mem_clk_max: 1600,
                    l1_size_kb: 0,
                    l1_line_size: 0,
                    l1_assoc: 0,
                    l2_size_kb: 4096,
                    l2_line_size: 0,
                    l2_assoc: 0,
                    num_sdma_engines: 5,
                    num_sdma_xgmi_engines: 12,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2700,
                },
            },
        },
        topology: gfx1250_topology(),
    }
}

/// Build the gfx1250 `soc -> {vram, iod, xcd -> {l2, cp, se -> cu}}`
/// component tree and its six link patterns, mirroring the rocjitsu
/// `amdgpu_gfx1250.json` config.
fn gfx1250_topology() -> AgentTopologyDef {
    let cu = ComponentDef {
        name: "cu[0:8]".to_string(),
        r#type: "compute_unit".to_string(),
        config: vec![
            entry("num_wf_slots", "32"),
            entry("sgprs_per_wf", "128"),
            entry("vgprs_per_wf", "1024"),
            entry("lds_size_kb", "160"),
        ],
        ..Default::default()
    };
    let se = ComponentDef {
        name: "se[0:4]".to_string(),
        r#type: "shader_engine".to_string(),
        children: vec![cu],
        ..Default::default()
    };
    let xcd = ComponentDef {
        name: "xcd[0:8]".to_string(),
        r#type: "xcd".to_string(),
        children: vec![leaf("l2", "l2_cache"), leaf("cp", "command_processor"), se],
        ..Default::default()
    };
    let iod = ComponentDef {
        name: "iod[0:2]".to_string(),
        r#type: "iod".to_string(),
        config: vec![entry("num_hbm_stacks", "4")],
        ..Default::default()
    };
    let root = ComponentDef {
        name: "soc".to_string(),
        r#type: "soc".to_string(),
        children: vec![leaf("vram", "gpu_memory"), iod, xcd],
        ..Default::default()
    };
    AgentTopologyDef {
        root,
        links: gfx1250_links(),
    }
}

/// The six gfx1250 link patterns wiring the command processor to the
/// CUs, the CUs to the L2 and IOD memory tier, the IODs to each
/// other, and adjacent CUs together.
fn gfx1250_links() -> Vec<LinkDef> {
    let ijk = || vec![range("i", 0, 8), range("j", 0, 4), range("k", 0, 8)];
    let ijk_adj = || vec![range("i", 0, 8), range("j", 0, 4), range("k", 0, 7)];
    vec![
        LinkDef {
            pattern: "xcd[i].cp.req_[j*8+k] -> xcd[i].se[j].cu[k].cpl".to_string(),
            for_ranges: ijk(),
            weight: 2,
            ..Default::default()
        },
        LinkDef {
            pattern: "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*8+k]".to_string(),
            for_ranges: ijk(),
            weight: 10,
            ..Default::default()
        },
        LinkDef {
            pattern: "xcd[i].l2.req -> iod[i/4].msc.cpl_[i%4]".to_string(),
            for_ranges: vec![range("i", 0, 8)],
            weight: 3,
            ..Default::default()
        },
        LinkDef {
            pattern: "iod[i].peer_req -> iod[j].peer_cpl".to_string(),
            for_ranges: vec![range("i", 0, 2), range("j", 0, 2)],
            where_expr: "i != j".to_string(),
            weight: 1,
            ..Default::default()
        },
        LinkDef {
            pattern: "xcd[i].se[j].cu[k].adj_req -> xcd[i].se[j].cu[k+1].adj_cpl".to_string(),
            for_ranges: ijk_adj(),
            weight: 2,
            ..Default::default()
        },
        LinkDef {
            pattern: "xcd[i].se[j].cu[k+1].adj_req_r -> xcd[i].se[j].cu[k].adj_cpl_r".to_string(),
            for_ranges: ijk_adj(),
            weight: 2,
            ..Default::default()
        },
    ]
}

/// Build the shared `soc -> xcd -> {l2, cp, se -> cu}` component tree.
///
/// `se_range` is the shader-engine range pattern (e.g. `"se[0:7]"`)
/// and `cu_lds_kb` is the per-CU `lds_size_kb` config value.
fn topology(se_range: &str, cu_lds_kb: &str) -> AgentTopologyDef {
    let cu = ComponentDef {
        name: "cu[0:4]".to_string(),
        r#type: "compute_unit".to_string(),
        config: vec![
            entry("num_wf_slots", "32"),
            entry("sgprs_per_wf", "104"),
            entry("vgprs_per_wf", "512"),
            entry("lds_size_kb", cu_lds_kb),
        ],
        ..Default::default()
    };
    let se = ComponentDef {
        name: se_range.to_string(),
        r#type: "shader_engine".to_string(),
        children: vec![cu],
        ..Default::default()
    };
    let xcd = ComponentDef {
        name: "xcd[0:1]".to_string(),
        r#type: "xcd".to_string(),
        children: vec![leaf("l2", "l2_cache"), leaf("cp", "command_processor"), se],
        ..Default::default()
    };
    let root = ComponentDef {
        name: "soc".to_string(),
        r#type: "soc".to_string(),
        children: vec![leaf("vram", "gpu_memory"), xcd],
        ..Default::default()
    };
    AgentTopologyDef {
        root,
        links: links(),
    }
}

/// The two link patterns wiring the command processor to the CUs and
/// the CUs back to the L2, shared by every builtin agent.
fn links() -> Vec<LinkDef> {
    vec![
        LinkDef {
            pattern: "xcd[i].cp.req_[j*4+k] -> xcd[i].se[j].cu[k].cpl".to_string(),
            for_ranges: ijk(),
            weight: 2,
            ..Default::default()
        },
        LinkDef {
            pattern: "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*4+k]".to_string(),
            for_ranges: ijk(),
            weight: 10,
            ..Default::default()
        },
    ]
}

/// The `i in [0,1), j in [0,1), k in [0,4)` loop variables shared by
/// both link patterns.
fn ijk() -> Vec<ForRange> {
    vec![range("i", 0, 1), range("j", 0, 1), range("k", 0, 4)]
}

fn leaf(name: &str, r#type: &str) -> ComponentDef {
    ComponentDef {
        name: name.to_string(),
        r#type: r#type.to_string(),
        ..Default::default()
    }
}

fn entry(key: &str, value: &str) -> ConfigEntry {
    ConfigEntry {
        key: key.to_string(),
        value: value.to_string(),
    }
}

fn range(var_name: &str, start: u32, end: u32) -> ForRange {
    ForRange {
        var_name: var_name.to_string(),
        start,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agents_have_expected_keys() {
        let a = agents();
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].0, "MI300X");
        assert_eq!(a[1].0, "MI350X");
        assert_eq!(a[2].0, "MI450X");
    }

    #[test]
    fn mi300x_identity() {
        let d = mi300x().vm.gpu.device;
        assert_eq!(d.marketing_name, "AMD Instinct MI350X");
        assert_eq!(d.num_shader_engines, 4);
    }

    #[test]
    fn mi350x_identity() {
        let d = mi350x().vm.gpu.device;
        assert_eq!(d.marketing_name, "AMD Instinct MI300X");
        assert_eq!(d.num_shader_engines, 1);
    }

    #[test]
    fn mi450x_identity() {
        let a = mi450x();
        assert_eq!(a.vm.arch, "gfx1250");
        assert_eq!(a.vm.gpu.device.marketing_name, "gfx1250");
        assert_eq!(a.vm.gpu.device.gfx_target_version, 120500);
        assert_eq!(a.topology.links.len(), 6);
    }
}
