//! Strongly-typed builtin [`AgentDef`]s.
//!
//! Each agent mirrors one of the rocjitsu `configs/*.json` files:
//! `MI300X` follows `gfx942_cdna3.json`, `MI350X` follows
//! `gfx950_cdna4.json`, and `MI450X` follows `gfx1250.json`.
//! All three share the same `soc -> {vram, iod, xcd -> {l2, cp,
//! se -> cu}}` component tree and six link patterns; they differ
//! only in their device identity, the IOD fan-out, and the per-CU
//! wavefront/register/LDS config.

use mirage_core::agent::{
    AgentDef, AgentTopologyDef, AmdgpuConfig, ComponentDef, ConfigEntry, ForRange, KfdDeviceInfo,
    LinkDef, VirtualMachineConfig,
};

/// All builtin agents, keyed by the name written to disk.
///
/// Agent names are case-insensitive and always stored lowercase, so the
/// registry keys are lowercase to match the on-disk filenames.
pub fn agents() -> Vec<(&'static str, AgentDef)> {
    vec![
        ("mi300x", mi300x()),
        ("mi350x", mi350x()),
        ("mi450x", mi450x()),
    ]
}

/// `MI300X` builtin agent (registry key `MI300X`), mirroring the
/// rocjitsu `gfx942_cdna3.json` config: `arch = cdna3`, marketing
/// name "AMD Instinct MI300X", and an 8-XCD / 4-SE / 8-CU shader
/// fabric over a 4-IOD memory tier.
pub fn mi300x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "cdna3".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                num_gpus: 1,
                device: KfdDeviceInfo {
                    gpu_id: 50148,
                    gfx_target_version: 90402,
                    vendor_id: 4098,
                    device_id: 29856,
                    family_id: 146,
                    unique_id: 0,
                    marketing_name: "AMD Instinct MI300X".to_string(),
                    // First DRM render node (`/dev/dri/renderD128`); the
                    // rocjitsu schema defaults this to 128 and a 0 here
                    // maps the emulated GPU onto a non-existent render
                    // node, so HSA aborts with OUT_OF_RESOURCES.
                    drm_render_minor: 128,
                    simd_count: 1216,
                    max_waves_per_simd: 8,
                    num_shader_engines: 4,
                    num_shader_arrays_per_engine: 2,
                    num_cu_per_sh: 5,
                    simd_per_cu: 4,
                    wave_front_size: 64,
                    local_mem_size: 206158430208,
                    lds_size_kb: 64,
                    mem_width: 8192,
                    mem_clk_max: 1300,
                    l2_size_kb: 4096,
                    num_sdma_engines: 4,
                    num_sdma_xgmi_engines: 6,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2100,
                    ..Default::default()
                },
            },
        },
        topology: topology(4, "2", "32", "104", "512", "64"),
    }
}

/// `MI350X` builtin agent (registry key `MI350X`), mirroring the
/// rocjitsu `gfx950_cdna4.json` config: `arch = cdna4`, marketing
/// name "AMD Instinct MI350X", and an 8-XCD / 4-SE / 8-CU shader
/// fabric over a 2-IOD memory tier.
pub fn mi350x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "cdna4".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                num_gpus: 1,
                device: KfdDeviceInfo {
                    gpu_id: 38144,
                    gfx_target_version: 90500,
                    vendor_id: 4098,
                    device_id: 5892,
                    family_id: 160,
                    unique_id: 5929628898254127105,
                    marketing_name: "AMD Instinct MI350X".to_string(),
                    // First DRM render node (`/dev/dri/renderD128`); the
                    // rocjitsu schema defaults this to 128 and a 0 here
                    // maps the emulated GPU onto a non-existent render
                    // node, so HSA aborts with OUT_OF_RESOURCES.
                    drm_render_minor: 128,
                    simd_count: 1024,
                    max_waves_per_simd: 8,
                    num_shader_engines: 4,
                    num_shader_arrays_per_engine: 2,
                    num_cu_per_sh: 4,
                    simd_per_cu: 4,
                    wave_front_size: 64,
                    local_mem_size: 309237645312,
                    lds_size_kb: 160,
                    mem_width: 8192,
                    mem_clk_max: 1600,
                    l2_size_kb: 4096,
                    num_sdma_engines: 5,
                    num_sdma_xgmi_engines: 12,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2700,
                    ..Default::default()
                },
            },
        },
        topology: topology(2, "4", "32", "104", "512", "160"),
    }
}

/// `MI450X` builtin agent (registry key `MI450X`), mirroring the
/// rocjitsu `gfx1250.json` config: `arch = gfx1250`, an
/// 8-XCD / 4-SE / 8-CU shader fabric and a 2-IOD memory tier.
pub fn mi450x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: "gfx1250".to_string(),
            gpu: AmdgpuConfig {
                num_xcds: 0,
                num_iods: 0,
                memory: None,
                num_gpus: 1,
                device: KfdDeviceInfo {
                    gpu_id: 1250,
                    gfx_target_version: 120500,
                    vendor_id: 4098,
                    device_id: 1250,
                    family_id: 0,
                    unique_id: 1250,
                    marketing_name: "gfx1250".to_string(),
                    // First DRM render node (`/dev/dri/renderD128`); the
                    // rocjitsu schema defaults this to 128 and a 0 here
                    // maps the emulated GPU onto a non-existent render
                    // node, so HSA aborts with OUT_OF_RESOURCES.
                    drm_render_minor: 128,
                    simd_count: 1024,
                    max_waves_per_simd: 20,
                    num_shader_engines: 4,
                    num_shader_arrays_per_engine: 2,
                    num_cu_per_sh: 4,
                    simd_per_cu: 4,
                    wave_front_size: 32,
                    local_mem_size: 309237645312,
                    lds_size_kb: 160,
                    mem_width: 8192,
                    mem_clk_max: 1600,
                    l2_size_kb: 4096,
                    num_sdma_engines: 5,
                    num_sdma_xgmi_engines: 12,
                    num_cp_queues: 128,
                    max_engine_clk_fcompute: 2700,
                    ..Default::default()
                },
            },
        },
        topology: topology(2, "4", "80", "128", "1024", "160"),
    }
}

/// Build the shared `soc -> {vram, iod, xcd -> {l2, cp, se -> cu}}`
/// component tree used by every builtin agent.
///
/// `num_iods` / `num_hbm_stacks` set the IOD memory-tier fan-out and
/// the remaining arguments populate the per-CU `num_wf_slots`,
/// `sgprs_per_wf`, `vgprs_per_wf`, and `lds_size_kb` config values.
fn topology(
    num_iods: u32,
    num_hbm_stacks: &str,
    num_wf_slots: &str,
    sgprs_per_wf: &str,
    vgprs_per_wf: &str,
    lds_size_kb: &str,
) -> AgentTopologyDef {
    let cu = ComponentDef {
        name: "cu[0:8]".to_string(),
        r#type: "compute_unit".to_string(),
        config: vec![
            entry("num_wf_slots", num_wf_slots),
            entry("sgprs_per_wf", sgprs_per_wf),
            entry("vgprs_per_wf", vgprs_per_wf),
            entry("lds_size_kb", lds_size_kb),
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
        name: format!("iod[0:{num_iods}]"),
        r#type: "iod".to_string(),
        config: vec![entry("num_hbm_stacks", num_hbm_stacks)],
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
        links: links(num_iods),
    }
}

/// The six link patterns wiring the command processor to the CUs,
/// the CUs to the L2 and IOD memory tier, the IODs to each other,
/// and adjacent CUs together. Only the L2->IOD fan-out and the IOD
/// peer range depend on `num_iods`.
fn links(num_iods: u32) -> Vec<LinkDef> {
    let xcds_per_iod = 8 / num_iods;
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
            pattern: format!("xcd[i].l2.req -> iod[i/{xcds_per_iod}].msc.cpl_[i%{xcds_per_iod}]"),
            for_ranges: vec![range("i", 0, 8)],
            weight: 3,
            ..Default::default()
        },
        LinkDef {
            pattern: "iod[i].peer_req -> iod[j].peer_cpl".to_string(),
            for_ranges: vec![range("i", 0, num_iods), range("j", 0, num_iods)],
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
        assert_eq!(a[0].0, "mi300x");
        assert_eq!(a[1].0, "mi350x");
        assert_eq!(a[2].0, "mi450x");
    }

    #[test]
    fn mi300x_identity() {
        let a = mi300x();
        let d = &a.vm.gpu.device;
        assert_eq!(d.marketing_name, "AMD Instinct MI300X");
        assert_eq!(d.num_shader_engines, 4);
        assert_eq!(d.simd_count, 1216);
        assert_eq!(a.topology.links.len(), 6);
    }

    #[test]
    fn mi350x_identity() {
        let d = mi350x().vm.gpu.device;
        assert_eq!(d.marketing_name, "AMD Instinct MI350X");
        assert_eq!(d.num_shader_engines, 4);
        assert_eq!(d.simd_count, 1024);
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
