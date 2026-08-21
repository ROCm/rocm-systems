//! Strongly-typed builtin [`AgentDef`]s.
//!
//! Each agent mirrors one of the rocjitsu `configs/*.json` files:
//! `MI300X` follows `gfx942_cdna3.json`, `MI350X` follows
//! `gfx950_mi355x.json`, and `MI450X` follows `gfx1250_mi455x.json`.
//! All three share the same `soc -> {vram, iod, xcd -> {l2, cp,
//! se -> cu}}` component tree and six link patterns; they differ
//! only in their device identity, the IOD fan-out, and the per-CU
//! wavefront/register/LDS config.
//!
//! # What comes from the preset and what does not
//!
//! The arch name and the four per-CU limits are read out of the preset
//! at build time (see [`crate::presets`]) rather than written here.
//! They are not mirage's to choose: they say what the emulator will
//! accept, and `cdna5` refusing more than 64 wave slots is a fact about
//! rocjitsu, not a preference. Copied, they drifted — `gfx1250.json`
//! became `gfx1250_mi455x.json` with 80 wave slots cut to 64, the arch
//! rename landed here and the wave slots did not, and every MI450X
//! session died at daemon start.
//!
//! The *shape* of the machine is mirage's own and stays here: the
//! component tree, the IOD fan-out and the KFD device identity. All
//! three builtins are a deliberate uniform 256 CUs, where the presets
//! they mirror are 320, 288 and 256 — an emulated grid mirage picked, not
//! a transcription of the hardware, and `every_agent_is_256_cus` holds it
//! there so that a later widening of this sync cannot quietly change what
//! a profile emulates.

use mirage_core::agent::{
    AgentDef, AgentTopologyDef, AmdgpuConfig, ComponentDef, ConfigEntry, ForRange, KfdDeviceInfo,
    LinkDef, VirtualMachineConfig,
};

use crate::presets::{MI300X, MI350X, MI450X, Preset};

/// All builtin agents, keyed by the name written to disk.
///
/// Agent names are case-insensitive and always stored lowercase, so the
/// registry keys are lowercase to match the on-disk filenames.
pub fn agents() -> Vec<(&'static str, AgentDef)> {
    vec![
        ("mi300x", mi300x()),
        ("mi350x", mi350x()),
        ("mi450x", mi450x()),
        // \NPI new GPU: add a builtin agent mirroring its configs/<gpu>.json here.
    ]
}

/// `MI300X` builtin agent (registry key `MI300X`), mirroring the
/// rocjitsu `gfx942_cdna3.json` config: `arch = cdna3`, marketing
/// name "AMD Instinct MI300X", and an 8-XCD / 4-SE / 8-CU shader
/// fabric over a 4-IOD memory tier.
pub fn mi300x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: MI300X.arch.to_string(),
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
        topology: topology(4, "2", &MI300X),
    }
}

/// `MI350X` builtin agent (registry key `MI350X`), mirroring the
/// rocjitsu `gfx950_mi355x.json` config: `arch = cdna4`, marketing
/// name "AMD Instinct MI350X", and an 8-XCD / 4-SE / 8-CU shader
/// fabric over a 2-IOD memory tier.
pub fn mi350x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: MI350X.arch.to_string(),
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
        topology: topology(2, "4", &MI350X),
    }
}

/// `MI450X` builtin agent (registry key `MI450X`), mirroring the
/// rocjitsu `gfx1250_mi455x.json` config: `arch = cdna5`, an
/// 8-XCD / 4-SE / 8-CU shader fabric and a 2-IOD memory tier.
pub fn mi450x() -> AgentDef {
    AgentDef {
        vm: VirtualMachineConfig {
            arch: MI450X.arch.to_string(),
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
        topology: topology(2, "4", &MI450X),
    }
}

/// Build the shared `soc -> {vram, iod, xcd -> {l2, cp, se -> cu}}`
/// component tree used by every builtin agent.
///
/// `num_iods` / `num_hbm_stacks` set the IOD memory-tier fan-out, which
/// is mirage's own; `preset` supplies the per-CU `num_wf_slots`,
/// `sgprs_per_wf`, `vgprs_per_wf` and `lds_size_kb`, which are rocjitsu's
/// and are read from `configs/*.json` at build time.
///
/// The `cu[0:8]` here is deliberate and is *not* the preset's — see the
/// module docs on why the CU count is mirage's to pick.
fn topology(num_iods: u32, num_hbm_stacks: &str, preset: &Preset) -> AgentTopologyDef {
    let cu = ComponentDef {
        name: "cu[0:8]".to_string(),
        r#type: "compute_unit".to_string(),
        config: vec![
            entry("num_wf_slots", preset.num_wf_slots),
            entry("sgprs_per_wf", preset.sgprs_per_wf),
            entry("vgprs_per_wf", preset.vgprs_per_wf),
            entry("lds_size_kb", preset.lds_size_kb),
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn agents_have_expected_keys() {
        let a = agents();
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].0, "mi300x");
        assert_eq!(a[1].0, "mi350x");
        assert_eq!(a[2].0, "mi450x");
    }

    /// The number of CUs a builtin emulates is mirage's, not the
    /// preset's, and this is what says so.
    ///
    /// The presets these agents mirror are 320, 288 and 256 CUs — the
    /// real parts. The builtins are a uniform 256, which is the grid
    /// mirage chose to emulate, and changing it changes every result
    /// every profile produces. Now that the per-CU limits are read from
    /// the preset it would be a small and reasonable-looking step to read
    /// the component tree from it too; this fails if anybody takes it.
    #[test]
    fn every_agent_is_256_cus() {
        for (name, agent) in agents() {
            assert_eq!(
                compute_units(&agent.topology.root),
                256,
                "`{name}` no longer emulates 256 CUs; the preset's own CU \
                 count is not mirage's to adopt — see the module docs"
            );
        }
    }

    /// The per-CU limits really do come from the preset.
    ///
    /// Tautological against the current code and deliberately so: it is
    /// here to fail the day somebody writes a literal back into
    /// `topology()`, which is how these drifted from the preset the first
    /// time.
    #[test]
    fn the_per_cu_limits_come_from_the_preset() {
        for (agent, preset) in [
            (mi300x(), &MI300X),
            (mi350x(), &MI350X),
            (mi450x(), &MI450X),
        ] {
            assert_eq!(agent.vm.arch, preset.arch, "{}", preset.preset);
            for (key, want) in [
                ("num_wf_slots", preset.num_wf_slots),
                ("sgprs_per_wf", preset.sgprs_per_wf),
                ("vgprs_per_wf", preset.vgprs_per_wf),
                ("lds_size_kb", preset.lds_size_kb),
            ] {
                assert_eq!(cu_config(&agent, key), want, "{} {key}", preset.preset);
            }
        }
    }

    /// Every compute unit under `node`, multiplying out the `[0:N]`
    /// ranges on the way down.
    fn compute_units(node: &ComponentDef) -> u32 {
        let span = node
            .name
            .rsplit_once("[0:")
            .and_then(|(_, rest)| rest.strip_suffix(']'))
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(1);
        if node.r#type == "compute_unit" {
            return span;
        }
        span * node.children.iter().map(compute_units).sum::<u32>()
    }

    /// One per-CU config value from an agent's component tree.
    ///
    /// Walks to the first compute unit rather than spelling out
    /// `soc -> xcd -> se -> cu`, which would break whenever the tree
    /// grows a level.
    fn cu_config(agent: &AgentDef, key: &str) -> String {
        fn find(node: &ComponentDef, key: &str) -> Option<String> {
            if node.r#type == "compute_unit"
                && let Some(entry) = node.config.iter().find(|e| e.key == key)
            {
                return Some(entry.value.clone());
            }
            node.children.iter().find_map(|child| find(child, key))
        }
        find(&agent.topology.root, key).unwrap_or_else(|| panic!("no `{key}` on any compute unit"))
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
        assert_eq!(a.vm.arch, "cdna5");
        assert_eq!(a.vm.gpu.device.marketing_name, "gfx1250");
        assert_eq!(a.vm.gpu.device.gfx_target_version, 120500);
        assert_eq!(a.topology.links.len(), 6);
    }
}
