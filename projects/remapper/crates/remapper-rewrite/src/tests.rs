use super::{
    rewrite::{Prefetch, RewriteOptions},
    table::Table,
    *,
};
use std::path::Path;
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/wgremap");
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(Path::new(FIXTURES).join(format!("{name}.hsaco"))).unwrap()
}
fn tools_available() -> bool {
    if llvm_tool("llvm-objdump").is_err() || llvm_tool("llvm-mc").is_err() {
        eprintln!("skipping remap fixture tests: ROCm LLVM tools unavailable");
        false
    } else {
        true
    }
}
fn options(cluster: bool) -> RewriteOptions {
    RewriteOptions {
        dispatch_index: 0,
        workgroups: if cluster { [512, 1, 1] } else { [256, 1, 1] },
        waves_per_workgroup: if cluster { 8 } else { 4 },
        cluster_size: if cluster { 4 } else { 1 },
        cluster_dims: if cluster { [4, 1, 1] } else { [1, 1, 1] },
        shader_engines: 16,
        traced_wgp: 0,
        prefetch: Prefetch::PerWave,
        launch_model: rewrite::LaunchModel::Table,
        padded_workgroups: 0,
        loop_markers: None,
    }
}
#[test]
fn fixture_analysis_and_prototype_bytes() {
    if !tools_available() {
        return;
    }
    for name in ["v8", "mxfp", "cluster2x2"] {
        let code = fixture(name);
        let a = analyze::analyze_code_object(&code, None).unwrap();
        a.refusal().unwrap();
        assert_eq!(a.reserved[&9], 105);
        if name == "v8" {
            assert!(a.kd.preload_length() > 0);
            assert!(a.rank_reads.is_empty());
        }
        if name == "mxfp" {
            assert!(a.reads_ttmp(7));
        }
        if name == "cluster2x2" {
            assert!(a
                .rank_reads
                .iter()
                .any(|r| r.text.ends_with(", 15") && r.kind == analyze::RankKind::Data));
            for mask in [13, 14] {
                assert!(a
                    .rank_reads
                    .iter()
                    .any(|r| r.text.ends_with(&format!(", {mask}"))
                        && r.kind == analyze::RankKind::Multicast));
            }
        }
        let mut opts = options(name == "cluster2x2");
        if name == "mxfp" {
            opts.workgroups = [64, 32, 1];
        }
        let dir = tempfile::tempdir().unwrap();
        let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
        let new = elf::Elf::parse(std::fs::read(dir.path().join(&m.code_object)).unwrap()).unwrap();
        let golden = elf::Elf::parse(fixture(&format!("{name}-golden"))).unwrap();
        let p = new.vaddr_to_offset(m.rewrite.prologue_vaddr).unwrap();
        let q = golden.vaddr_to_offset(m.rewrite.prologue_vaddr).unwrap();
        assert_eq!(
            new.data[p..p + m.rewrite.prologue_bytes as usize],
            golden.data[q..q + m.rewrite.prologue_bytes as usize],
            "{name} prologue"
        );
        let p = new.vaddr_to_offset(a.kernel_vaddr).unwrap();
        let q = golden.vaddr_to_offset(a.kernel_vaddr).unwrap();
        assert_eq!(
            new.data[p..p + a.kernel_size as usize],
            golden.data[q..q + a.kernel_size as usize],
            "{name} body"
        );
        let kd = analyze::KernelDescriptor::parse(&new, a.kd.vaddr).unwrap();
        assert_eq!(kd.entry_vaddr, m.rewrite.prologue_vaddr);
        assert_eq!(kd.kernarg_size as usize, m.control_kernarg_offset + 8);
        assert_eq!(kd.preload_length(), 0);
        let b = m.rewrite.prologue_vaddr + m.rewrite.prologue_bytes - 4;
        let off = new.vaddr_to_offset(b).unwrap();
        let simm = i16::from_le_bytes(new.data[off..off + 2].try_into().unwrap()) as i64;
        assert_eq!((b as i64 + 4 + simm * 4) as u64, a.kd.entry_vaddr);
    }
}
/// Rewritten code object, prologue source and manifest for each fixture and table-model option set, hashed. These
/// must not change when other launch models are added: the table path is what every validated kernel uses.
const TABLE_PATH_HASHES: &[(&str, &str)] = &[
    (
        "v8/per-wave",
        "8895c3dfa35fa639dd8fdf99fe7ed384a521ba8d7f99419ad4d45d264a602e2e",
    ),
    (
        "mxfp/per-wave",
        "99b56eed5dda2ebc5f6f05266a6df468ec319b810e2ce6b117cc69d31543eaf0",
    ),
    (
        "cluster2x2/per-wave",
        "5b9cded493ec0112ddda02f15dc4ae3565c69fae1f11973cc84028c86bf36279",
    ),
    (
        "v8/wave0",
        "4aba595b0e900fb85043428c90dd17d2efdeb6c1f3b504202b92b7c12fe7f1e3",
    ),
    (
        "mxfp/wave0",
        "e0792d189978aefffd62ea1fc6a60f07066c04684955353160b7ffd71c6ab65e",
    ),
    (
        "cluster2x2/wave0",
        "ffb53a36e7ae5d32a8742705dbc001d9b4202c2554d7da25363a415244d96e8d",
    ),
    (
        "v8/none",
        "6c7d1583e3cd97efc90251441fd06fe08f02e53f41af1a68ceb98a43e9c273ad",
    ),
    (
        "mxfp/none",
        "f9eac678e94593ec4ef97027734d371202a452ae327b1919d405d6cb54a3ed4e",
    ),
    (
        "cluster2x2/none",
        "f8c72b8098d7838d241cd60e37fcfba1505a91716f765e7dd0908df240439085",
    ),
    (
        "fp8-4x4/per-wave",
        "940be06e98afb87d18b4baf6037638a838f9dc03626c47d8797464d8bb4130a0",
    ),
    (
        "warp-pipeline-b/per-wave",
        "ef479af8364927bce34a7f49cf7f7991e9e1463964b2b6693e1f6639174939de",
    ),
];

fn table_path_cases() -> Vec<(String, &'static str, RewriteOptions)> {
    let mut cases = vec![];
    for prefetch in [Prefetch::PerWave, Prefetch::Wave0, Prefetch::None] {
        let p = prefetch.name();
        let mut o = options(false);
        o.prefetch = prefetch;
        cases.push((format!("v8/{p}"), "v8", o));
        let mut o = options(false);
        o.prefetch = prefetch;
        o.workgroups = [64, 32, 1];
        cases.push((format!("mxfp/{p}"), "mxfp", o));
        let mut o = options(true);
        o.prefetch = prefetch;
        cases.push((format!("cluster2x2/{p}"), "cluster2x2", o));
    }
    let mut o = options(true);
    o.workgroups = [1024, 1, 1];
    o.cluster_size = 16;
    o.cluster_dims = [16, 1, 1];
    cases.push(("fp8-4x4/per-wave".into(), "fp8-4x4", o));
    cases.push((
        "warp-pipeline-b/per-wave".into(),
        "warp-pipeline-b",
        options(false),
    ));
    cases
}

#[test]
fn table_path_outputs_unchanged() {
    if !tools_available() {
        return;
    }
    let mut got = vec![];
    for (label, name, opts) in table_path_cases() {
        let code = fixture(name);
        let a = analyze::analyze_code_object(&code, None).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
        let mut bytes = std::fs::read(dir.path().join(&m.code_object)).unwrap();
        bytes.extend(std::fs::read(dir.path().join("rewrite/prologue.s")).unwrap());
        bytes.extend(serde_json::to_vec(&m).unwrap());
        got.push((label, blake3::hash(&bytes).to_hex().to_string()));
    }
    for (label, hash) in &got {
        println!("    (\"{label}\", \"{hash}\"),");
    }
    assert_eq!(got.len(), TABLE_PATH_HASHES.len(), "hash table out of date");
    for ((label, hash), (want_label, want)) in got.iter().zip(TABLE_PATH_HASHES) {
        assert_eq!((label.as_str(), hash.as_str()), (*want_label, *want));
    }
}

#[test]
fn operand_patch_accepts_unaligned_vop3_sources() {
    // unified_attention_3d: VOP3 src1 = ttmp9 (117) at bits [49:41] -> s105.
    let old = [0x14, 0x00, 0x35, 0xd6, 0x15, 0xea, 0x50, 0x04];
    let new = [0x14, 0x00, 0x35, 0xd6, 0x15, 0xd2, 0x50, 0x04];
    assert!(prologue::operand_patch_ok(&old, &new, &[(117, 105)]));
    // Byte-aligned SOP operand.
    assert!(prologue::operand_patch_ok(
        &[0x75, 0x00, 0x8a, 0xbe],
        &[0x69, 0x00, 0x8a, 0xbe],
        &[(117, 105)]
    ));
    // Any change outside the operand window, including the 9-bit field's high bit, is refused.
    let mut wrong = new;
    wrong[6] ^= 0x01;
    assert!(!prologue::operand_patch_ok(&old, &wrong, &[(117, 105)]));
    assert!(!prologue::operand_patch_ok(&old, &new, &[(117, 104)]));
    assert!(!prologue::operand_patch_ok(&old, &old, &[(117, 105)]));
}

#[test]
fn few_wave_workgroups_key_by_simd() {
    if !tools_available() {
        return;
    }
    let code = fixture("warp-pipeline-b");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    for (waves, keys) in [(2, 4096), (4, 1024)] {
        let mut opts = options(false);
        opts.waves_per_workgroup = waves;
        let dir = tempfile::tempdir().unwrap();
        let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
        let src = std::fs::read_to_string(dir.path().join("rewrite/prologue.s")).unwrap();
        assert_eq!(m.num_keys, keys);
        assert_eq!(src.contains("0x20008"), waves < 4);
    }
}

#[test]
fn synthetic_unsafe_flow_refuses() {
    if !tools_available() {
        return;
    }
    let code = fixture("v8");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let mut listing = disassemble_bytes(&code).unwrap();
    let first = &a.insts[0];
    let text = first.text();
    listing = listing.replacen(&text, "s_swappc_b64 s[30:31], s[16:17]", 1);
    assert!(analyze::analyze(&code, None, &listing)
        .unwrap()
        .problems
        .iter()
        .any(|p| p.contains("callees are not analyzed")));
    let listing = format!("\ts_branch 32767 // {:012X}: BFA07FFF\n", a.kernel_vaddr);
    assert!(analyze::analyze(&code, None, &listing)
        .unwrap()
        .problems
        .iter()
        .any(|p| p.contains("leaves the kernel")));
}
fn manifest() -> manifest::Manifest {
    // Keep table tests independent of LLVM by deserializing a small explicit manifest.
    serde_json::from_value(serde_json::json!({
        "version":1,"symbol":"test","code_object":"kernel.hsaco","source_code_object":"source.hsaco","dispatch_index":0,"workgroups":[2,1,1],"num_workgroups":2,"waves_per_workgroup":4,"control_kernarg_offset":8,"control_bytes":64,"num_keys":2,"log_words":8,"num_log_records":2,"min_group_segment_bytes":8,"extra_buffers":[],"cluster":null,"table":null,
        "rewrite":{"prologue_vaddr":4096,"prologue_bytes":128,"original_entry_vaddr":256,"kernel_vaddr":256,"kernel_size":128,"original_prefetch_lines":1,"descriptor_prefetch_lines":1,"body_prefetch":"per-wave","reserved_sgprs":{},"prologue_sgprs":"s4..s27","workgroup_id_sites":[],"cluster_rank_sites":[],"preload_converted":false}
    })).unwrap()
}
#[test]
fn ids_the_grid_does_not_vary_become_zero() {
    if !tools_available() {
        return;
    }
    let code = fixture("mxfp");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    assert!(a.reads_ttmp(7) && a.reserved.contains_key(&7));
    let mut one_d = a.clone();
    one_d.zero_constant_ids([256, 1, 1]);
    assert!(!one_d.reads_ttmp(7) && !one_d.reserved.contains_key(&7));
    assert!(one_d.sites.iter().any(|s| s.zeroed == [7]));
    assert_eq!(one_d.reserved[&9], a.reserved[&9]);
    // A grid one workgroup wide keeps ttmp9; one that varies in nothing has nothing to remap.
    let mut column = a.clone();
    column.zero_constant_ids([1, 64, 1]);
    assert!(!column.reads_ttmp(9) && column.reads_ttmp(7) && !column.reserved.contains_key(&9));
    column.refusal().unwrap();
    let mut single = a.clone();
    single.zero_constant_ids([1, 1, 1]);
    assert!(single.refusal().is_err());
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &options(false), dir.path()).unwrap();
    assert!(!m.rewrite.reserved_sgprs.contains_key("ttmp7"));
    let ttmp7_reads: Vec<_> = m
        .rewrite
        .workgroup_id_sites
        .iter()
        .filter(|s| s.original.contains("ttmp7"))
        .collect();
    assert!(!ttmp7_reads.is_empty());
    for s in ttmp7_reads {
        assert!(
            !s.patched.contains("ttmp7") && s.patched.contains(", 0"),
            "{} -> {}",
            s.original,
            s.patched
        );
    }
    // A two-dimensional grid still remaps Y and Z.
    let mut opts = options(false);
    opts.workgroups = [16, 16, 1];
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
    assert!(m.rewrite.reserved_sgprs.contains_key("ttmp7"));
    assert_eq!(m.rewrite.workgroup_id_sites.len(), a.sites.len());
}

#[test]
fn isa_table_agrees_with_llvm_on_every_fixture_instruction() {
    if !tools_available() {
        return;
    }
    let (mut checked, mut undecided) = (0, 0);
    for name in [
        "v8",
        "mxfp",
        "cluster2x2",
        "fp8-4x4",
        "warp-pipeline-b",
        "gemm-proto-4x4",
    ] {
        let code = fixture(name);
        let elf = elf::Elf::parse(code.clone()).unwrap();
        for i in isa::parse_objdump(&disassemble_bytes(&code).unwrap()) {
            if i.mnemonic == ".long" || i.mnemonic.starts_with('<') {
                continue;
            }
            let off = elf.vaddr_to_offset(i.addr).unwrap();
            match spec::decode(&code[off..code.len().min(off + 16)]) {
                Some(d) => {
                    checked += 1;
                    assert_eq!(
                        d.len as u64,
                        i.size(),
                        "{name}: length of {} at {:#x}",
                        i.text(),
                        i.addr
                    );
                    let base = |m: &str| {
                        m.trim_end_matches("_e32")
                            .trim_end_matches("_e64")
                            .trim_end_matches("_e64_dpp")
                            .trim_end_matches("_dpp")
                            .to_string()
                    };
                    assert_eq!(
                        base(&d.name),
                        base(&i.mnemonic),
                        "{name}: {} at {:#x}",
                        i.text(),
                        i.addr
                    );
                }
                None => undecided += 1,
            }
        }
    }
    // Branches, dual-issue (VOPD) forms and encodings with two matching layouts are left undecided on purpose.
    assert!(
        checked > 5 * undecided,
        "{checked} decoded, {undecided} not decided"
    );
}

#[test]
fn one_dimensional_grids_ignore_the_unwritten_ttmp7() {
    // Recorded on gfx1250: a kernel without workgroup ids Y and Z sees leftover ttmp7 values.
    let m = manifest();
    let log = vec![[0, 0, 0, 153, 1, 0, 0, 153], [1, 0, 1, 1023, 1, 0, 1, 1023]];
    let t = Table::build(&m, &log).unwrap();
    assert_eq!(t.entries, [[0, 153], [1, 1023]]);
    assert!(t.check(&m, &log, None).unwrap().valid());
    let mut twice = log.clone();
    twice[1][2] = 0;
    twice[1][6] = 0;
    assert!(
        Table::build(&m, &twice).is_err(),
        "x alone still names the tile"
    );
    let mut two_d = manifest();
    two_d.workgroups = [1, 2, 1];
    assert!(
        Table::build(&two_d, &log).is_err(),
        "a 2-D grid's ttmp7 is its y id"
    );
}

#[test]
fn placement_invariants() {
    let m = manifest();
    let log = vec![[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]];
    let t = Table::build(&m, &log).unwrap();
    assert!(t.check(&m, &log, None).unwrap().valid());
    let mut duplicate = log.clone();
    duplicate[1][0] = 0;
    assert!(Table::build(&m, &duplicate).is_err());
    let mut unstable = log.clone();
    unstable[1][0] = 0;
    unstable[1][1] = 1;
    assert!(t.check_record(&m, &unstable).is_err());
    let mut wrong = log.clone();
    wrong[0][2] = 1;
    assert!(!t.check(&m, &wrong, None).unwrap().valid());
    let mut overflow = log.clone();
    overflow[0][1] = 100;
    assert!(t.check(&m, &overflow, None).is_err());
    let drifted = mode::Outcome {
        overflow: 1,
        logged: 2,
        expected_log_records: 2,
        golden_ok: Some(false),
    };
    assert!(!drifted.valid(mode::Mode::Table) && drifted.drifted());
}

#[test]
fn prefetch_variants_and_cluster_preload_render() {
    if !tools_available() {
        return;
    }
    let code = fixture("cluster2x2");
    let mut a = analyze::analyze_code_object(&code, None).unwrap();
    let mut opts = options(true);
    for prefetch in [Prefetch::PerWave, Prefetch::Wave0, Prefetch::None] {
        opts.prefetch = prefetch;
        let (bytes, _, _) = prologue::build_prologue(&a, &opts, 0x10000, 96).unwrap();
        assert!(!bytes.is_empty());
        if prefetch == Prefetch::Wave0 {
            let n = a.kd.inst_pref_lines().div_ceil(31);
            let skip = i16::from_le_bytes(bytes[12..14].try_into().unwrap());
            assert_eq!(skip as u32, 2 * n);
        }
    }
    // Exercise the cluster preload conversion path without a GPU.
    a.kd.kernarg_preload = 4;
    a.preload_sgpr_base = 2;
    opts.traced_wgp = 3;
    opts.shader_engines = 8;
    let (_, source, _) = prologue::build_prologue(&a, &opts, 0x10000, 96).unwrap();
    assert!(source.contains("s_load_b64 s[2:3], s[0:1], 0x0"));
    assert!(source.contains("s_load_b64 s[4:5], s[0:1], 0x8"));
    // Only location selection uses traced WGP; the leader is always wave/rank zero.
    assert!(source.contains("s_or_b32 s60, s52, s63\n\ts_cmp_lg_u32 s60, 0"));
}

#[test]
fn cluster_placement_checks_logical_rank_and_slot() {
    let mut m = manifest();
    m.workgroups = [8, 1, 1];
    m.num_workgroups = 8;
    m.num_keys = 4;
    m.cluster = Some(manifest::ClusterLayout {
        size: 4,
        dims: [4, 1, 1],
        num_clusters: 2,
        shader_engines: 2,
        traced_wgp: 3,
    });
    let log = vec![[0, 0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 2, 1, 0, 1, 0]];
    let t = Table::build(&m, &log).unwrap();
    let mut renamed = log.clone();
    renamed[0][3] = 3;
    renamed[0][7] = 2;
    let slots = vec![(1 << 31) | (3 << 16) | 3, (1 << 31) | (2 << 16) | 35];
    let p = t.check(&m, &renamed, Some(&slots)).unwrap();
    assert!(p.valid());
    assert_eq!(p.renamed, Some(1));
    let p = t.check(&m, &renamed, Some(&[0, 0])).unwrap();
    assert_eq!(p.slot_ok, Some(false));
    assert!(!p.valid());
    renamed[0][7] = 1;
    assert!(!t.check(&m, &renamed, Some(&slots)).unwrap().valid());
}

#[test]
fn two_dimensional_cluster_ranks_use_both_fields() {
    let c = manifest::ClusterLayout {
        size: 16,
        dims: [4, 4, 1],
        num_clusters: 16,
        shader_engines: 16,
        traced_wgp: 0,
    };
    assert!(c.valid_rank(0x33) && c.valid_rank(0x30) && !c.valid_rank(0x04) && !c.valid_rank(0x40));
    let old = manifest::ClusterLayout {
        dims: [0; 3],
        ..c.clone()
    };
    assert_eq!(old.dims(), [16, 1, 1]);
    // A renamed traced member: hardware rank (x 1, y 2) with c = (x 3, y 1) is logical (x 2, y 3).
    let mut m = manifest();
    m.workgroups = [16, 16, 1];
    m.num_workgroups = 256;
    m.num_keys = 32;
    m.num_log_records = 16;
    m.cluster = Some(c);
    let log: Vec<[u32; 8]> = (0..16).map(|i| [i, 0, i, 0x32, 1, 0, i, 0]).collect();
    let t = Table::build(&m, &log).unwrap();
    let mut renamed = log.clone();
    renamed[5][3] = 0x21;
    renamed[5][7] = 0x13;
    assert!(t.check(&m, &renamed, None).unwrap().valid());
    renamed[5][7] = 0x12;
    assert!(!t.check(&m, &renamed, None).unwrap().valid());
}

#[test]
fn fp8_4x4_data_rank_reads_are_renamed_and_multicast_reads_kept() {
    if !tools_available() {
        return;
    }
    let a = analyze::analyze_code_object(&fixture("fp8-4x4"), None).unwrap();
    a.refusal().unwrap();
    assert!(a.kd.preload_length() > 0);
    for (suffix, kind) in [
        (", 6", analyze::RankKind::Data),
        (", 15", analyze::RankKind::Data),
        (", 3", analyze::RankKind::Multicast),
        (", 12", analyze::RankKind::Multicast),
    ] {
        assert!(
            a.rank_reads
                .iter()
                .any(|r| r.text.ends_with(suffix) && r.kind == kind),
            "missing {suffix} {kind:?}"
        );
    }
    let mut opts = options(true);
    opts.workgroups = [1024, 1, 1];
    opts.cluster_size = 16;
    opts.cluster_dims = [16, 1, 1];
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&fixture("fp8-4x4"), &a.symbol, &opts, dir.path()).unwrap();
    assert!(m.rewrite.preload_converted);
    assert_eq!(m.num_log_records, 64);
}

#[test]
fn prologue_without_slack_after_text_goes_below_it() {
    if !tools_available() {
        return;
    }
    let code = fixture("warp-pipeline-b");
    let original = elf::Elf::parse(code.clone()).unwrap();
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &options(false), dir.path()).unwrap();
    let text = original.section(".text").unwrap();
    let span = m.rewrite.prologue_bytes.div_ceil(256) * 256;
    assert_eq!(m.rewrite.prologue_vaddr, text.addr - span);
    let new =
        elf::Elf::parse(std::fs::read(dir.path().join("rewrite/kernel.hsaco")).unwrap()).unwrap();
    let new_text = new.section(".text").unwrap();
    assert_eq!(
        (new_text.addr, new_text.size),
        (text.addr - span, text.size + span)
    );
    let loads = |e: &elf::Elf| -> Vec<(u64, u64)> {
        e.segments
            .iter()
            .filter(|p| p.kind == elf::PT_LOAD)
            .map(|p| (p.vaddr, p.vaddr + p.memsz))
            .collect()
    };
    let (old_loads, new_loads) = (loads(&original), loads(&new));
    assert_eq!(old_loads.len(), new_loads.len());
    for (o, n) in old_loads.iter().zip(&new_loads) {
        if o.0 == text.addr {
            assert_eq!(*n, (o.0 - span, o.1));
        } else {
            assert_eq!(n, o);
        }
    }
    for s in original
        .sections
        .iter()
        .filter(|s| s.flags & elf::SHF_ALLOC != 0 && s.name != ".text")
    {
        let t = new.section(&s.name).unwrap();
        assert_eq!((t.addr, t.size), (s.addr, s.size), "{}", s.name);
        assert_eq!(t.offset % elf::PAGE, s.offset % elf::PAGE, "{}", s.name);
        // .rodata (descriptor) and .dynsym (kernel symbol range) are patched on purpose.
        if ![".rodata", ".dynsym"].contains(&s.name.as_str()) && s.kind != elf::SHT_NOBITS {
            let r = |e: &elf::Elf, x: &elf::Section| {
                e.data[x.offset as usize..(x.offset + x.size) as usize].to_vec()
            };
            assert_eq!(r(&new, t), r(&original, s), "{}", s.name);
        }
    }
    let old_sym = original.symbol(&a.symbol).unwrap();
    for table in [".symtab", ".dynsym"] {
        let sym = new
            .symbols(table)
            .unwrap()
            .into_iter()
            .find(|s| s.name == a.symbol)
            .unwrap();
        assert_eq!(
            (sym.value, sym.value + sym.size),
            (m.rewrite.prologue_vaddr, old_sym.value + old_sym.size)
        );
    }
    let body = |e: &elf::Elf| {
        let o = e.vaddr_to_offset(text.addr).unwrap();
        e.data[o..o + text.size as usize].to_vec()
    };
    let changed = body(&new)
        .iter()
        .zip(body(&original))
        .filter(|(x, y)| *x != y)
        .count();
    assert!(changed > 0 && changed <= m.rewrite.workgroup_id_sites.len() * 2);
    let off = new.vaddr_to_offset(m.rewrite.prologue_vaddr).unwrap();
    let bytes = &new.data[off..off + m.rewrite.prologue_bytes as usize];
    let branch = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    let target = (m.rewrite.prologue_vaddr + m.rewrite.prologue_bytes) as i64
        + 4 * (branch as u16 as i16) as i64;
    assert_eq!(branch >> 16, 0xBFA0);
    assert_eq!(target as u64, a.kd.entry_vaddr);
}

fn sparse_manifest(locations: u64) -> manifest::Manifest {
    let mut m = manifest();
    m.workgroups = [3, 1, 1];
    m.num_workgroups = 3;
    m.num_keys = 1024;
    m.num_log_records = locations;
    m
}

/// Probe log over shader engines 0..se with WGPs 0..per_se of shader array 0: one first arrival per location.
fn probe_log(se: u32, per_se: u32) -> table::Log {
    (0..se)
        .flat_map(|s| (0..per_se).map(move |w| sparse_table::location(s, 0, w)))
        .enumerate()
        .map(|(i, loc)| [loc, 0, manifest::FAKE, 0, 1, 0, i as u32, 0])
        .collect()
}

#[test]
fn sparse_probe_assignment_and_placement() {
    let m = sparse_manifest(8);
    let log = probe_log(4, 2);
    let locs = sparse_table::validate_probe(&m, &[log.clone(), log.clone()], 4, 0).unwrap();
    assert_eq!(locs.len(), 8);
    // A second workgroup at a location, a changed WGP set, and a missing traced WGP all refuse.
    let mut twice = log.clone();
    twice[1][0] = twice[0][0];
    assert!(sparse_table::validate_probe(&m, &[twice], 4, 0).is_err());
    let mut moved = log.clone();
    moved[1][0] = sparse_table::location(0, 1, 3);
    assert!(sparse_table::validate_probe(&m, &[log.clone(), moved], 4, 0).is_err());
    assert!(sparse_table::validate_probe(&m, std::slice::from_ref(&log), 4, 5).is_err());

    // One per XCC first: shader engines 0, 2, then 1, 3.
    assert_eq!(sparse_table::engine_order(4), [0, 2, 1, 3]);
    let (assignment, traced) = sparse_table::assign(&locs, [3, 1, 1], 4, 0).unwrap();
    let at = |se| sparse_table::location(se, 0, 0);
    assert_eq!(traced, [at(0), at(2), at(1)]);
    assert_eq!(
        assignment
            .iter()
            .map(|a| (a.location, a.ttmp9))
            .collect::<Vec<_>>(),
        [(at(0), 0), (at(2), 1), (at(1), 2)]
    );
    // More workgroups than shader engines: the rest go to other WGPs, round-robin over shader engines.
    let (more, traced) = sparse_table::assign(&locs, [6, 1, 1], 4, 0).unwrap();
    assert_eq!(traced.len(), 4);
    assert_eq!(more[4].location, sparse_table::location(0, 0, 1));
    assert_eq!(more[5].location, sparse_table::location(2, 0, 1));
    // Multi-dimensional logical ids: ttmp7 = y | z << 16.
    let (md, _) = sparse_table::assign(&locs, [2, 2, 2], 4, 0).unwrap();
    assert_eq!((md[7].ttmp9, md[7].ttmp7), (1, 1 | 1 << 16));
    assert!(sparse_table::assign(&locs, [9, 1, 1], 4, 0).is_err());

    let entries = sparse_table::build_table(1024, &locs, &assignment).unwrap();
    assert_eq!(entries[at(2) as usize], [1, 0]);
    assert_eq!(
        entries[sparse_table::location(0, 0, 1) as usize],
        [manifest::FAKE, 0]
    );
    assert_eq!(entries[1000], [manifest::NO_ENTRY; 2]);

    let expected = |log: &table::Log| -> table::Log {
        log.iter()
            .map(|r| {
                let mut r = *r;
                if let Some(a) = assignment.iter().find(|a| a.location == r[0]) {
                    r[2] = a.ttmp9;
                    r[3] = a.ttmp7;
                }
                r
            })
            .collect()
    };
    let good = expected(&log);
    let p = sparse_table::check_placement(&m, &good, &locs, &assignment).unwrap();
    assert_eq!((p.placed, p.fakes, p.every_tile_once), (3, 5, true));
    let mut wrong = good.clone();
    wrong[0][2] = 2;
    assert!(sparse_table::check_placement(&m, &wrong, &locs, &assignment).is_err());
    let mut missing = good.clone();
    missing[0][2] = manifest::FAKE;
    assert!(sparse_table::check_placement(&m, &missing, &locs, &assignment).is_err());
    let mut second = good.clone();
    second[3][1] = 1;
    assert!(sparse_table::check_placement(&m, &second, &locs, &assignment).is_err());
}

#[test]
fn sparse_refuses_dispatch_packet_grid_reads() {
    if !tools_available() {
        return;
    }
    let mut a = analyze::analyze_code_object(&fixture("v8"), None).unwrap();
    // dispatch_ptr + kernarg_segment_ptr: the dispatch pointer is s[0:1].
    a.kd.code_properties = (a.kd.code_properties & !0x7f) | 0b1010;
    let check = |a: &mut analyze::Analysis, body: &str| {
        let listing: String = body
            .lines()
            .enumerate()
            .map(|(j, l)| format!("\t{l} // {:012X}: 00000000\n", j * 4))
            .collect();
        a.insts = isa::parse_objdump(&listing);
        a.dispatch_packet_grid_problems()
    };
    assert!(check(&mut a, "s_load_b32 s4, s[0:1], 0x4\ns_endpgm").is_empty());
    assert!(!check(&mut a, "s_load_b32 s4, s[0:1], 0xc\ns_endpgm").is_empty());
    assert!(!check(&mut a, "s_load_b128 s[4:7], s[0:1], 0x8\ns_endpgm").is_empty());
    assert!(!check(&mut a, "s_mov_b64 s[8:9], s[0:1]\ns_endpgm").is_empty());
    // Once the pointer registers are overwritten, later reads are not packet reads.
    assert!(check(
        &mut a,
        "s_mov_b64 s[0:1], 0\ns_load_b32 s4, s[0:1], 0xc\ns_endpgm"
    )
    .is_empty());
    a.kd.code_properties &= !0b10;
    assert!(check(&mut a, "s_load_b32 s4, s[0:1], 0xc\ns_endpgm").is_empty());
}

#[test]
fn sparse_prologue_assembles_and_cluster_refuses() {
    if !tools_available() {
        return;
    }
    for name in ["v8", "mxfp", "warp-pipeline-b"] {
        let code = fixture(name);
        let a = analyze::analyze_code_object(&code, None).unwrap();
        let mut opts = options(false);
        opts.launch_model = rewrite::LaunchModel::Sparse;
        opts.workgroups = [8, 1, 1];
        let dir = tempfile::tempdir().unwrap();
        let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
        let source = std::fs::read_to_string(dir.path().join("rewrite/prologue.s")).unwrap();
        assert!(
            source.contains(", 0x40010\n"),
            "{name}: XCC must come from 0x87 bits [19:16]"
        );
        assert!(
            source.contains("s_endpgm") && source.contains("0xa2000000"),
            "{name}"
        );
        let mut one_d = a.clone();
        one_d.zero_constant_ids(opts.workgroups);
        assert_eq!(m.rewrite.workgroup_id_sites.len(), one_d.sites.len());
        // The body prefetch sits just before the final branch and still targets the original entry.
        let new = std::fs::read(dir.path().join("rewrite/kernel.hsaco")).unwrap();
        let insts = isa::parse_objdump(&disassemble_bytes(&new).unwrap());
        let end = m.rewrite.prologue_vaddr + m.rewrite.prologue_bytes;
        let prefetch: Vec<_> = insts
            .iter()
            .filter(|i| {
                i.mnemonic == "s_prefetch_inst_pc_rel"
                    && i.addr >= m.rewrite.prologue_vaddr
                    && i.addr < end
            })
            .collect();
        assert_eq!(
            prefetch.len() as u32,
            a.kd.inst_pref_lines().div_ceil(31),
            "{name}"
        );
        let Some(&first) = prefetch.first() else {
            continue;
        };
        assert_eq!(
            prefetch.last().unwrap().end(),
            end - 4,
            "{name}: prefetch must end at the final branch"
        );
        let off = first.operand_list()[0].trim();
        let off = match off.strip_prefix("-0x") {
            Some(h) => -(i64::from_str_radix(h, 16).unwrap()),
            None => i64::from_str_radix(off.trim_start_matches("0x"), 16).unwrap(),
        };
        assert_eq!(
            (first.end() as i64 + off) as u64,
            a.kd.entry_vaddr,
            "{name}"
        );
    }
    let code = fixture("cluster2x2");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let mut opts = options(true);
    opts.launch_model = rewrite::LaunchModel::Sparse;
    let dir = tempfile::tempdir().unwrap();
    assert!(rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).is_err());
}

/// Rewrite `name` with and without loop markers; return both code objects, the marked manifest and the analysis.
fn marked_and_plain(
    name: &str,
    opts: &RewriteOptions,
    which: markers::LoopMarkers,
) -> (
    elf::Elf,
    elf::Elf,
    manifest::Manifest,
    manifest::Manifest,
    analyze::Analysis,
) {
    let code = fixture(name);
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let plain_dir = tempfile::tempdir().unwrap();
    let plain = rewrite::rewrite(&code, &a.symbol, opts, plain_dir.path()).unwrap();
    let marked_opts = RewriteOptions {
        loop_markers: Some(which),
        ..*opts
    };
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &marked_opts, dir.path()).unwrap();
    assert!(dir.path().join("rewrite/loop_markers.s").is_file());
    assert!(!plain_dir.path().join("rewrite/loop_markers.s").exists());
    let read = |d: &Path, m: &manifest::Manifest| {
        elf::Elf::parse(std::fs::read(d.join(&m.code_object)).unwrap()).unwrap()
    };
    (
        read(dir.path(), &m),
        read(plain_dir.path(), &plain),
        m,
        plain,
        a,
    )
}

#[test]
fn loop_markers_route_barriers_through_stubs_and_keep_every_other_byte() {
    if !tools_available() {
        return;
    }
    let mut fp8 = options(true);
    fp8.workgroups = [1024, 1, 1];
    fp8.cluster_size = 16;
    fp8.cluster_dims = [16, 1, 1];
    // gfxjit gemm_proto: 16x16 grid of 4x4 clusters, 4 waves per workgroup.
    let mut proto = options(true);
    proto.workgroups = [16, 16, 1];
    proto.waves_per_workgroup = 4;
    proto.cluster_size = 16;
    proto.cluster_dims = [4, 4, 1];
    let cases = [
        (
            "gemm-proto-4x4",
            proto,
            markers::LoopMarkers::Cluster,
            25usize,
        ),
        ("fp8-4x4", fp8, markers::LoopMarkers::Cluster, 16usize),
        (
            "cluster2x2",
            options(true),
            markers::LoopMarkers::Cluster,
            22,
        ),
        (
            "cluster2x2",
            options(true),
            markers::LoopMarkers::ClusterAndWorkgroup,
            94,
        ),
        // No slack after .text: prologue and stubs go below it.
        (
            "warp-pipeline-b",
            options(false),
            markers::LoopMarkers::ClusterAndWorkgroup,
            11,
        ),
    ];
    for (name, opts, which, expected_sites) in cases {
        let (new, plain, m, pm, a) = marked_and_plain(name, &opts, which);
        assert!(pm.rewrite.loop_markers.is_none());
        assert!(
            !serde_json::to_string(&pm).unwrap().contains("loop_markers"),
            "{name}: default manifest changed"
        );
        let info = m.rewrite.loop_markers.as_ref().unwrap();
        assert_eq!(info.sites.len(), expected_sites, "{name} {which:?}");
        assert_eq!(
            info.stubs_vaddr,
            m.rewrite.prologue_vaddr + m.rewrite.prologue_bytes
        );
        // No fixture body reads M0 (TDM takes descriptors; MSG_DEALLOC_VGPRS has no payload), so nothing is saved.
        assert_eq!(info.save_sgpr, None, "{name}");
        assert_eq!(
            info.stubs_bytes as usize,
            expected_sites * markers::stub_bytes(None)
        );
        // Same prologue as the unmarked rewrite (placed at the same address unless the stubs push it further down).
        let off = |e: &elf::Elf, v: u64| e.vaddr_to_offset(v).unwrap();
        let p = |e: &elf::Elf, mm: &manifest::Manifest| {
            let o = off(e, mm.rewrite.prologue_vaddr);
            e.data[o..o + mm.rewrite.prologue_bytes as usize].to_vec()
        };
        assert_eq!(m.rewrite.prologue_bytes, pm.rewrite.prologue_bytes);
        if m.rewrite.prologue_vaddr == pm.rewrite.prologue_vaddr {
            assert_eq!(p(&new, &m), p(&plain, &pm), "{name}: prologue changed");
        }
        // Body: identical to the unmarked rewrite except one s_branch per site.
        let (bo, po) = (off(&new, a.kernel_vaddr), off(&plain, a.kernel_vaddr));
        let body = &new.data[bo..bo + a.kernel_size as usize];
        let plain_body = &plain.data[po..po + a.kernel_size as usize];
        let changed: Vec<u64> = (0..body.len())
            .filter(|&i| body[i] != plain_body[i])
            .map(|i| (a.kernel_vaddr + i as u64) & !3)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let site_addrs: Vec<u64> = info.sites.iter().map(|s| s.addr).collect();
        assert!(
            changed.iter().all(|c| site_addrs.contains(c)),
            "{name}: bytes changed outside marker sites"
        );
        let listing = isa::parse_objdump(&disassemble_bytes(&new.data).unwrap());
        let at = |v: u64| listing.iter().find(|i| i.addr == v).unwrap();
        for s in &info.sites {
            let original = a.insts.iter().find(|i| i.addr == s.addr).unwrap();
            let expected_tag = match (original.mnemonic.as_str(), original.operands.as_str()) {
                ("s_barrier_signal", "-3") => markers::TAG_CLUSTER_SIGNAL,
                ("s_barrier_wait", "0xfffd") => markers::TAG_CLUSTER_WAIT,
                ("s_barrier_wait", "0xffff") => markers::TAG_WORKGROUP_WAIT,
                other => panic!("{name}: unexpected marker site {other:?}"),
            };
            assert_eq!(s.tag, expected_tag);
            let site = at(s.addr);
            assert_eq!(
                (site.mnemonic.as_str(), site.branch_target()),
                ("s_branch", Some(s.stub_vaddr))
            );
            let stub: Vec<_> = [0u64, 8, 12, 16]
                .iter()
                .map(|k| at(s.stub_vaddr + k))
                .collect();
            assert_eq!(
                stub[0].text(),
                format!(
                    "s_pack_ll_b32_b16 m0, ttmp6, {:#x}",
                    (s.tag as u32) << 8 | s.index
                )
            );
            assert_eq!(stub[1].text(), "s_ttracedata");
            assert_eq!(stub[2].bytes, original.bytes);
            assert_eq!(stub[3].branch_target(), Some(s.addr + 4));
        }
        // The kernel symbol covers prologue, stubs and body.
        let sym = new
            .symbols(".symtab")
            .unwrap()
            .into_iter()
            .find(|s| s.name == a.symbol)
            .unwrap();
        assert!(
            sym.value <= m.rewrite.prologue_vaddr
                && info.stubs_vaddr + info.stubs_bytes <= sym.value + sym.size
        );
        assert!(
            sym.value <= a.kernel_vaddr && a.kernel_vaddr + a.kernel_size <= sym.value + sym.size
        );
    }
}

#[test]
fn loop_markers_refuse_kernels_without_sites_or_a_free_sgpr() {
    if !tools_available() {
        return;
    }
    for name in ["v8", "warp-pipeline-b"] {
        let code = fixture(name);
        let a = analyze::analyze_code_object(&code, None).unwrap();
        let mut opts = options(false);
        opts.loop_markers = Some(markers::LoopMarkers::Cluster);
        let err = rewrite::rewrite(&code, &a.symbol, &opts, tempfile::tempdir().unwrap().path())
            .unwrap_err();
        assert!(
            err.to_string().contains("no matching barrier"),
            "{name}: {err}"
        );
    }
    // cluster2x2 references every SGPR below the reserved ones: fine while M0 is dead, refused once it may be live.
    let mut a = analyze::analyze_code_object(&fixture("cluster2x2"), None).unwrap();
    assert_eq!(markers::m0_save(&a).unwrap(), None);
    assert!(markers::save_sgpr(&a).is_err());
    let mut reader = a.insts[0].clone();
    reader.mnemonic = "s_sendmsg".into();
    reader.operands = "sendmsg(MSG_GS_ALLOC_REQ)".into();
    a.insts.push(reader);
    let err = markers::m0_save(&a).unwrap_err().to_string();
    assert!(
        err.contains("no SGPR is free") && err.contains("MSG_GS_ALLOC_REQ"),
        "{err}"
    );
    // With a free SGPR the stub saves M0 there; the highest unreferenced one is chosen.
    a.sgprs_used.remove(&90);
    a.sgprs_used.remove(&91);
    assert_eq!(markers::m0_save(&a).unwrap(), Some(91));
}

#[test]
fn m0_readers_are_found_and_payload_free_messages_exempt() {
    let listing = |text: &str| {
        isa::parse_objdump(
            &text
                .lines()
                .enumerate()
                .map(|(j, l)| format!("\t{l} // {:012X}: 00000000\n", j * 4))
                .collect::<String>(),
        )
    };
    for text in [
        "s_mov_b32 m0, s3",
        "s_barrier_signal m0",
        "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
        "global_load_block v[0:31], v0, s[0:1]",
        "cluster_load_b128 v[0:3], v0, s[0:1]",
        "s_movrels_b32 s0, s1",
        "s_ttracedata",
    ] {
        assert!(markers::m0_reader(&listing(text)).is_some(), "{text}");
    }
    for text in [
        "s_sendmsg_rtn_b32 s17, sendmsg(MSG_RTN_GET_SE_AID_ID)",
        "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)",
        "tensor_load_to_lds s[24:27], s[28:35], s[36:39], null",
        "s_mov_b32 s100, s10",
        "s_ttracedata_imm 0x1",
    ] {
        assert!(markers::m0_reader(&listing(text)).is_none(), "{text}");
    }
}

#[test]
fn saving_stubs_reproduce_the_original_and_branch_back() {
    if !tools_available() {
        return;
    }
    let a = analyze::analyze_code_object(&fixture("cluster2x2"), None).unwrap();
    let sites = markers::select_sites(&a, markers::LoopMarkers::Cluster).unwrap();
    let base = 0x10000;
    let (code, source, out) = markers::build_stubs(&a, &sites, Some(100), base).unwrap();
    let len = markers::stub_bytes(Some(100));
    assert_eq!((len, code.len()), (28, sites.len() * len));
    assert!(source.contains("s_mov_b32 s100, m0") && source.contains("s_mov_b32 m0, s100"));
    for (n, s) in out.iter().enumerate() {
        let o = n * len;
        let original = &a.insts[sites[n].0];
        assert_eq!(s.stub_vaddr, base + o as u64);
        assert_eq!(code[o + 20..o + 24], original.bytes[..]);
        let branch = u32::from_le_bytes(code[o + 24..o + 28].try_into().unwrap());
        assert_eq!(branch >> 16, 0xBFA0);
        let target = (s.stub_vaddr + 28) as i64 + 4 * (branch as u16 as i16) as i64;
        assert_eq!(target as u64, s.addr + 4);
    }
}

#[test]
fn admission_manifest_is_not_sparse() {
    let mut m = manifest();
    m.launch = Some(manifest::LaunchLayout {
        model: "admission".into(),
        padded_grid_size: [1024, 1, 1],
        padded_workgroups: 8,
        assignment: vec![],
        traced_locations: vec![],
        slack: 6,
    });
    assert!(m.is_admission() && !m.is_sparse());
    m.launch.as_mut().unwrap().model = "sparse".into();
    assert!(m.is_sparse() && !m.is_admission());
}

#[test]
fn admission_prologue_parks_and_manifest_carries_its_buffer() {
    if !tools_available() {
        return;
    }
    for name in ["v8", "mxfp", "warp-pipeline-b"] {
        let code = fixture(name);
        let a = analyze::analyze_code_object(&code, None).unwrap();
        let mut opts = options(false);
        opts.launch_model = rewrite::LaunchModel::Admission;
        opts.padded_workgroups = 256 + 4096;
        let dir = tempfile::tempdir().unwrap();
        let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
        let source = std::fs::read_to_string(dir.path().join("rewrite/prologue.s")).unwrap();
        for needle in [
            "global_atomic_sub_u32",
            "s_sleep 127",
            "0xa2000000",
            "s_endpgm",
            "scope:SCOPE_DEV",
            "s_cselect_b32",
        ] {
            assert!(source.contains(needle), "{name}: {needle}");
        }
        assert!(
            !m.is_sparse() && m.extra_buffers.len() == 1 && m.extra_buffers[0].name == "admission"
        );
        assert_eq!(
            m.extra_buffers[0].bytes,
            rewrite::ADMISSION_WORDS as u64 * 4
        );
        assert_eq!(m.num_log_records, 256);
        let mut one_d = a.clone();
        one_d.zero_constant_ids(opts.workgroups);
        assert_eq!(m.rewrite.workgroup_id_sites.len(), one_d.sites.len());
    }
    let code = fixture("v8");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let mut opts = options(false);
    opts.launch_model = rewrite::LaunchModel::Admission;
    opts.padded_workgroups = 256;
    assert!(
        rewrite::rewrite(&code, &a.symbol, &opts, tempfile::tempdir().unwrap().path()).is_err()
    );
    // Single-wave workgroups key by (location, SIMD).
    opts.padded_workgroups = 256 + 4096;
    opts.waves_per_workgroup = 1;
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
    let source = std::fs::read_to_string(dir.path().join("rewrite/prologue.s")).unwrap();
    assert_eq!(m.num_keys, 4096);
    assert!(source.contains(", 0x20008\n") && source.contains(", 7\n"));
    let code = fixture("cluster2x2");
    let a = analyze::analyze_code_object(&code, None).unwrap();
    let mut opts = options(true);
    opts.launch_model = rewrite::LaunchModel::Admission;
    opts.padded_workgroups = 1 << 14;
    // Cluster admission: the leader parks the whole cluster with cluster id 0xffff; slot and mailbox cover the
    // padded hardware clusters and the admission buffer follows them in the control block.
    let dir = tempfile::tempdir().unwrap();
    let m = rewrite::rewrite(&code, &a.symbol, &opts, dir.path()).unwrap();
    let source = std::fs::read_to_string(dir.path().join("rewrite/prologue.s")).unwrap();
    for needle in [
        "s_cmp_eq_u32 s",
        ", 0xffff\n",
        "global_atomic_sub_u32",
        "s_sleep 127",
        "0xa2000000",
        "s_endpgm",
    ] {
        assert!(source.contains(needle), "cluster: {needle}");
    }
    let names: Vec<_> = m
        .extra_buffers
        .iter()
        .map(|x| (x.name.as_str(), x.control_offset, x.bytes))
        .collect();
    assert_eq!(
        names,
        [
            ("slot", 40, 4096 * 4),
            ("mailbox", 48, 4096 * 4),
            ("admission", 56, rewrite::ADMISSION_WORDS as u64 * 4)
        ]
    );
    assert_eq!(m.num_log_records, 128);
    opts.padded_workgroups = (1 << 14) + 2;
    assert!(
        rewrite::rewrite(&code, &a.symbol, &opts, tempfile::tempdir().unwrap().path()).is_err()
    );
    let x: manifest::ExtraBuffer = serde_json::from_value(
        serde_json::json!({"name":"a","control_offset":40,"bytes":8,"fill":7}),
    )
    .unwrap();
    assert!(x.init.is_none());
}
