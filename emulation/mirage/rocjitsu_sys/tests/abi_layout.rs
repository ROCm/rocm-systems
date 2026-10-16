//! Optional cross-check of the FFI layouts against the real C headers.
//!
//! `src/lib.rs` carries a block of `const _: () = assert!(...)` layout
//! assertions. Those run on every build and cost nothing, but they only
//! encode what a human read off the header once: they catch a careless
//! edit to the *Rust* side, and are blind to the *C* side changing.
//!
//! This test closes that gap when it can. It finds the rocjitsu headers
//! and a C compiler, generates a probe that prints `sizeof` and
//! `offsetof` for each type and field *by name*, compiles it against the
//! real headers, and compares the result with the Rust layouts. A field
//! that was renamed, reordered, retyped or appended on the C side shows
//! up here — an appended one as a size mismatch, which is exactly how
//! `rj_vm_map_t`'s `map_errno` went unnoticed.
//!
//! It is deliberately best-effort. mirage does not build rocjitsu, so on
//! most checkouts the headers are simply absent — and a `cc` is not
//! guaranteed either. Either one missing makes this test print why and
//! return, rather than fail; nothing in this crate's build depends on
//! the headers existing anywhere.
//!
//! To point it at a specific install: `ROCJITSU_INCLUDE_DIR=/path/to/include`
//! (the directory that *contains* `rocjitsu/vm/rj_vm.h`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::mem::{MaybeUninit, offset_of};
use std::path::{Path, PathBuf};
use std::process::Command;

use rocjitsu_sys::{
    RjDaemonStatus, RjHandle, RjStatus, RjVmCmd, RjVmGpuInfo, RjVmMap, RjVmMode, RjVmUnmap,
};

/// One Rust field and the C field it is supposed to sit on top of.
#[derive(Debug)]
struct Field {
    /// Field name in the C struct — and in the Rust struct, which is
    /// kept identical on purpose so a rename on either side is caught.
    name: &'static str,
    offset: usize,
    size: usize,
}

/// A C type to interrogate, and what Rust currently believes about it.
#[derive(Debug)]
struct Ty {
    /// The C spelling, e.g. `rj_vm_map_t`. Used verbatim in the probe.
    c_name: &'static str,
    size: usize,
    /// Empty for scalar typedefs, where only the width is meaningful.
    fields: Vec<Field>,
}

/// The width of a pointee type, without needing a value of it.
fn width_of<T>(_: *const T) -> usize {
    size_of::<T>()
}

/// Describe one field: its name, its offset in the Rust struct, and its
/// width. The width matters on its own — a last field widened inside
/// existing tail padding changes neither the offsets nor the struct size.
macro_rules! field {
    ($ty:ty, $name:ident) => {{
        let uninit = MaybeUninit::<$ty>::uninit();
        let base = uninit.as_ptr();
        Field {
            name: stringify!($name),
            offset: offset_of!($ty, $name),
            // SAFETY: `addr_of!` computes an address inside an allocated
            // (if uninitialised) object without reading it, and
            // `width_of` only inspects the pointee's *type*.
            size: width_of(unsafe { std::ptr::addr_of!((*base).$name) }),
        }
    }};
}

fn types() -> Vec<Ty> {
    vec![
        Ty {
            c_name: "rj_vm_cmd_t",
            size: size_of::<RjVmCmd>(),
            fields: vec![
                field!(RjVmCmd, cmd),
                field!(RjVmCmd, buf),
                field!(RjVmCmd, buf_size),
                field!(RjVmCmd, result),
                field!(RjVmCmd, shared_handle),
                field!(RjVmCmd, in_handle),
            ],
        },
        Ty {
            c_name: "rj_vm_map_t",
            size: size_of::<RjVmMap>(),
            fields: vec![
                field!(RjVmMap, addr),
                field!(RjVmMap, length),
                field!(RjVmMap, offset),
                field!(RjVmMap, prot),
                field!(RjVmMap, flags),
                field!(RjVmMap, mapped_addr),
                field!(RjVmMap, map_errno),
            ],
        },
        Ty {
            c_name: "rj_vm_unmap_t",
            size: size_of::<RjVmUnmap>(),
            fields: vec![field!(RjVmUnmap, addr), field!(RjVmUnmap, length)],
        },
        // Listed field by field rather than sampled: this struct is the
        // one that goes out on the daemon handshake wire, so a mismatch
        // makes a client misread GPU metadata rather than crash, which
        // is far harder to notice.
        Ty {
            c_name: "rj_vm_gpu_info_t",
            size: size_of::<RjVmGpuInfo>(),
            fields: vec![
                field!(RjVmGpuInfo, present),
                field!(RjVmGpuInfo, gpu_id),
                field!(RjVmGpuInfo, gfx_target_version),
                field!(RjVmGpuInfo, vendor_id),
                field!(RjVmGpuInfo, device_id),
                field!(RjVmGpuInfo, family_id),
                field!(RjVmGpuInfo, unique_id),
                field!(RjVmGpuInfo, location_id),
                field!(RjVmGpuInfo, domain),
                field!(RjVmGpuInfo, hive_id),
                field!(RjVmGpuInfo, drm_render_minor),
                field!(RjVmGpuInfo, revision_id),
                field!(RjVmGpuInfo, pci_revision_id),
                field!(RjVmGpuInfo, simd_count),
                field!(RjVmGpuInfo, max_waves_per_simd),
                field!(RjVmGpuInfo, num_shader_engines),
                field!(RjVmGpuInfo, num_shader_arrays_per_engine),
                field!(RjVmGpuInfo, num_cu_per_sh),
                field!(RjVmGpuInfo, simd_per_cu),
                field!(RjVmGpuInfo, wave_front_size),
                field!(RjVmGpuInfo, num_xcc),
                field!(RjVmGpuInfo, max_slots_scratch_cu),
                field!(RjVmGpuInfo, local_mem_size),
                field!(RjVmGpuInfo, vram_type),
                field!(RjVmGpuInfo, lds_size_kb),
                field!(RjVmGpuInfo, mem_width),
                field!(RjVmGpuInfo, mem_clk_max),
                field!(RjVmGpuInfo, l1_size_kb),
                field!(RjVmGpuInfo, l1_line_size),
                field!(RjVmGpuInfo, l1_assoc),
                field!(RjVmGpuInfo, l2_size_kb),
                field!(RjVmGpuInfo, l2_line_size),
                field!(RjVmGpuInfo, l2_assoc),
                field!(RjVmGpuInfo, num_sdma_engines),
                field!(RjVmGpuInfo, num_sdma_xgmi_engines),
                field!(RjVmGpuInfo, num_cp_queues),
                field!(RjVmGpuInfo, max_engine_clk_fcompute),
                field!(RjVmGpuInfo, capability),
                field!(RjVmGpuInfo, capability2),
                field!(RjVmGpuInfo, debug_prop),
                field!(RjVmGpuInfo, fw_version),
                field!(RjVmGpuInfo, sdma_fw_version),
                field!(RjVmGpuInfo, marketing_name),
            ],
        },
        // Scalar typedefs: only the width is meaningful, but a width
        // change is exactly as ABI-breaking as a moved field.
        Ty {
            c_name: "rj_status_t",
            size: size_of::<RjStatus>(),
            fields: Vec::new(),
        },
        Ty {
            c_name: "rj_handle_t",
            size: size_of::<RjHandle>(),
            fields: Vec::new(),
        },
        Ty {
            c_name: "rj_client_pid_t",
            size: size_of::<i32>(),
            fields: Vec::new(),
        },
        Ty {
            c_name: "rj_vm_mode_t",
            size: size_of::<RjVmMode>(),
            fields: Vec::new(),
        },
        Ty {
            c_name: "rj_daemon_status_t",
            size: size_of::<RjDaemonStatus>(),
            fields: Vec::new(),
        },
    ]
}

/// Find the directory that contains `rocjitsu/vm/rj_vm.h`.
///
/// `ROCJITSU_INCLUDE_DIR` wins if set. Otherwise: the conventional ROCm
/// install locations, then a walk up from this crate towards the
/// filesystem root looking for a rocjitsu source tree beside mirage —
/// which is how a developer with both checked out will have it, without
/// anyone's absolute path being written down here.
fn locate_include_dir() -> Option<PathBuf> {
    // Both headers, because the probe includes both; a tree with only
    // one of them should skip rather than fail to compile.
    let usable = |dir: &Path| {
        dir.join("rocjitsu/vm/rj_vm.h").is_file()
            && dir.join("rocjitsu/daemon/rj_daemon.h").is_file()
    };

    if let Some(dir) = std::env::var_os("ROCJITSU_INCLUDE_DIR").filter(|v| !v.is_empty()) {
        let dir = PathBuf::from(dir);
        return usable(&dir).then_some(dir);
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(root) = std::env::var_os("ROCM_HOME").filter(|v| !v.is_empty()) {
        candidates.push(PathBuf::from(root).join("include"));
    }
    candidates.push(PathBuf::from("/opt/rocm/include"));

    // `.../mirage/rocjitsu_sys` -> `.../mirage` -> `...` -> ...
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        candidates.push(ancestor.join("rocjitsu/lib/rocjitsu/include"));
    }

    candidates.into_iter().find(|dir| usable(dir))
}

/// Pick a C compiler that actually runs.
fn locate_cc() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(cc) = std::env::var_os("CC").filter(|v| !v.is_empty()) {
        candidates.push(cc.to_string_lossy().into_owned());
    }
    candidates.extend(["cc", "gcc", "clang"].map(String::from));

    candidates.into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success())
    })
}

/// Generate a probe printing one `T` line per type and one `F` line per
/// field.
fn probe_source(types: &[Ty]) -> String {
    let mut src = String::from(
        "#include <stdio.h>\n\
         #include <stddef.h>\n\
         #include \"rocjitsu/vm/rj_vm.h\"\n\
         #include \"rocjitsu/daemon/rj_daemon.h\"\n\
         \n\
         int main(void) {\n",
    );
    for ty in types {
        let c = ty.c_name;
        src.push_str(&format!("  printf(\"T {c} %zu\\n\", sizeof({c}));\n"));
        for f in &ty.fields {
            let n = f.name;
            src.push_str(&format!(
                "  printf(\"F {c} {n} %zu %zu\\n\", offsetof({c}, {n}), sizeof((({c} *)0)->{n}));\n"
            ));
        }
    }
    src.push_str("  return 0;\n}\n");
    src
}

#[test]
fn rust_layouts_match_the_rocjitsu_headers() {
    let Some(include_dir) = locate_include_dir() else {
        eprintln!(
            "no rocjitsu headers found (set ROCJITSU_INCLUDE_DIR to the directory \
             containing rocjitsu/vm/rj_vm.h); skipping the ABI layout cross-check"
        );
        return;
    };
    let Some(cc) = locate_cc() else {
        eprintln!("no C compiler found (set CC); skipping the ABI layout cross-check");
        return;
    };
    eprintln!(
        "checking FFI layouts against {} with {cc}",
        include_dir.display()
    );

    let types = types();
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("probe.c");
    let bin = dir.path().join("probe");
    std::fs::write(&src, probe_source(&types)).expect("write probe");

    let build = Command::new(&cc)
        .arg("-I")
        .arg(&include_dir)
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .output()
        .expect("run compiler");
    assert!(
        build.status.success(),
        "the layout probe failed to compile against {}:\n{}",
        include_dir.display(),
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&bin).output().expect("run probe");
    assert!(run.status.success(), "probe exited with {}", run.status);
    let stdout = String::from_utf8(run.stdout).expect("probe output is utf-8");

    let mut c_sizes: BTreeMap<&str, usize> = BTreeMap::new();
    let mut c_fields: BTreeMap<(&str, &str), (usize, usize)> = BTreeMap::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["T", ty, size] => {
                c_sizes.insert(*ty, size.parse().expect("type size"));
            }
            ["F", ty, name, offset, size] => {
                c_fields.insert(
                    (*ty, *name),
                    (
                        offset.parse().expect("field offset"),
                        size.parse().expect("field size"),
                    ),
                );
            }
            _ => panic!("unparseable probe output line: {line:?}"),
        }
    }

    // Collect every disagreement before failing: seeing all of them at
    // once is what tells you whether a field moved or the whole struct
    // was reshaped.
    let mut problems: Vec<String> = Vec::new();
    for ty in &types {
        let c_size = c_sizes
            .get(ty.c_name)
            .copied()
            .expect("probe printed a size for every type");
        // A field appended in C — the `map_errno` case — surfaces here
        // and nowhere else, since a per-field comparison can only see
        // the fields Rust already knows to ask about.
        if c_size != ty.size {
            problems.push(format!(
                "{}: Rust says {} bytes, the header says {c_size}",
                ty.c_name, ty.size
            ));
        }
        for f in &ty.fields {
            let Some(&c_field) = c_fields.get(&(ty.c_name, f.name)) else {
                problems.push(format!("{}.{}: no such field in C", ty.c_name, f.name));
                continue;
            };
            if c_field != (f.offset, f.size) {
                problems.push(format!(
                    "{}.{}: Rust says offset {} size {}, the header says offset {} size {}",
                    ty.c_name, f.name, f.offset, f.size, c_field.0, c_field.1
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "the Rust FFI types disagree with the headers in {}:\n  {}",
        include_dir.display(),
        problems.join("\n  ")
    );
}
