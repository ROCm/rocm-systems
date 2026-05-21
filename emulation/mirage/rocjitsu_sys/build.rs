use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rocjitsu_root = env_path("ROCJITSU_ROOT")
        .unwrap_or_else(|| manifest_dir.join("..").join("..").join("rocjitsu"));
    let include_dir = env_path("ROCJITSU_INCLUDE_DIR")
        .unwrap_or_else(|| rocjitsu_root.join("lib").join("rocjitsu").join("include"));
    let schema_dir =
        env_path("ROCJITSU_SCHEMA_DIR").unwrap_or_else(|| rocjitsu_root.join("schemas"));

    emit_rerun_directives(&include_dir, &schema_dir);
    emit_link_directives(&rocjitsu_root);
    generate_bindings(&manifest_dir, &out_dir, &include_dir);
    generate_flatbuffers(&rocjitsu_root, &out_dir, &schema_dir);

    println!(
        "cargo:rustc-env=ROCJITSU_SYS_SCHEMA_DIR={}",
        schema_dir.display()
    );
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).map(PathBuf::from)
}

fn emit_rerun_directives(include_dir: &Path, schema_dir: &Path) {
    for name in [
        "ROCJITSU_ROOT",
        "ROCJITSU_INCLUDE_DIR",
        "ROCJITSU_SCHEMA_DIR",
        "ROCJITSU_LIB_DIR",
        "FLATC",
        "LIBCLANG_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    for header in [
        "rocjitsu/rocjitsu.h",
        "rocjitsu/base/api.h",
        "rocjitsu/base/rj_compiler.h",
        "rocjitsu/base/rj_status.h",
        "rocjitsu/code/api.h",
        "rocjitsu/code/rj_code.h",
        "rocjitsu/vm/api.h",
        "rocjitsu/vm/rj_vm.h",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            include_dir.join(header).display()
        );
    }

    for schema in ["simulation_config.fbs", "checkpoint.fbs"] {
        println!(
            "cargo:rerun-if-changed={}",
            schema_dir.join(schema).display()
        );
    }

    println!("cargo:rerun-if-changed=wrapper.h");
}

fn emit_link_directives(rocjitsu_root: &Path) {
    if let Some(lib_dir) = env_path("ROCJITSU_LIB_DIR") {
        emit_link_search(&lib_dir);
    }

    for lib_dir in [
        rocjitsu_root.join("build"),
        rocjitsu_root.join("build").join("lib"),
        rocjitsu_root.join("build-clean"),
        rocjitsu_root.join("build-clean").join("lib"),
        rocjitsu_root.join("artifacts").join("lib"),
    ] {
        if lib_dir.exists() {
            emit_link_search(&lib_dir);
        }
    }

    if let Some(rocm_path) = env_path("ROCM_PATH") {
        let rocm_lib_dir = rocm_path.join("lib");
        if rocm_lib_dir.exists() {
            emit_link_search(&rocm_lib_dir);
        }
    }

    println!("cargo:rustc-link-lib=dylib=rocjitsu");
}

fn emit_link_search(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!(
            "cargo:rustc-link-arg-tests=-Wl,-rpath,{}",
            lib_dir.display()
        );
    }
}

fn generate_bindings(manifest_dir: &Path, out_dir: &Path, include_dir: &Path) {
    let wrapper = manifest_dir.join("wrapper.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++20")
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("rj_.*")
        .allowlist_type("rj_.*")
        .allowlist_var("RJ_.*")
        .allowlist_var("ROCJITSU_.*")
        .rustified_enum("rj_status_e")
        .rustified_enum("rj_code_arch_e")
        .rustified_enum("rj_code_target_id_t")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap_or_else(|err| panic!("failed to generate rocjitsu bindings: {err}"));

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap_or_else(|err| panic!("failed to write rocjitsu bindings: {err}"));
}

fn generate_flatbuffers(rocjitsu_root: &Path, out_dir: &Path, schema_dir: &Path) {
    let fbs_out = out_dir.join("fbs");
    fs::create_dir_all(&fbs_out)
        .unwrap_or_else(|err| panic!("failed to create FlatBuffers output dir: {err}"));

    let flatc = find_flatc(rocjitsu_root);
    let simulation_schema = schema_dir.join("simulation_config.fbs");
    let checkpoint_schema = schema_dir.join("checkpoint.fbs");
    let status = Command::new(&flatc)
        .arg("--rust")
        .arg("--rust-module-root-file")
        .arg("-I")
        .arg(schema_dir)
        .arg("-o")
        .arg(&fbs_out)
        .arg(&simulation_schema)
        .arg(&checkpoint_schema)
        .status()
        .unwrap_or_else(|err| panic!("failed to run flatc at {}: {err}", flatc.display()));

    if !status.success() {
        panic!("flatc failed with status {status}");
    }
}

fn find_flatc(rocjitsu_root: &Path) -> PathBuf {
    if let Some(flatc) = env_path("FLATC") {
        return flatc;
    }

    for flatc in [
        rocjitsu_root
            .join("build")
            .join("_deps")
            .join("flatbuffers-build")
            .join("flatc"),
        rocjitsu_root
            .join("third_party")
            .join("flatbuffers-build")
            .join("flatc"),
    ] {
        if flatc.exists() {
            return flatc;
        }
    }

    if Command::new("flatc").arg("--version").output().is_ok() {
        return PathBuf::from("flatc");
    }

    panic!(
        "flatc was not found; set FLATC or build rocjitsu so its FlatBuffers compiler is available"
    );
}
