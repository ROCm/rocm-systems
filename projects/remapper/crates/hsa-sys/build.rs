fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HSA_RUNTIME_LIB_DIR");
    let dir = std::env::var("HSA_RUNTIME_LIB_DIR").unwrap_or_else(|_| {
        let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        format!("{rocm}/lib")
    });
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=hsa-runtime64");
}
