//! Native shared-library identity for the AMDF C ABI.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_RUNTIME_AMDF_SONAME");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let soname =
            env::var("ROCM_RUNTIME_AMDF_SONAME").unwrap_or_else(|_| "libamdf.so.0".to_owned());
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
    }
}
