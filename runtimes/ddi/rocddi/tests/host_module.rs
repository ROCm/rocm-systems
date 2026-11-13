// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native module ownership tests with a tiny independently linked fixture.
#![allow(unsafe_code, clippy::unwrap_used)]

use rocddi::host::module::ResidentModule;
use std::process::Command;

#[test]
fn acquired_modules_keep_symbols_resident_after_reference_release() {
    let directory = std::env::temp_dir().join(format!("rocddi-module-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("fixture.c");
    std::fs::write(
        &source,
        br"
#ifdef _WIN32
__declspec(dllexport)
#endif
int rocddi_module_fixture(void) { return 731; }
",
    )
    .unwrap();
    #[cfg(target_os = "linux")]
    let library = directory.join("libfixture.so");
    #[cfg(windows)]
    let library = directory.join("fixture.dll");
    let compiler = std::env::var_os("CC").unwrap_or_else(|| {
        if cfg!(windows) {
            "cl".into()
        } else {
            "cc".into()
        }
    });
    let mut command = Command::new(compiler);
    command.current_dir(&directory);
    #[cfg(target_os = "linux")]
    command
        .args(["-shared", "-fPIC", "-o"])
        .arg(&library)
        .arg(&source);
    #[cfg(windows)]
    command
        .args(["/nologo", "/LD"])
        .arg(&source)
        .arg(format!("/Fe:{}", library.display()));
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "fixture build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // SAFETY: This test built the fixture and knows its only exported C ABI.
    unsafe {
        assert!(ResidentModule::acquire_loaded(library.as_os_str()).is_err());
        let module = ResidentModule::load(library.as_os_str()).unwrap();
        let acquired = ResidentModule::acquire_loaded(library.as_os_str()).unwrap();
        assert_eq!(module.id(), acquired.id());
        let symbol = module.symbol(c"rocddi_module_fixture").unwrap().unwrap();
        let function: unsafe extern "C" fn() -> i32 = std::mem::transmute(symbol.as_ptr());
        let missing = module.symbol(c"rocddi_module_missing_symbol");
        assert!(missing.is_err() || missing.unwrap().is_none());
        drop(module);
        drop(acquired);
        assert_eq!(function(), 731);
        let still_loaded = ResidentModule::acquire_loaded(library.as_os_str()).unwrap();
        assert_eq!(
            still_loaded
                .symbol(c"rocddi_module_fixture")
                .unwrap()
                .unwrap(),
            symbol
        );
    }
    // Windows keeps a pinned DLL file open until this test process exits.
    let _ = std::fs::remove_dir_all(directory);
}
