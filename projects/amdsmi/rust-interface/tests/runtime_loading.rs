// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#![cfg(feature = "dynamic-loading")]

use amdsmi::{AmdsmiInitFlagsT, AmdsmiMemoryTypeT, AmdsmiStatusT};
use std::path::Path;
use std::process::Command;

fn mock_library() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/runtime_library.c");
    let output = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(source)
        .arg("-o")
        .arg(directory.path().join("libamd_smi.so"))
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    directory
}

fn run_case(case: &str, library_directory: &Path) {
    // Each child gets its own process-wide loader state, without modifying the
    // environment of the parallel test runner.
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "runtime_loading_child",
            "--ignored",
            "--nocapture",
        ])
        .env("AMDSMI_TEST_CASE", case)
        .env("AMDSMI_LIB_DIR", library_directory)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}

#[test]
fn missing_library_returns_an_error_without_preventing_startup() {
    let directory = tempfile::tempdir().unwrap();
    run_case("missing_library", directory.path());
}

#[test]
fn missing_symbol_does_not_prevent_other_queries() {
    let library = mock_library();
    run_case("missing_symbol", library.path());
}

#[test]
fn queries_preserve_arguments_outputs_and_statuses() {
    let library = mock_library();
    run_case("queries", library.path());
}

#[test]
fn empty_library_directory_is_treated_as_unset() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("libamd_smi.so"), "not a library").unwrap();
    let library = mock_library();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "runtime_loading_child",
            "--ignored",
            "--nocapture",
        ])
        .current_dir(directory.path())
        .env("AMDSMI_TEST_CASE", "missing_symbol")
        .env("AMDSMI_LIB_DIR", "")
        .env("LD_LIBRARY_PATH", library.path())
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}

#[test]
#[ignore = "run by the parent tests with an isolated library configuration"]
fn runtime_loading_child() {
    match std::env::var("AMDSMI_TEST_CASE").unwrap().as_str() {
        "missing_library" => {
            let error = amdsmi::amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).unwrap_err();
            assert_eq!(error, AmdsmiStatusT::AmdsmiStatusFailLoadModule);
            assert_eq!(error.to_string(), "Failed to load libamd_smi.so");
        }
        "missing_symbol" => {
            assert!(matches!(
                amdsmi::amdsmi_get_gpu_device_bdf(std::ptr::null_mut()),
                Err(AmdsmiStatusT::AmdsmiStatusFailLoadSymbol)
            ));
            assert_eq!(
                AmdsmiStatusT::AmdsmiStatusFailLoadSymbol.to_string(),
                "Failed to load an AMD SMI symbol"
            );
            amdsmi::amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).unwrap();
            amdsmi::amdsmi_shut_down().unwrap();
        }
        "queries" => {
            assert_eq!(
                amdsmi::amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAllProcessors).unwrap_err(),
                AmdsmiStatusT::AmdsmiStatusInval
            );
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    std::thread::spawn(|| {
                        for _ in 0..10 {
                            amdsmi::amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus).unwrap();
                            assert_eq!(
                                amdsmi::amdsmi_get_gpu_memory_total(
                                    std::ptr::null_mut(),
                                    AmdsmiMemoryTypeT::AmdsmiMemTypeVram,
                                )
                                .unwrap(),
                                123456
                            );
                            amdsmi::amdsmi_shut_down().unwrap();
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        }
        case => panic!("unknown test case: {case}"),
    }
}
