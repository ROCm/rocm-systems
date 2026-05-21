use std::error::Error;
use std::path::PathBuf;

use rocjitsu_sys::{Decoder, Vm, fb, ffi};

const S_NOP: u32 = 0xBF80_0000;

#[test]
fn generated_flatbuffers_round_trip_minimal_config() -> Result<(), Box<dyn Error>> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let exec_mode = builder.create_string("functional");
    let config = fb::SimulationConfig::create(
        &mut builder,
        &fb::SimulationConfigArgs {
            max_ticks: 7,
            num_threads: 1,
            exec_mode: Some(exec_mode),
            ..Default::default()
        },
    );
    fb::finish_simulation_config_buffer(&mut builder, config);

    let parsed = fb::root_as_simulation_config(builder.finished_data())?;
    assert_eq!(parsed.max_ticks(), 7);
    assert_eq!(parsed.num_threads(), 1);
    assert_eq!(parsed.exec_mode(), Some("functional"));

    Ok(())
}

#[test]
fn decodes_known_instruction_through_rocjitsu() -> Result<(), Box<dyn Error>> {
    let mut decoder = Decoder::create(ffi::rj_code_arch_e::ROCJITSU_CODE_ARCH_CDNA4)?;
    let inst = decoder.decode(&S_NOP)?;

    assert_eq!(inst.mnemonic(), Some("s_nop"));
    assert_eq!(inst.size(), 4);
    assert!(inst.disassemble()?.contains("s_nop"));

    Ok(())
}

#[test]
fn creates_vm_from_repo_config() -> Result<(), Box<dyn Error>> {
    let config = rocjitsu_root().join("configs/amdgpu_cdna3.json");
    assert!(config.exists(), "missing test config: {}", config.display());

    let mut vm = Vm::create_with_default_schema(&config)?;
    assert!(!vm.step()?);

    Ok(())
}

fn rocjitsu_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("rocjitsu")
}
