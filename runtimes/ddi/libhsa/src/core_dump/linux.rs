// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux ELF emission and `ROCr` file/pipe output policy.

use super::pipe::write_pipe;
use rocddi::device::gpu::diagnostic::{
    CoreDeviceInfo, CoreDeviceSnapshot, CoreQueueSnapshot, DiagnosticMemory, DiagnosticReader,
};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Converts native diagnostic failures without losing their cause.
fn native(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

/// A load interval's final position in the ELF stream.
struct Load<'a> {
    /// Borrow of the retained native or loader memory.
    memory: DiagnosticMemory<'a>,
    /// File offset of the captured bytes.
    offset: u64,
    /// Zero for private VRAM that has no post-fault CPU reader.
    file_size: u64,
}

/// Selects output before capture; spawning a handler never holds HSA locks.
pub(super) fn dump() -> io::Result<()> {
    let (pattern, size_limit) = {
        let guard = crate::lifecycle::lock()
            .map_err(|status| native(format!("registry status {status}")))?;
        let runtime = guard
            .as_ref()
            .ok_or_else(|| native("no active HSA session"))?;
        if runtime.config.core_dump.disabled {
            return Ok(());
        }
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: The kernel writes one live rlimit structure.
        if unsafe { libc::getrlimit(libc::RLIMIT_CORE, &raw mut limit) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if limit.rlim_cur == 0 {
            return Ok(());
        }
        (
            output_pattern(runtime.config.core_dump.pattern.as_deref())?,
            limit.rlim_cur,
        )
    };
    if let Some(command) = pattern.strip_prefix('|') {
        // pre_exec uses fork, which can invoke application pthread_atfork
        // handlers. Such handlers may call HSA and must see an unlocked registry.
        write_pipe(command, |writer| capture(writer, u64::MAX))
    } else {
        write_file(&pattern, size_limit, capture)
    }
}

/// Applies file limits and removes incomplete output before fatal termination.
fn write_file(
    path: &str,
    core_limit: u64,
    emit: impl FnOnce(&mut dyn Write, u64) -> io::Result<()>,
) -> io::Result<()> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: The kernel writes one live rlimit structure. Crossing this limit
    // in write_all could deliver SIGXFSZ before cleanup and SIGABRT are reached.
    if unsafe { libc::getrlimit(libc::RLIMIT_FSIZE, &raw mut limit) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let size_limit = core_limit.min(limit.rlim_cur);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    let result = emit(&mut file, size_limit).and_then(|()| file.flush());
    if result.is_err() {
        let _ = std::fs::remove_file(path);
    }
    result
}

/// Retains every owner under the registry until actual memory has been emitted.
fn capture(writer: &mut dyn Write, size_limit: u64) -> io::Result<()> {
    let mut guard =
        crate::lifecycle::lock().map_err(|status| native(format!("registry status {status}")))?;
    let runtime = guard
        .as_mut()
        .ok_or_else(|| native("no active HSA session"))?;
    let devices = runtime
        .agents
        .gpus
        .iter()
        .map(|gpu| {
            gpu.device
                .gpu()
                .and_then(|gpu| gpu.core_snapshot())
                .map_err(native)
        })
        .collect::<io::Result<Vec<_>>>()?;
    if devices.is_empty() {
        return Err(native("no GPU state to capture"));
    }
    // Stop every queue before reading any wave or backing state.
    for queue in runtime.queue_state.queues.values_mut() {
        queue.native.inactivate().map_err(native)?;
        queue.inactivated = true;
    }
    let mut queues = Vec::new();
    let mut memory = Vec::new();
    let mut ids = HashSet::new();
    for queue in runtime.queue_state.queues.values_mut() {
        let (snapshot, scratch) = queue.core_snapshot().map_err(native)?;
        if ids.insert((snapshot.gpu_id, snapshot.queue_id)) {
            memory.extend(snapshot.memory);
            memory.extend(scratch);
            queues.push(snapshot);
        }
    }
    for allocation in runtime
        .memory_state
        .allocations
        .values()
        .chain(runtime.memory_state.ipc_allocations.values())
        .chain(runtime.memory_state.interop_allocations.values())
    {
        memory.push(allocation.allocation.diagnostic_memory().map_err(native)?);
    }
    for locked in &runtime.memory_state.locked_allocations {
        if locked.retired {
            continue;
        }
        if let Some(allocation) = &locked.allocation {
            memory.push(allocation.diagnostic_memory().map_err(native)?);
        }
    }
    for mapping in runtime.memory_state.vmem_mappings.values() {
        memory.push(if let Some(host) = &mapping.host_mapping {
            host.diagnostic_memory().map_err(native)?
        } else {
            // SAFETY: No CPU address is read; the registry retains this GPU
            // interval and records it explicitly as unavailable in the core.
            unsafe {
                DiagnosticMemory::borrow(mapping, mapping.address as u64, None, mapping.size as u64)
            }
            .map_err(native)?
        });
    }
    for reservation in runtime.memory_state.vmem_reservations.values() {
        for range in managed_svm_memory(&reservation.svm_ranges) {
            memory.push(range?);
        }
    }
    for executable in runtime.loader.executables.values() {
        memory.extend(executable.debugger.diagnostic_memory().map_err(native)?);
        for object in executable.core.loaded() {
            memory.extend(object.image.diagnostic_memories().map_err(native)?);
        }
    }
    memory.push(crate::debugger::diagnostic_record().map_err(native)?);
    memory.sort_by_key(|memory| (memory.address(), memory.size(), memory.host_address()));
    memory.dedup_by_key(|memory| (memory.address(), memory.size(), memory.host_address()));
    let note = encode_note(&devices, &queues)?;
    emit_elf(writer, &note, memory, size_limit)
}

/// Preflights the complete ELF before emitting any bytes to the output.
fn emit_elf(
    writer: &mut dyn Write,
    note: &[u8],
    memory: Vec<DiagnosticMemory<'_>>,
    size_limit: u64,
) -> io::Result<()> {
    let (headers, loads, total) = layout(note, memory)?;
    if total > size_limit {
        return Err(native(format!(
            "capture requires {total} bytes, exceeding effective file limit {size_limit} \
             (minimum of RLIMIT_CORE and RLIMIT_FSIZE)"
        )));
    }
    write_elf(writer, &DiagnosticReader::new()?, &headers, note, &loads)
}

/// Records managed extents without faulting potentially migrated pages back.
fn managed_svm_memory(
    ranges: &[(usize, usize)],
) -> impl Iterator<Item = io::Result<DiagnosticMemory<'_>>> {
    ranges.iter().map(|&(address, size)| {
        // SAFETY: No host alias is exposed. Capture retains the reservation
        // under the registry lock; commitment does not prove CPU residency.
        unsafe { DiagnosticMemory::borrow(ranges, address as u64, None, size as u64) }
            .map_err(native)
    })
}

/// Appends a native integer in the core format's little-endian order.
fn u32le(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}
/// Appends a native integer in the core format's little-endian order.
fn u64le(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}

/// Encodes the versioned `NT_AMDGPU_CORE_STATE` descriptor without Rust padding.
fn encode_note(
    devices: &[CoreDeviceSnapshot],
    queues: &[CoreQueueSnapshot<'_>],
) -> io::Result<Vec<u8>> {
    let first = devices
        .first()
        .ok_or_else(|| native("missing core device"))?;
    if devices
        .iter()
        .any(|device| device.version != first.version || device.runtime != first.runtime)
    {
        return Err(native(
            "GPU snapshots belong to different native runtime contexts",
        ));
    }
    let mut desc = Vec::new();
    u64le(&mut desc, 1);
    for version in first.version {
        u32le(&mut desc, version);
    }
    u64le(&mut desc, 16);
    u32le(&mut desc, u32::try_from(devices.len()).map_err(native)?);
    u32le(&mut desc, 120);
    u32le(&mut desc, u32::try_from(queues.len()).map_err(native)?);
    u32le(&mut desc, 64);
    u64le(&mut desc, first.runtime.r_debug);
    u32le(&mut desc, first.runtime.runtime_state);
    u32le(&mut desc, first.runtime.ttmp_setup);
    for device in devices {
        encode_device(&mut desc, device.device);
    }
    for queue in queues {
        u64le(&mut desc, 0); // No debugger exception mask was queried.
        u64le(&mut desc, queue.memory[0].address());
        u64le(&mut desc, queue.write_pointer);
        u64le(&mut desc, queue.read_pointer);
        u64le(&mut desc, queue.memory[2].address());
        for value in [
            queue.queue_id,
            queue.gpu_id,
            queue.ring_size_bytes,
            queue.queue_type,
            queue.context_size,
            0,
        ] {
            u32le(&mut desc, value);
        }
    }
    let mut note = Vec::new();
    u32le(&mut note, 7);
    u32le(&mut note, u32::try_from(desc.len()).map_err(native)?);
    u32le(&mut note, 33);
    note.extend(b"AMDGPU\0\0");
    note.extend(desc);
    Ok(note)
}

/// Serializes the native KFD device record in its public ABI field order.
fn encode_device(bytes: &mut Vec<u8>, info: CoreDeviceInfo) {
    for value in [
        info.exception_status,
        info.lds_base,
        info.lds_limit,
        info.scratch_base,
        info.scratch_limit,
        info.gpuvm_base,
        info.gpuvm_limit,
    ] {
        u64le(bytes, value);
    }
    for value in [
        info.gpu_id,
        info.location_id,
        info.vendor_id,
        info.device_id,
        info.revision_id,
        info.subsystem_vendor_id,
        info.subsystem_device_id,
        info.fw_version,
        info.gfx_target_version,
        info.simd_count,
        info.max_waves_per_simd,
        info.array_count,
        info.simd_arrays_per_engine,
        info.num_xcc,
        info.capability,
        info.debug_prop,
    ] {
        u32le(bytes, value);
    }
}

/// Builds ELF64 headers and checks every offset before creating output.
fn layout<'a>(
    note: &[u8],
    memory: Vec<DiagnosticMemory<'a>>,
) -> io::Result<(Vec<u8>, Vec<Load<'a>>, u64)> {
    let count = u16::try_from(memory.len() + 1).map_err(native)?;
    if count == 0xffff {
        return Err(native("too many ELF core segments"));
    }
    let mut header = vec![0; 64];
    header[..9].copy_from_slice(b"\x7fELF\x02\x01\x01\x40\x00");
    header[16..18].copy_from_slice(&4_u16.to_le_bytes()); // ET_CORE
    header[18..20].copy_from_slice(&224_u16.to_le_bytes()); // EM_AMDGPU
    header[20..24].copy_from_slice(&1_u32.to_le_bytes());
    header[32..40].copy_from_slice(&64_u64.to_le_bytes());
    header[52..54].copy_from_slice(&64_u16.to_le_bytes());
    header[54..56].copy_from_slice(&56_u16.to_le_bytes());
    header[56..58].copy_from_slice(&count.to_le_bytes());
    let note_offset = 64 + u64::from(count) * 56;
    program_header(&mut header, 4, 0, note_offset, 0, note.len() as u64, 0, 4);
    let mut end = note_offset
        .checked_add(note.len() as u64)
        .ok_or_else(|| native("core size overflow"))?;
    let mut loads = Vec::new();
    for memory in memory {
        let padding = (memory.address().wrapping_sub(end)) & 15;
        let offset = end
            .checked_add(padding)
            .ok_or_else(|| native("core offset overflow"))?;
        let file_size = if memory.host_address().is_some() {
            memory.size()
        } else {
            0
        };
        end = offset
            .checked_add(file_size)
            .ok_or_else(|| native("core size overflow"))?;
        // Permissions are conservative: diagnostic reads do not imply execute or write.
        program_header(
            &mut header,
            1,
            4,
            offset,
            memory.address(),
            file_size,
            memory.size(),
            16,
        );
        loads.push(Load {
            memory,
            offset,
            file_size,
        });
    }
    Ok((header, loads, end))
}

/// Writes one ELF program-header record without native layout assumptions.
#[allow(
    clippy::too_many_arguments,
    reason = "ELF64 program header has these eight fields"
)]
fn program_header(
    bytes: &mut Vec<u8>,
    kind: u32,
    flags: u32,
    offset: u64,
    address: u64,
    file_size: u64,
    memory_size: u64,
    alignment: u64,
) {
    u32le(bytes, kind);
    u32le(bytes, flags);
    for value in [offset, address, 0, file_size, memory_size, alignment] {
        u64le(bytes, value);
    }
}

/// Streams actual memory in bounded chunks; any unreadable interval fails capture.
fn write_elf(
    writer: &mut dyn Write,
    reader: &DiagnosticReader,
    headers: &[u8],
    note: &[u8],
    loads: &[Load<'_>],
) -> io::Result<()> {
    writer.write_all(headers)?;
    writer.write_all(note)?;
    let mut position = (headers.len() + note.len()) as u64;
    let mut buffer = vec![0; 1024 * 1024];
    for load in loads {
        let padding = usize::try_from(load.offset - position).map_err(native)?;
        writer.write_all(&[0; 16][..padding])?;
        position = load.offset;
        let mut offset = 0;
        while offset < load.file_size {
            let count = usize::try_from((load.file_size - offset).min(buffer.len() as u64))
                .map_err(native)?;
            reader.read(load.memory, offset, &mut buffer[..count])?;
            writer.write_all(&buffer[..count])?;
            offset += count as u64;
        }
        position += load.file_size;
    }
    Ok(())
}

/// Selects `ROCr`'s file fallback for kernel CPU-only pipe handlers.
fn output_pattern(custom: Option<&str>) -> io::Result<String> {
    if let Some(pattern) = custom {
        return substitute(pattern);
    }
    let kernel = std::fs::read_to_string("/proc/sys/kernel/core_pattern").unwrap_or_default();
    let kernel = kernel.trim_end_matches('\n');
    let pattern = if kernel.is_empty() || kernel.starts_with('|') {
        "gpucore.%p"
    } else {
        kernel
    };
    Ok(format!("{}.gpu", substitute(pattern)?))
}

/// Expands documented `ROCr` substitutions; unknown specifiers are removed.
fn substitute(pattern: &str) -> io::Result<String> {
    let mut hostname = [0_u8; 256];
    // SAFETY: The kernel writes at most the supplied buffer length.
    if unsafe { libc::gethostname(hostname.as_mut_ptr().cast(), hostname.len()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let hostname =
        String::from_utf8_lossy(hostname.split(|byte| *byte == 0).next().unwrap_or_default());
    let exe = std::env::current_exe()?;
    let exe = exe
        .file_name()
        .ok_or_else(|| native("missing executable name"))?
        .to_string_lossy();
    let mut chars = pattern.chars();
    let mut result = String::new();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some('%') => result.push('%'),
            Some('p') => result.push_str(&std::process::id().to_string()),
            // SAFETY: gettid takes no pointers and has no failure mode.
            Some('i') => result.push_str(&unsafe { libc::syscall(libc::SYS_gettid) }.to_string()),
            Some('h') => result.push_str(&hostname),
            Some('e') => result.push_str(&exe),
            Some('t') => result.push_str(
                &SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(native)?
                    .as_secs()
                    .to_string(),
            ),
            Some(_) | None => {}
        }
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::core_dump::pipe::{command_args, write_pipe_with_timeout};
    use rocddi::device::gpu::diagnostic::CoreRuntimeInfo;
    use std::time::{Duration, Instant};

    /// Exercises real file-limit signals only in an isolated subprocess.
    fn file_limit_child(case: &str, path: &str, memory: DiagnosticMemory<'_>, total: u64) -> ! {
        let (core_limit, file_limit) = [
            ("fsize-zero", (total + 1, 0)),
            ("fsize-small", (total + 1, total - 1)),
            ("core-small", (total - 1, total + 1)),
            ("exact", (total, total)),
            ("pipe", (total - 1, 0)),
        ]
        .into_iter()
        .find_map(|(name, limits)| (name == case).then_some(limits))
        .unwrap();
        let limit = libc::rlimit {
            rlim_cur: file_limit,
            rlim_max: file_limit,
        };
        let no_cpu_core = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: Only this child changes its resource limits, signal
        // disposition and calling-thread mask. All native records are live.
        unsafe {
            assert_eq!(
                libc::setrlimit(libc::RLIMIT_CORE, &raw const no_cpu_core),
                0
            );
            assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &raw const limit), 0);
            assert_ne!(libc::signal(libc::SIGXFSZ, libc::SIG_DFL), libc::SIG_ERR);
            let mut set: libc::sigset_t = std::mem::zeroed();
            assert_eq!(libc::sigemptyset(&raw mut set), 0);
            assert_eq!(libc::sigaddset(&raw mut set, libc::SIGXFSZ), 0);
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &raw const set, std::ptr::null_mut()),
                0
            );
        }
        // dump supplies its sampled CORE value to write_file. Passing it here
        // avoids raising an inherited hard CORE limit just to run this test.
        if case == "pipe" {
            write_pipe("sh -c \"cat >/dev/null\"", |writer| {
                emit_elf(writer, &[], vec![memory], u64::MAX)
            })
            .unwrap();
        } else {
            let result = write_file(path, core_limit, |writer, limit| {
                emit_elf(writer, &[], vec![memory], limit)
            });
            if case == "exact" {
                result.unwrap();
            } else {
                let message = result.unwrap_err().to_string();
                assert!(message.contains(&format!(
                    "effective file limit {}",
                    core_limit.min(file_limit)
                )));
                assert!(message.contains("RLIMIT_CORE and RLIMIT_FSIZE"));
            }
        }
        // SAFETY: Disable CPU dumps after emission: doing so earlier can deny
        // our /proc/self/mem open. This affects only the isolated subprocess.
        assert_eq!(unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) }, 0);
        // There is no HSA session in this child, so the actual fatal path
        // cannot capture again. It must still terminate with SIGABRT.
        crate::core_dump::fatal_abort();
    }

    #[test]
    fn file_size_limits_preserve_cleanup_and_fatal_abort() {
        use std::os::unix::process::ExitStatusExt;

        let source = [0xa5_u8; 4096];
        // SAFETY: The source remains live through the complete emission.
        let memory = unsafe {
            DiagnosticMemory::borrow(
                &source,
                0x4000,
                Some(source.as_ptr() as usize),
                source.len() as u64,
            )
            .unwrap()
        };
        let (_, _, total) = layout(&[], vec![memory]).unwrap();
        if let Ok(case) = std::env::var("LIBHSA_CORE_LIMIT_TEST_CHILD") {
            let path = std::env::var("LIBHSA_CORE_LIMIT_TEST_PATH").unwrap();
            file_limit_child(&case, &path, memory, total);
        }
        for case in ["fsize-small", "fsize-zero", "core-small", "exact", "pipe"] {
            let path = std::env::temp_dir().join(format!(
                "libhsa-core-limits-{}-{case}.gpu",
                std::process::id()
            ));
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "core_dump::linux::tests::file_size_limits_preserve_cleanup_and_fatal_abort",
                ])
                .env("LIBHSA_CORE_LIMIT_TEST_CHILD", case)
                .env("LIBHSA_CORE_LIMIT_TEST_PATH", &path)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    break child.wait().unwrap();
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            let output = std::fs::read(&path);
            if output.is_ok() {
                std::fs::remove_file(&path).unwrap();
            }
            assert_eq!(
                status.signal(),
                Some(libc::SIGABRT),
                "{case}: {status}, output bytes {:?}",
                output.as_ref().ok().map(Vec::len)
            );
            if case == "exact" {
                let bytes = output.unwrap();
                assert_eq!(bytes.len() as u64, total);
                assert_eq!(&bytes[..4], b"\x7fELF");
                assert_eq!(&bytes[bytes.len() - source.len()..], source.as_slice());
            } else {
                assert_eq!(
                    output.unwrap_err().kind(),
                    io::ErrorKind::NotFound,
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn managed_svm_extents_are_emitted_without_reading_their_pages() {
        // These addresses have no readable backing. Passing them through the
        // production enumeration and writer would fail if an alias were given.
        let ranges = [(0, 4096), (0xdead_0000, 8192)];
        let memory = managed_svm_memory(&ranges)
            .collect::<io::Result<Vec<_>>>()
            .unwrap();
        assert!(memory.iter().all(|region| region.host_address().is_none()));
        let (headers, loads, total) = layout(&[], memory).unwrap();
        let mut output = Vec::new();
        write_elf(
            &mut output,
            &DiagnosticReader::new().unwrap(),
            &headers,
            &[],
            &loads,
        )
        .unwrap();
        assert_eq!(loads.len(), ranges.len());
        for (index, &(address, size)) in ranges.iter().enumerate() {
            let header = 64 + (index + 1) * 56;
            let u64at = |offset| u64::from_le_bytes(output[offset..offset + 8].try_into().unwrap());
            assert_eq!(&output[header..header + 4], &1_u32.to_le_bytes());
            assert_eq!(u64at(header + 16), address as u64);
            assert_eq!(u64at(header + 32), 0);
            assert_eq!(u64at(header + 40), size as u64);
        }
        assert_eq!(output.len() as u64, total);
    }

    #[test]
    fn elf_contains_actual_cpu_alias_bytes_at_gpu_addresses_and_native_note_records() {
        let source = (0..=255_u8).collect::<Vec<_>>();
        // SAFETY: The vector retains the entire readable mapping through capture.
        let memory = unsafe {
            DiagnosticMemory::borrow(
                &source,
                0x1234_0007,
                Some(source.as_ptr() as usize),
                source.len() as u64,
            )
            .unwrap()
        };
        // SAFETY: An unavailable alias never causes a host read.
        let missing =
            unsafe { DiagnosticMemory::borrow(&source, 0x8000_0000, None, 4096).unwrap() };
        let device = CoreDeviceSnapshot {
            version: [1, 23],
            runtime: CoreRuntimeInfo {
                r_debug: 0x1122_3344,
                runtime_state: 1,
                ttmp_setup: 1,
            },
            device: CoreDeviceInfo {
                gpu_id: 42,
                gfx_target_version: 120_001,
                ..CoreDeviceInfo::default()
            },
        };
        let queue = CoreQueueSnapshot {
            queue_id: 9,
            gpu_id: 42,
            queue_type: 2,
            ring_size_bytes: 1024,
            write_pointer: 0x9000,
            read_pointer: 0x9008,
            context_size: 65536,
            memory: [memory; 3],
        };
        let note = encode_note(&[device], &[queue]).unwrap();
        let (headers, loads, total) = layout(&note, vec![memory, missing]).unwrap();
        let mut output = Vec::new();
        write_elf(
            &mut output,
            &DiagnosticReader::new().unwrap(),
            &headers,
            &note,
            &loads,
        )
        .unwrap();
        let u32at = |offset| u32::from_le_bytes(output[offset..offset + 4].try_into().unwrap());
        let u64at = |offset| u64::from_le_bytes(output[offset..offset + 8].try_into().unwrap());
        assert_eq!(&output[..9], b"\x7fELF\x02\x01\x01\x40\x00");
        assert_eq!(&output[16..20], &[4, 0, 224, 0]);
        assert_eq!(u64at(32), 64);
        assert_eq!(u32at(64), 4); // PT_NOTE
        let note_at = u64at(72) as usize;
        assert_eq!(u32at(note_at), 7);
        assert_eq!(u32at(note_at + 8), 33);
        assert_eq!(&output[note_at + 12..note_at + 20], b"AMDGPU\0\0");
        let desc = note_at + 20;
        assert_eq!(u64at(desc), 1);
        assert_eq!((u32at(desc + 8), u32at(desc + 12)), (1, 23));
        assert_eq!(
            (u64at(desc + 16), u32at(desc + 28), u32at(desc + 36)),
            (16, 120, 64)
        );
        assert_eq!(u64at(desc + 40), 0x1122_3344);
        let q = desc + 40 + 16 + 120;
        assert_eq!((u32at(q + 40), u32at(q + 44), u32at(q + 48)), (9, 42, 1024));
        assert_eq!(u32at(q + 56), 65536);
        assert_eq!(u64at(64 + 56 + 16), memory.address());
        assert_eq!(loads[0].offset % 16, memory.address() % 16);
        let data = loads[0].offset as usize;
        assert_eq!(&output[data..data + source.len()], source.as_slice());
        assert_eq!((u64at(64 + 112 + 32), u64at(64 + 112 + 40)), (0, 4096));
        assert_eq!(output.len() as u64, total);
        assert!(
            DiagnosticReader::new()
                .unwrap()
                .read(memory, 255, &mut [0; 2])
                .is_err()
        );
        assert!(
            DiagnosticReader::new()
                .unwrap()
                .read(missing, 0, &mut [0])
                .is_err()
        );
    }

    #[test]
    fn patterns_preserve_quoted_arguments_and_expand_only_supported_specifiers() {
        assert_eq!(
            command_args("sh -c \"cat > /tmp/gpu core\" \"\" x\\ y").unwrap(),
            ["sh", "-c", "cat > /tmp/gpu core", "", "x y"]
        );
        assert!(command_args("sh \"unterminated").is_err());
        assert!(command_args(" ").is_err());
        assert!(command_args("cat\\").is_err());
        assert_eq!(
            substitute("x.%%.%p.%x.%").unwrap(),
            format!("x.%.{}..", std::process::id())
        );
    }

    #[test]
    fn pipe_consumption_and_early_exit_preserve_the_calling_thread_signal_mask() {
        let mut before: libc::sigset_t = unsafe { std::mem::zeroed() };
        // SAFETY: A null input set queries the current thread's initialized output.
        assert_eq!(
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &raw mut before) },
            0
        );
        write_pipe("sh -c \"cat >/dev/null\"", |writer| {
            writer.write_all(&vec![0xA5; 65536])
        })
        .unwrap();
        assert!(
            write_pipe("sh -c \"exit 7\"", |writer| writer
                .write_all(&vec![0; 65536]))
            .is_err()
        );
        let mut after: libc::sigset_t = unsafe { std::mem::zeroed() };
        // SAFETY: Both masks are initialized; sigismember only reads them.
        unsafe {
            assert_eq!(
                libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), &raw mut after),
                0
            );
            assert_eq!(
                libc::sigismember(&raw const before, libc::SIGPIPE),
                libc::sigismember(&raw const after, libc::SIGPIPE)
            );
        }
    }

    #[test]
    fn timed_out_pipe_retires_its_descendant_process_group() {
        let path =
            std::env::temp_dir().join(format!("libhsa-core-pipe-{}.pid", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let command = format!("sh -c \"sleep 60 & echo $! > {}; wait\"", path.display());
        let error = write_pipe_with_timeout(&command, Duration::from_millis(100), |writer| {
            writer.write_all(&vec![0; 65536])
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        let pid: u32 = std::fs::read_to_string(&path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
            if stat
                .as_ref()
                .map_or(true, |stat| stat.split_whitespace().nth(2) == Some("Z"))
            {
                break;
            }
            assert!(Instant::now() < deadline, "pipe descendant still running");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn short_output_fails_instead_of_claiming_a_dump() {
        let source = [42_u8; 16];
        // SAFETY: The valid source remains owned through the attempted stream write.
        let memory = unsafe {
            DiagnosticMemory::borrow(&source, 0x4000, Some(source.as_ptr() as usize), 16).unwrap()
        };
        let (headers, loads, _) = layout(&[], vec![memory]).unwrap();
        let mut storage = [0_u8; 10];
        let mut short = storage.as_mut_slice();
        assert!(
            write_elf(
                &mut short,
                &DiagnosticReader::new().unwrap(),
                &headers,
                &[],
                &loads
            )
            .is_err()
        );
    }
}
