// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native context extents for the internal AQL cache-publication queue.

use super::*;

fn reported_gfx9_node(target: u32) -> sysfs::KfdNode {
    let mut node = legacy_gfx942_node();
    node.queues.gfx_target = target;
    // Positive KFD sizes and geometry captured from MI210 and gfx950 nodes.
    let (compute_units, xcc_count, engines_per_xcc, lds, context, stack) = match target {
        90010 => (104, 1, 8, 65_536, 63_508_480, 36_864),
        90500 => (256, 8, 4, 163_840, 22_687_744, 12_288),
        90402 => (304, 8, 4, 65_536, 23_203_840, 12_288),
        _ => panic!("target has no reported-size fixture"),
    };
    node.queues.compute_units = compute_units;
    node.queues.xcc_count = xcc_count;
    node.queues.shader_engine_count_per_xcc = engines_per_xcc;
    node.queues.local_data_share_byte_length = lds;
    node.queues.context_size = context;
    node.queues.control_stack_size = stack;
    node
}

#[test]
fn gfx9_native_context_extents_follow_saved_waves() {
    for (target, debug_size, total_size) in [
        (90010, 131_072, 63_639_552),
        (90402, 48_640, 186_019_840),
        (90500, 40_960, 181_829_632),
    ] {
        let node = reported_gfx9_node(target);
        let compute = compute_storage(node.queues).unwrap();
        assert_eq!(compute.context_size, node.queues.context_size);
        assert_eq!(compute.control_stack_size, node.queues.control_stack_size);
        assert_eq!(compute.debug_size, debug_size, "target {target}");
        assert_eq!(compute.total_size, total_size, "target {target}");
    }
    // gfx908 has the same qualified pre-Navi save-wave contract.
    let mut node = reported_gfx9_node(90010);
    node.queues.gfx_target = 90008;
    assert_eq!(compute_storage(node.queues).unwrap().debug_size, 131_072);
}

#[test]
fn reported_gfx942_uses_kernel_geometry_without_changing_legacy_derivation() {
    let mut node = legacy_gfx942_node();
    node.queues.compute_units = 40;
    node.queues.xcc_count = 1;
    node.queues.shader_engine_count_per_xcc = 2;
    node.queues.simd_arrays_per_engine = 2;
    let legacy = compute_storage(node.queues).unwrap();
    assert_eq!(legacy.debug_size, 16_384);
    assert_eq!(legacy.context_size, 0x0174_a000);
    node.queues.context_size = 65_536;
    node.queues.control_stack_size = 4096;
    let reported = compute_storage(node.queues).unwrap();
    assert_eq!(reported.debug_size, 32_768);
    assert_eq!(reported.total_size, 98_304);
    assert_eq!(reported.context_size, 65_536);
    assert_eq!(reported.control_stack_size, 4096);
}

#[test]
fn reported_gfx9_bound_reconstructs_total_shader_engines_across_xccs() {
    let mut node = reported_gfx9_node(90500);
    // Raw array_count 8 / (one array per engine * eight XCCs) gives one
    // shader engine per XCC. KFD's total-engine bound is 8 * 512, so all
    // 32 * 40 saved waves per XCC fit. A per-XCC bound would admit only 512.
    node.queues.shader_engine_count_per_xcc = 1;
    let compute = compute_storage(node.queues).unwrap();
    assert_eq!(compute.debug_size, 40_960);
    assert_eq!(compute.total_size, 181_829_632);
}

/// Values observed at the production ioctl boundary before simulated rejection.
struct SubmittedContext {
    /// Exact input struct submitted by production queue construction.
    args: uapi::CreateQueue,
    /// Native ring mapping whose start and extent KFD validates.
    ring: uapi::AllocMemory,
    /// Native pointer mapping shared by the two page-relative indices.
    pointers: uapi::AllocMemory,
    /// EOP allocation retained alongside the compute context.
    eop: uapi::AllocMemory,
    /// Complete SVM range or native BO extent covering CWSR and debugger data.
    extent: u64,
    /// Actual initialized native headers, one per XCC.
    headers: Vec<[u8; 40]>,
}

#[derive(Default)]
struct ContextIo {
    /// Native allocations still owned by production cleanup.
    buffers: BTreeMap<u64, uapi::AllocMemory>,
    /// Next sparse-file offset used by fake native allocation.
    offset: u64,
    /// Monotonic fake handle, independent of the number of live allocations.
    handle: u64,
    /// Accepted SVM context range, absent when registration is rejected.
    svm: Option<(u64, u64)>,
    /// Inputs retained before the fake `CREATE_QUEUE` rejects the request.
    submitted: Option<SubmittedContext>,
}

fn submitted_context(node: sysfs::KfdNode, reject_svm: bool) -> SubmittedContext {
    let state = Arc::new(Mutex::new(ContextIo::default()));
    let observed = state.clone();
    let render = backing_file();
    let contents = render.try_clone().unwrap();
    let process_memory = File::open("/proc/self/mem").unwrap();
    let kfd = shared(connection::Kfd::with_hook(
        File::open("/dev/null").unwrap(),
        Arc::new(move |call| {
            let mut state = observed.lock().unwrap();
            match call {
                connection::Call::Wait(args, _) => args.result = uapi::WAIT_TIMEOUT,
                connection::Call::DestroyEvent(_) | connection::Call::SetScratchBackingVa(_) => {}
                connection::Call::Allocate(args) => {
                    state.handle += 1;
                    args.handle = state.handle;
                    args.mmap_offset = state.offset;
                    state.offset += args.size;
                    contents.set_len(state.offset).unwrap();
                    state.buffers.insert(args.handle, **args);
                }
                connection::Call::Map(args, _) | connection::Call::Unmap(args, _) => {
                    args.success = args.count;
                }
                connection::Call::Free(args) => {
                    state.buffers.remove(&args.handle).unwrap();
                }
                connection::Call::Svm(args, _) => {
                    if reject_svm {
                        return Err(io::Error::from_raw_os_error(errno::EOPNOTSUPP));
                    }
                    state.svm = Some((args.start_address, args.size));
                }
                connection::Call::CreateQueue(args) => {
                    let allocation = |address| {
                        *state
                            .buffers
                            .values()
                            .find(|buffer| {
                                address >= buffer.va && address < buffer.va + buffer.size
                            })
                            .unwrap()
                    };
                    let (extent, context) = if let Some((address, size)) = state.svm {
                        assert_eq!(args.context_address, address);
                        (size, None)
                    } else {
                        let buffer = allocation(args.context_address);
                        (buffer.size, Some(buffer))
                    };
                    let mut headers = Vec::new();
                    for xcc in 0..node.queues.xcc_count {
                        let offset = u64::from(xcc) * u64::from(args.context_size);
                        let mut header = [0; 40];
                        if let Some(context) = context {
                            contents
                                .read_exact_at(&mut header, context.mmap_offset + offset)
                                .unwrap();
                        } else {
                            process_memory
                                .read_exact_at(&mut header, args.context_address + offset)
                                .unwrap();
                        }
                        headers.push(header);
                    }
                    let submitted = SubmittedContext {
                        args: **args,
                        ring: allocation(args.ring_address),
                        pointers: allocation(args.read_pointer),
                        eop: allocation(args.eop_address),
                        extent,
                        headers,
                    };
                    state.submitted = Some(submitted);
                    // Stop before a queue can be published or any packet submitted.
                    return Err(io::Error::from_raw_os_error(errno::EINVAL));
                }
                _ => panic!("unexpected native context operation"),
            }
            Ok(())
        }),
    ));
    let vm = memory::queue_fixture(kfd, render, node);
    let request = Request::validate(&node, descriptor(aql(QueueProducerMode::Single))).unwrap();
    let failure = KfdQueue::create(vm, &request, DriverContextLifetime::Process, false)
        .err()
        .unwrap();
    assert_eq!(
        failure.native_status().unwrap().errno_code(),
        Some(errno::EINVAL)
    );
    let mut state = state.lock().unwrap();
    assert!(state.buffers.is_empty());
    state.submitted.take().unwrap()
}

#[test]
fn gfx9_create_queue_submits_complete_native_context_extent() {
    for (target, debug_size, total_size) in [
        (90010, 131_072_u32, 63_639_552),
        (90402, 48_640, 186_019_840),
        (90500, 40_960, 181_829_632),
    ] {
        for reject_svm in [false, true] {
            let node = reported_gfx9_node(target);
            let submitted = submitted_context(node, reject_svm);
            let args = submitted.args;
            assert_eq!(args.queue_type, 2);
            assert_eq!(args.ring_size, 4096);
            assert_eq!((args.gpu_id, args.percentage, args.priority), (42, 100, 7));
            assert_eq!((args.sdma_engine_id, args.metadata_ring_size), (0, 0));
            assert_eq!((args.queue_id, args.doorbell_offset), (0, 0));
            assert_eq!(submitted.ring.va, args.ring_address);
            assert_eq!(submitted.ring.size, 4096);
            assert_eq!(submitted.pointers.size, 4096);
            assert_eq!(
                args.read_pointer,
                submitted.pointers.va + AQL_READ_OFFSET as u64
            );
            assert_eq!(
                args.write_pointer,
                submitted.pointers.va + AQL_WRITE_OFFSET as u64
            );
            assert_eq!((args.eop_address, args.eop_size), (submitted.eop.va, 4096));
            assert_eq!(args.context_size, node.queues.context_size);
            assert_eq!(args.control_stack_size, node.queues.control_stack_size);
            assert_eq!(
                submitted.extent, total_size,
                "target {target}, native={reject_svm}"
            );
            for (xcc, header) in submitted.headers.iter().enumerate() {
                let xcc = u32::try_from(xcc).unwrap();
                let mut expected = [0; 40];
                expected[16..20].copy_from_slice(
                    &((node.queues.xcc_count - xcc) * args.context_size).to_ne_bytes(),
                );
                expected[20..24]
                    .copy_from_slice(&(debug_size * node.queues.xcc_count).to_ne_bytes());
                assert_eq!(header, &expected);
            }
        }
    }
}

#[test]
fn gfx9_reported_context_rejects_missing_geometry_and_invalid_extents() {
    let base = reported_gfx9_node(90500).queues;
    for queues in [
        sysfs::KfdQueueProperties {
            xcc_count: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: 255,
            ..base
        },
        sysfs::KfdQueueProperties {
            shader_engine_count_per_xcc: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            simd_arrays_per_engine: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            shader_engine_count_per_xcc: u32::MAX,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: u32::MAX - 7,
            ..base
        },
        sysfs::KfdQueueProperties {
            context_size: 0,
            control_stack_size: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            context_size: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            context_size: 22_687_745,
            ..base
        },
        sysfs::KfdQueueProperties {
            control_stack_size: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            control_stack_size: 12_289,
            ..base
        },
        sysfs::KfdQueueProperties {
            context_size: 0xffff_f000,
            ..base
        },
    ] {
        assert!(compute_storage(queues).is_err(), "{queues:?}");
    }
}
