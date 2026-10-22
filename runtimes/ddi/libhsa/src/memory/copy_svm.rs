// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! SVM policy, asynchronous copies, and native copy route execution.

use super::*;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_memory_copy(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        match memory_copy_has_work(dst, src, size) {
            Ok(false) => return SUCCESS,
            Ok(true) => {}
            Err(status) => return status,
        }
        let host_destination = host_copy_address(runtime, dst as usize, size, ACCESS_PERMISSION_WO);
        let host_source = host_copy_address(runtime, src as usize, size, ACCESS_PERMISSION_RO);
        if let (Ok(destination), Ok(source)) = (host_destination, host_source) {
            // SAFETY: HSA requires both ranges to be valid for size bytes.
            // ptr::copy tolerates overlap. The registry lock retains owned
            // backing throughout this host copy.
            unsafe { std::ptr::copy(source as *const u8, destination as *mut u8, size) };
            fence(Ordering::SeqCst);
            return SUCCESS;
        }

        let resolved = match resolve_sync_gpu_copy(
            runtime,
            dst as usize,
            src as usize,
            size,
            host_destination,
            host_source,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        let mut retained = [None; 2];
        for (slot, base) in retained.iter_mut().zip(resolved.borrowed_memory) {
            if let Some(base) = base {
                if let Err(status) = runtime.memory_state.retain_copy_borrow(base) {
                    for base in retained.into_iter().flatten() {
                        runtime.memory_state.release_copy_borrow(base);
                    }
                    return status;
                }
                *slot = Some(base);
            }
        }
        let stop = runtime.stop_workers.clone();
        let quarantine = runtime.memory_state.async_copy_quarantine.clone();
        drop(guard);

        let (status, may_be_live) = match execute_sync_gpu_copy(resolved.route, &stop) {
            Ok(SyncGpuCopyOutcome::Done) => (SUCCESS, false),
            Ok(SyncGpuCopyOutcome::LoadedDestination { address, bytes }) => {
                (write_loaded_copy_destination(address, &bytes), false)
            }
            Err(failure) => (failure.status, failure.operands_may_be_live),
        };
        if may_be_live {
            quarantine.store(true, Ordering::Release);
        }
        release_async_resources(&[], &retained, may_be_live);
        status
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_fill")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_fill(
    pointer: *mut c_void,
    value: u32,
    count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let size = match memory_fill_size(pointer, count) {
            Ok(Some(size)) => size,
            Ok(None) => return SUCCESS,
            Err(status) => return status,
        };
        let Some(description) = pointer_description(runtime, pointer as usize) else {
            let host =
                match crate::loader::loaded_host_range(&runtime.loader, pointer as usize, size) {
                    Ok(Some(host)) => host,
                    Ok(None) => return INVALID_ALLOCATION,
                    Err(status) => return status,
                };
            if host % align_of::<u32>() != 0 {
                return INVALID_ALLOCATION;
            }
            // SAFETY: The registry lock retains the complete writable loaded
            // mapping until the fill returns.
            unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                .fill(MaybeUninit::new(value));
            fence(Ordering::SeqCst);
            return SUCCESS;
        };
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return INVALID_ALLOCATION;
        }
        if let Some(host) = description.host_range(pointer as usize, size) {
            if vmem_copy_access(
                runtime,
                &description,
                HsaAgent { handle: CPU_AGENT },
                ACCESS_PERMISSION_WO,
            )
            .is_ok()
            {
                // SAFETY: The complete aligned output range belongs to a live
                // allocation retained by the runtime lock. Its prior contents
                // need not have been initialized as u32 values.
                unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                    .fill(MaybeUninit::new(value));
                fence(Ordering::SeqCst);
                return SUCCESS;
            }
        }
        let Some((index, agent)) = selected_copy_gpu(runtime, &description) else {
            return INVALID_ALLOCATION;
        };
        let (address, base) = match gpu_copy_address(
            runtime,
            Some(&description),
            pointer as usize,
            size,
            agent,
            ACCESS_PERMISSION_WO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let Ok(native_count) = u64::try_from(count) else {
            return INVALID_ARGUMENT;
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        if let Err(status) = runtime.memory_state.retain_copy_borrow(base) {
            return status;
        }
        let device = runtime.gpus[index].device.clone();
        let stop = runtime.stop_workers.clone();
        let quarantine = runtime.memory_state.async_copy_quarantine.clone();
        drop(guard);

        let (status, may_be_live) = match device.gpu() {
            Ok(gpu) => {
                // SAFETY: The complete writable GPU range was checked above.
                // Its allocation borrow and in-flight token remain live until
                // retirement or process teardown after an uncertain result.
                match unsafe { gpu.fill_u32(address, value, native_count, &stop) } {
                    Ok(()) => (SUCCESS, false),
                    Err(failure) => (map_error(failure.error), failure.operands_may_be_live),
                }
            }
            Err(error) => (map_error(error), false),
        };
        if may_be_live {
            quarantine.store(true, Ordering::Release);
        }
        release_async_resources(&[], &[Some(base)], may_be_live);
        status
    })
}

pub(super) fn svm_range(pointer: *mut c_void, size: usize) -> Result<(u64, u64), Status> {
    if pointer.is_null() || size == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let page_size = rocddi::memory::host::host_page_size().map_err(map_error)?;
    let page_size = usize::try_from(page_size).map_err(|_| INVALID_ARGUMENT)?;
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(ERROR);
    }
    let address = pointer as usize;
    let base = address & !(page_size - 1);
    let end = address
        .checked_add(size)
        .and_then(|end| end.checked_add(page_size - 1))
        .map(|end| end & !(page_size - 1))
        .ok_or(INVALID_ARGUMENT)?;
    let length = end.checked_sub(base).ok_or(INVALID_ARGUMENT)?;
    Ok((base as u64, length as u64))
}

fn svm_location_from_agent(
    runtime: &HsaState,
    agent: HsaAgent,
    allow_null: bool,
) -> Result<SvmLocation, Status> {
    if agent.handle == 0 && allow_null {
        return Ok(SvmLocation::Undefined);
    }
    if agent.handle == CPU_AGENT {
        return Ok(SvmLocation::System);
    }
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    Ok(SvmLocation::Device(runtime.gpus[index].endpoint.id))
}

fn svm_agent_from_location(runtime: &HsaState, location: SvmLocation) -> Result<HsaAgent, Status> {
    match location {
        SvmLocation::System => Ok(HsaAgent { handle: CPU_AGENT }),
        SvmLocation::Undefined => Ok(HsaAgent { handle: 0 }),
        SvmLocation::Device(identity) => runtime
            .gpus
            .iter()
            .position(|gpu| gpu.endpoint.id == identity)
            .map(|index| HsaAgent {
                handle: GPU_AGENT_BASE + index as u64,
            })
            .ok_or(ERROR),
    }
}

fn reserve_svm_attributes(count: usize) -> Result<(Vec<SvmAttribute>, Vec<Option<usize>>), Status> {
    if count > SVM_MAX_ATTRIBUTES {
        return Err(INVALID_ARGUMENT);
    }
    let mut attributes = Vec::new();
    attributes
        .try_reserve_exact(count.saturating_add(2))
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    Ok((attributes, indices))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_set")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_svm_attributes_set(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let pairs = if attribute_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires attribute_count readable pairs.
            unsafe { std::slice::from_raw_parts(attribute_list, attribute_count) }
        };
        let (mut attributes, _) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut scalar_attributes = 0_u16;
        let mut agents = Vec::new();
        if agents.try_reserve_exact(attribute_count).is_err() {
            return OUT_OF_RESOURCES;
        }
        let mut set_flags = 0_u32;
        let mut clear_flags = 0_u32;
        for pair in pairs {
            let scalar_bit = match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_MIGRATION_GRANULARITY
                | AMD_SVM_ATTRIB_PREFERRED_LOCATION
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => Some(1_u16 << pair.attribute),
                _ => None,
            };
            if let Some(bit) = scalar_bit {
                if scalar_attributes & bit != 0 {
                    return INCOMPATIBLE_ARGUMENTS;
                }
                scalar_attributes |= bit;
            }
            match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => match pair.value {
                    AMD_SVM_GLOBAL_FLAG_FINE_GRAINED => set_flags |= SVM_FLAG_COHERENT,
                    AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED => clear_flags |= SVM_FLAG_COHERENT,
                    _ => return INVALID_ARGUMENT,
                },
                AMD_SVM_ATTRIB_READ_ONLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_ONLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_ONLY;
                    }
                }
                AMD_SVM_ATTRIB_HIVE_LOCAL => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_HIVE_LOCAL;
                    } else {
                        clear_flags |= SVM_FLAG_HIVE_LOCAL;
                    }
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    attributes.push(SvmAttribute::MigrationGranularity(
                        pair.value.min(SVM_MAX_MIGRATION_GRANULARITY) as u32,
                    ))
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    let location = match svm_location_from_agent(
                        runtime,
                        HsaAgent { handle: pair.value },
                        true,
                    ) {
                        Ok(location) => location,
                        Err(status) => return status,
                    };
                    attributes.push(SvmAttribute::PreferredLocation(location));
                }
                AMD_SVM_ATTRIB_READ_MOSTLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    }
                }
                AMD_SVM_ATTRIB_GPU_EXEC => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_EXECUTE;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_EXECUTE;
                    }
                }
                AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE
                | AMD_SVM_ATTRIB_AGENT_NO_ACCESS => {
                    let agent = HsaAgent { handle: pair.value };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if agents.contains(&agent.handle) {
                        return INCOMPATIBLE_ARGUMENTS;
                    }
                    agents.push(agent.handle);
                    if agent.handle == CPU_AGENT {
                        if pair.attribute == AMD_SVM_ATTRIB_AGENT_NO_ACCESS {
                            clear_flags |= SVM_FLAG_HOST_ACCESS;
                        } else {
                            set_flags |= SVM_FLAG_HOST_ACCESS;
                        }
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        let access = match pair.attribute {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE => SvmAccess::Accessible,
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE => {
                                SvmAccess::AccessibleInPlace
                            }
                            _ => SvmAccess::NoAccess,
                        };
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
            clear_flags &= !SVM_FLAG_HOST_ACCESS;
        }
        if clear_flags != 0 {
            attributes.push(SvmAttribute::ClearFlags(clear_flags));
        }
        if set_flags != 0 {
            attributes.push(SvmAttribute::SetFlags(set_flags));
        }
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        linux_interop::set_kfd_svm_attributes(&runtime.session, address, size, &attributes)
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_get")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_svm_attributes_get(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let (mut attributes, mut indices) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut get_flags = false;
        for index in 0..attribute_count {
            // SAFETY: The caller supplies this complete pair array. Only each
            // attribute tag is initialized for a query; its value is input
            // only for ACCESS_QUERY.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => {
                    get_flags = true;
                    indices.push(None);
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::MigrationGranularity(0));
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PreferredLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PrefetchLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY requires the agent value as input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if agent.handle == CPU_AGENT {
                        get_flags = true;
                        indices.push(None);
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        indices.push(Some(attributes.len()));
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access: SvmAccess::Accessible,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        let flag_indices = if get_flags {
            let clear = attributes.len();
            attributes.push(SvmAttribute::ClearFlags(0));
            let set = attributes.len();
            attributes.push(SvmAttribute::SetFlags(0));
            Some((clear, set))
        } else {
            None
        };
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        if let Err(error) =
            linux_interop::get_kfd_svm_attributes(&runtime.session, address, size, &mut attributes)
        {
            return map_error(error);
        }
        let (clear_flags, set_flags) = flag_indices.map_or((0, 0), |(clear, set)| {
            let clear = match attributes.get(clear).copied() {
                Some(SvmAttribute::ClearFlags(value)) => value,
                _ => 0,
            };
            let set = match attributes.get(set).copied() {
                Some(SvmAttribute::SetFlags(value)) => value,
                _ => 0,
            };
            (clear, set)
        });
        for (index, native_index) in indices.iter().copied().enumerate() {
            // SAFETY: The caller retains the complete writable pair array.
            // Read only the initialized tag until an output value is written.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            let value = match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => {
                    if set_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_FINE_GRAINED
                    } else if clear_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED
                    } else {
                        AMD_SVM_GLOBAL_FLAG_INDETERMINATE
                    }
                }
                AMD_SVM_ATTRIB_READ_ONLY => u64::from(set_flags & SVM_FLAG_GPU_READ_ONLY),
                AMD_SVM_ATTRIB_HIVE_LOCAL => u64::from(set_flags & SVM_FLAG_HIVE_LOCAL),
                AMD_SVM_ATTRIB_READ_MOSTLY => u64::from(set_flags & SVM_FLAG_GPU_READ_MOSTLY),
                AMD_SVM_ATTRIB_GPU_EXEC => u64::from(set_flags & SVM_FLAG_GPU_EXECUTE),
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let Some(SvmAttribute::MigrationGranularity(value)) =
                        attributes.get(native).copied()
                    else {
                        return ERROR;
                    };
                    u64::from(value)
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION | AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let location = match attributes.get(native).copied() {
                        Some(SvmAttribute::PreferredLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFERRED_LOCATION =>
                        {
                            location
                        }
                        Some(SvmAttribute::PrefetchLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFETCH_LOCATION =>
                        {
                            location
                        }
                        _ => return ERROR,
                    };
                    let agent = match svm_agent_from_location(runtime, location) {
                        Ok(agent) => agent,
                        Err(status) => return status,
                    };
                    agent.handle
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY retains its initialized agent input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    let result = if agent.handle == CPU_AGENT {
                        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                        } else {
                            AMD_SVM_ATTRIB_AGENT_NO_ACCESS
                        }
                    } else {
                        let Some(native) = native_index else {
                            return ERROR;
                        };
                        match attributes.get(native).copied() {
                            Some(SvmAttribute::Access {
                                access: SvmAccess::Accessible,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::AccessibleInPlace,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::NoAccess,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
                            _ => return ERROR,
                        }
                    };
                    // SAFETY: The pair's initialized tag is the output field
                    // for ACCESS_QUERY; its input agent value is preserved.
                    unsafe { (&raw mut (*pair).attribute).write(result) };
                    continue;
                }
                _ => return INVALID_ARGUMENT,
            };
            // SAFETY: The caller supplied writable value storage. Its prior
            // bytes need not have been initialized for output-only queries.
            unsafe { (&raw mut (*pair).value).write(value) };
        }
        SUCCESS
    })
}

/// Rolls back signal and allocation retention until the worker accepts them.
struct PendingAsyncRetention<'a> {
    signals: &'a mut SignalState,
    memory: &'a mut MemoryState,
    /// Signals retained for the pending worker.
    handles: Arc<Vec<HsaSignal>>,
    /// Prefix successfully retained before a possible failure.
    retained: usize,
    /// Allocation bases borrowed until worker ownership is published.
    borrowed_memory: Vec<usize>,
}

impl<'a> PendingAsyncRetention<'a> {
    fn new(
        signals: &'a mut SignalState,
        memory: &'a mut MemoryState,
        handles: Arc<Vec<HsaSignal>>,
    ) -> Result<Self, Status> {
        let mut pending = Self {
            signals,
            memory,
            handles,
            retained: 0,
            borrowed_memory: Vec::new(),
        };
        for signal in pending.handles.iter() {
            pending.signals.retain_async_signal(*signal)?;
            pending.retained += 1;
        }
        Ok(pending)
    }

    fn retain_memory(&mut self, base: usize) -> Result<(), Status> {
        self.borrowed_memory
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        self.memory.retain_copy_borrow(base)?;
        self.borrowed_memory.push(base);
        Ok(())
    }

    fn transfer_to_worker(&mut self) {
        self.retained = 0;
        self.borrowed_memory.clear();
    }
}

impl MemoryState {
    pub(super) fn retain_copy_borrow(&mut self, base: usize) -> Result<(), Status> {
        if !self.async_copy_borrows.contains_key(&base) {
            self.async_copy_borrows
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
        }
        let count = self.async_copy_borrows.entry(base).or_default();
        *count = count.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(())
    }

    pub(super) fn release_copy_borrow(&mut self, base: usize) {
        if let Some(count) = self.async_copy_borrows.get_mut(&base) {
            *count -= 1;
            if *count == 0 {
                self.async_copy_borrows.remove(&base);
            }
        }
    }

    pub(super) fn copy_borrowed(&self, base: usize) -> bool {
        self.async_copy_borrows
            .get(&base)
            .is_some_and(|count| *count != 0)
    }
}

impl Drop for PendingAsyncRetention<'_> {
    fn drop(&mut self) {
        for signal in self.handles.iter().take(self.retained) {
            self.signals.release_async_signal(*signal);
        }
        for &base in &self.borrowed_memory {
            self.memory.release_copy_borrow(base);
        }
    }
}

fn release_async_resources(
    handles: &[HsaSignal],
    borrowed_memory: &[Option<usize>],
    retain_memory: bool,
) {
    if let Ok(mut guard) = lock() {
        if let Some(runtime) = guard.as_mut() {
            for signal in handles {
                runtime.signal_state.release_async_signal(*signal);
            }
            if !retain_memory {
                for &base in borrowed_memory.iter().flatten() {
                    runtime.memory_state.release_copy_borrow(base);
                }
            }
        }
    }
    // Shutdown removes the runtime from the registry before joining workers;
    // it retains all signal storage until after this worker exits.
}

fn complete_async_signal_value(signal: &crate::signal::AmdSignal, success: bool) {
    if success {
        signal.value.fetch_sub(1, Ordering::Release);
    } else {
        signal.value.store(-1, Ordering::Release);
    }
}

fn finish_async_signal(completion: HsaSignal, success: bool) {
    if completion.handle == 0 {
        return;
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            complete_async_signal_value(signal, success);
        })
    };
}

fn finish_async_copy_signal(
    completion: HsaSignal,
    success: bool,
    profile: Option<AsyncCopyProfile>,
) {
    if completion.handle == 0 {
        return;
    }
    let profile = if success { profile } else { None };
    let mut guard = lock().ok();
    if let Some(runtime) = guard.as_mut().and_then(|guard| guard.as_mut()) {
        if let Some(record) = runtime
            .signal_state
            .async_signal_refs
            .get_mut(&(completion.handle as usize))
        {
            record.copy_profile = profile;
        }
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined. The
    // registry lock orders profile publication with the completion update.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            if let Some(profile) = profile {
                signal.start_ts.store(profile.start, Ordering::Relaxed);
                signal.end_ts.store(profile.end, Ordering::Relaxed);
                crate::signal::flush_signal_timestamps(signal);
            }
            complete_async_signal_value(signal, success);
        })
    };
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the worker owns the session and stop token for its full asynchronous lifetime"
)]
fn svm_prefetch_worker(
    session: Session,
    stop: Arc<AtomicBool>,
    dependencies: &[HsaSignal],
    completion: HsaSignal,
    address: u64,
    size: u64,
    location: SvmLocation,
) {
    if !wait_async_dependencies(dependencies, &stop) {
        finish_async_signal(completion, false);
        return;
    }
    let result = linux_interop::set_kfd_svm_attributes(
        &session,
        address,
        size,
        &[SvmAttribute::PrefetchLocation(location)],
    );
    finish_async_signal(completion, result.is_ok());
}

fn wait_async_dependencies(dependencies: &[HsaSignal], stop: &AtomicBool) -> bool {
    for &dependency in dependencies {
        loop {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            // SAFETY: Submission retains every dependency until the worker
            // exits; final shutdown joins workers before releasing storage.
            let Some(ready) = (unsafe {
                crate::signal::with_signal(dependency, |signal| {
                    signal.value.load(Ordering::Acquire) == 0
                })
            }) else {
                return false;
            };
            if ready {
                break;
            }
            thread::sleep(Duration::from_micros(20));
        }
    }
    true
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_prefetch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_svm_prefetch_async(
    pointer: *mut c_void,
    size: usize,
    agent: HsaAgent,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if dependency_count != 0 && dependencies.is_null() {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        let location = match svm_location_from_agent(runtime, agent, false) {
            Ok(location) => location,
            Err(status) => return status,
        };
        if dependency_slice
            .iter()
            .any(|dependency| !runtime.signal_state.owns_signal(*dependency))
            || (completion.handle != 0 && !runtime.signal_state.owns_signal(completion))
        {
            return INVALID_SIGNAL;
        }
        if completion.handle != 0 {
            let status = runtime.signal_state.async_completion_status(completion);
            if status != SUCCESS {
                return status;
            }
        }
        let mut signal_handles = Vec::new();
        runtime.memory_state.reap_finished_workers();
        if signal_handles
            .try_reserve_exact(dependency_slice.len() + usize::from(completion.handle != 0))
            .is_err()
            || runtime.memory_state.workers.try_reserve(1).is_err()
        {
            return OUT_OF_RESOURCES;
        }
        signal_handles.extend_from_slice(dependency_slice);
        if completion.handle != 0 {
            signal_handles.push(completion);
        }
        let mut pending = match PendingAsyncRetention::new(
            &mut runtime.signal_state,
            &mut runtime.memory_state,
            Arc::new(signal_handles),
        ) {
            Ok(pending) => pending,
            Err(status) => return status,
        };
        let session = runtime.session.clone();
        let stop = runtime.stop_workers.clone();
        let worker_handles = pending.handles.clone();
        let dependency_len = dependency_slice.len();
        let worker = match thread::Builder::new()
            .name("rocddi-svm-prefetch".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    svm_prefetch_worker(
                        session,
                        stop,
                        &worker_handles[..dependency_len],
                        completion,
                        address,
                        size,
                        location,
                    );
                }));
                if result.is_err() {
                    finish_async_signal(completion, false);
                }
                release_async_resources(&worker_handles, &[], false);
            }) {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        pending.transfer_to_worker();
        pending.memory.workers.push(worker);
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_svm_discard_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null() || sizes.is_null() || count == 0 {
            return INVALID_ARGUMENT;
        }
        // GFX1201 reports XNACK disabled through hsa_system_get_info, so ROCr
        // rejects this operation before inspecting dependencies or ranges.
        XNACK_DISABLED
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_and_prefetch_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_svm_discard_and_prefetch_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    destination_agents: *const HsaAgent,
    destination_count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null()
            || sizes.is_null()
            || count == 0
            || destination_agents.is_null()
            || destination_count == 0
        {
            return INVALID_ARGUMENT;
        }
        // See hsa_amd_svm_discard_batch_async: XNACK is unavailable on the
        // currently supported target, matching ROCr's required status.
        XNACK_DISABLED
    })
}

/// Execution path selected for an asynchronous HSA copy request.
enum AsyncCopyRoute {
    /// Zero-byte copy that only completes dependencies and its signal.
    Noop,
    /// CPU copy performed after dependency signals complete.
    Host {
        destination: HostCopyDestination,
        source: HostCopySource,
        size: usize,
    },
    /// One native SDMA operation on an activated GPU.
    Gpu {
        device: Box<Device>,
        index: usize,
        entry: GpuCopyEntry,
    },
    /// Several ordered SDMA operations sharing one native sequence.
    GpuBatch {
        device: Box<Device>,
        index: usize,
        entries: Vec<GpuCopyEntry>,
        controls: BatchCopyControls,
    },
}

/// One operation within a native GPU copy route.
enum GpuCopyEntry {
    /// Rectangular copy with row and slice strides.
    Rect(CopyRect),
    /// Direct linear copy.
    Linear {
        destination: u64,
        source: u64,
        size: u64,
    },
    /// Exchange between two GPU address ranges.
    Swap {
        address_a: u64,
        address_b: u64,
        size: u64,
    },
    /// Copy whose addresses may come from GPU-visible pointer slots.
    Indirect {
        destination: u64,
        source: u64,
        size: u64,
        indirect_source: bool,
        indirect_destination: bool,
    },
    /// Copy from host storage staged for SDMA access.
    FromHost {
        destination: u64,
        source: HostCopySource,
    },
    /// Copy to a host destination staged for SDMA access.
    ToHost {
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

/// Optional wait and signal attached to a native batch copy.
#[derive(Clone, Copy, Default)]
struct BatchCopyControls {
    /// GPU comparison that must pass before the batch begins.
    wait: Option<GpuCopyWait>,
    /// GPU atomic notification after the final packet.
    signal: Option<GpuCopySignal>,
}

/// Native route selected for one synchronous HSA copy.
enum SyncGpuCopyRoute {
    /// Direct copy between GPU-addressable ranges.
    Direct {
        device: Box<Device>,
        destination: u64,
        source: u64,
        size: u64,
    },
    /// Copy from a CPU-only source through staging.
    FromHost {
        device: Box<Device>,
        destination: u64,
        source: HostCopySource,
    },
    /// Copy into a CPU-only destination through staging.
    ToHost {
        device: Box<Device>,
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

/// CPU source storage and when its bytes must be captured.
enum HostCopySource {
    /// Direct host pointer and byte extent.
    Address(usize, usize),
    /// Loader-owned mapping resolved after dependencies finish.
    LoadedAddress(usize, usize),
    /// Owned snapshot of a loader mapping for immediate copy.
    LoadedBytes(Vec<u8>),
}

/// Point at which a loader-owned source is read.
#[derive(Clone, Copy)]
enum HostSourceTiming {
    /// Snapshot before releasing the registry lock.
    Immediate,
    /// Resolve the live loader mapping after dependency completion.
    AfterDependencies,
}

/// CPU destination, including loader mappings needing a later lookup.
enum HostCopyDestination {
    /// Direct host address.
    Address(usize),
    /// Loader-owned address resolved while the registry lock is held.
    LoadedAddress(usize),
}

/// Completion data that must be published after a synchronous GPU copy.
enum SyncGpuCopyOutcome {
    /// Native and host copies completed without deferred publication.
    Done,
    /// Staged bytes to write to a loader-owned mapping.
    LoadedDestination { address: usize, bytes: Vec<u8> },
}

/// Frontend status and whether native work may still read its operands.
struct CopyRouteFailure {
    /// HSA status translated from validation or native failure.
    status: Status,
    /// Requires retaining borrowed memory until retirement is proved.
    operands_may_be_live: bool,
}

impl CopyRouteFailure {
    fn retired(status: Status) -> Self {
        Self {
            status,
            operands_may_be_live: false,
        }
    }
}

impl From<rocddi::device::gpu::CopyFailure> for CopyRouteFailure {
    fn from(failure: rocddi::device::gpu::CopyFailure) -> Self {
        Self {
            status: map_error(failure.error),
            operands_may_be_live: failure.operands_may_be_live,
        }
    }
}

/// Synchronous route and public allocations borrowed by its operands.
struct ResolvedSyncGpuCopy {
    /// Native copy plan selected after pointer validation.
    route: SyncGpuCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    borrowed_memory: [Option<usize>; 2],
}

/// Asynchronous route and public allocations borrowed by its operands.
struct ResolvedAsyncCopy {
    /// Native or host copy plan selected after pointer validation.
    route: AsyncCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    borrowed_memory: [Option<usize>; 2],
}

fn copy_agent(
    runtime: &HsaState,
    requested: HsaAgent,
    description: Option<&PointerDescription>,
) -> Result<HsaAgent, Status> {
    let agent = if requested.handle == 0 {
        description.map_or(HsaAgent { handle: CPU_AGENT }, |description| {
            description.owner
        })
    } else {
        requested
    };
    runtime
        .is_agent(agent)
        .then_some(agent)
        .ok_or(INVALID_AGENT)
}

fn selected_copy_gpu(
    runtime: &HsaState,
    description: &PointerDescription,
) -> Option<(usize, HsaAgent)> {
    std::iter::once(description.owner)
        .chain(description.accessible.iter().copied())
        .filter(|agent| description.accessible.contains(agent))
        .find_map(|agent| runtime.gpu_index(agent).map(|index| (index, agent)))
}

fn copy_pointer_description(
    runtime: &HsaState,
    pointer: usize,
    size: usize,
) -> Result<Option<PointerDescription>, Status> {
    if pointer.checked_add(size).is_none() {
        return Err(INVALID_ARGUMENT);
    }
    let description = pointer_description(runtime, pointer);
    if let Some(description) = &description {
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return Err(INVALID_ALLOCATION);
        }
        if description.offset_range(pointer, size).is_none() {
            return Err(INVALID_ALLOCATION);
        }
    }
    Ok(description)
}

fn gpu_copy_address(
    runtime: &HsaState,
    description: Option<&PointerDescription>,
    pointer: usize,
    size: usize,
    gpu: HsaAgent,
    required_access: u32,
) -> Result<(u64, usize), Status> {
    let description = description.ok_or(INVALID_ALLOCATION)?;
    if !description.accessible.contains(&gpu) {
        return Err(INVALID_AGENT);
    }
    vmem_copy_access(runtime, description, gpu, required_access)?;
    let offset = description
        .offset_range(pointer, size)
        .ok_or(INVALID_ALLOCATION)?;
    let address = description
        .agent_base
        .checked_add(offset)
        .ok_or(INVALID_ALLOCATION)?;
    Ok((
        u64::try_from(address).map_err(|_| INVALID_ALLOCATION)?,
        description.agent_base,
    ))
}

/// Resolves a caller-owned image backing range to this agent's GPU address.
/// Image data lifetime remains the caller's responsibility under HSA.
pub(crate) fn image_device_address(
    runtime: &HsaState,
    pointer: usize,
    size: usize,
    agent: HsaAgent,
    permission: u32,
) -> Result<u64, Status> {
    let description = copy_pointer_description(runtime, pointer, size)?;
    gpu_copy_address(
        runtime,
        description.as_ref(),
        pointer,
        size,
        agent,
        permission,
    )
    .map(|(address, _)| address)
}

fn prepare_host_copy_source(
    runtime: &HsaState,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
    timing: HostSourceTiming,
) -> Result<HostCopySource, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
    {
        if matches!(timing, HostSourceTiming::AfterDependencies) {
            return Ok(HostCopySource::LoadedAddress(pointer, size));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| OUT_OF_RESOURCES)?;
        bytes.resize(size, 0);
        // SAFETY: The registry lock retains this loaded mapping while the
        // complete range is copied into owned host storage.
        unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
        Ok(HostCopySource::LoadedBytes(bytes))
    } else {
        Ok(HostCopySource::Address(host, size))
    }
}

fn prepare_host_copy_destination(
    runtime: &HsaState,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
) -> Result<HostCopyDestination, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
    {
        Ok(HostCopyDestination::LoadedAddress(pointer))
    } else {
        Ok(HostCopyDestination::Address(host))
    }
}

pub(super) fn vmem_copy_access(
    runtime: &HsaState,
    description: &PointerDescription,
    agent: HsaAgent,
    required_access: u32,
) -> Result<(), Status> {
    if description.pointer_type != POINTER_TYPE_HSA_VMEM {
        return Ok(());
    }
    let mapping = runtime
        .memory_state
        .vmem_mappings
        .get(&description.agent_base)
        .ok_or(INVALID_ALLOCATION)?;
    if mapping
        .access
        .get(&agent.handle)
        .is_none_or(|access| access.permissions & required_access != required_access)
    {
        return Err(INVALID_ALLOCATION);
    }
    Ok(())
}

fn resolve_async_copy(
    runtime: &HsaState,
    destination: usize,
    dst_agent: HsaAgent,
    source: usize,
    src_agent: HsaAgent,
    size: usize,
    copy_gpu_index: Option<usize>,
) -> Result<ResolvedAsyncCopy, Status> {
    if size == 0 {
        if (dst_agent.handle != 0 && !runtime.is_agent(dst_agent))
            || (src_agent.handle != 0 && !runtime.is_agent(src_agent))
        {
            return Err(INVALID_AGENT);
        }
        return Ok(ResolvedAsyncCopy {
            route: AsyncCopyRoute::Noop,
            borrowed_memory: [None, None],
        });
    }
    let dst_description = copy_pointer_description(runtime, destination, size)?;
    let src_description = copy_pointer_description(runtime, source, size)?;
    let dst_agent = copy_agent(runtime, dst_agent, dst_description.as_ref())?;
    let src_agent = copy_agent(runtime, src_agent, src_description.as_ref())?;
    let dst_gpu = runtime.gpu_index(dst_agent);
    let src_gpu = runtime.gpu_index(src_agent);
    if let Some(index) = copy_gpu_index.or(src_gpu.or(dst_gpu)) {
        if !runtime.gpus[index]
            .device
            .gpu()
            .is_ok_and(|gpu| gpu.supports_linear_copy())
        {
            return Err(NOT_SUPPORTED);
        }
        let gpu_agent = HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        };
        let gpu_destination = gpu_copy_address(
            runtime,
            dst_description.as_ref(),
            destination,
            size,
            gpu_agent,
            ACCESS_PERMISSION_WO,
        );
        let gpu_source = gpu_copy_address(
            runtime,
            src_description.as_ref(),
            source,
            size,
            gpu_agent,
            ACCESS_PERMISSION_RO,
        );
        let device = Box::new(runtime.gpus[index].device.clone());
        return match (gpu_destination, gpu_source) {
            (Ok((destination, dst_base)), Ok((source, src_base))) => Ok(ResolvedAsyncCopy {
                route: AsyncCopyRoute::Gpu {
                    device,
                    index,
                    entry: GpuCopyEntry::Linear {
                        destination,
                        source,
                        size: u64::try_from(size).map_err(|_| INVALID_ARGUMENT)?,
                    },
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            }),
            (Ok((destination, dst_base)), Err(_)) if src_gpu.is_none() => {
                let host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
                let source = prepare_host_copy_source(
                    runtime,
                    src_description.as_ref(),
                    source,
                    host,
                    size,
                    HostSourceTiming::AfterDependencies,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        entry: GpuCopyEntry::FromHost {
                            destination,
                            source,
                        },
                    },
                    borrowed_memory: [
                        Some(dst_base),
                        src_description.map(|description| description.agent_base),
                    ],
                })
            }
            (Err(_), Ok((source, src_base))) if dst_gpu.is_none() => {
                let host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
                let destination = prepare_host_copy_destination(
                    runtime,
                    dst_description.as_ref(),
                    destination,
                    host,
                    size,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        entry: GpuCopyEntry::ToHost {
                            destination,
                            source,
                            size,
                        },
                    },
                    borrowed_memory: [
                        dst_description.map(|description| description.agent_base),
                        Some(src_base),
                    ],
                })
            }
            (Err(status), _) | (_, Err(status)) => Err(status),
        };
    }
    let destination_host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
    let source_host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
    let destination = prepare_host_copy_destination(
        runtime,
        dst_description.as_ref(),
        destination,
        destination_host,
        size,
    )?;
    let source = prepare_host_copy_source(
        runtime,
        src_description.as_ref(),
        source,
        source_host,
        size,
        HostSourceTiming::AfterDependencies,
    )?;
    Ok(ResolvedAsyncCopy {
        route: AsyncCopyRoute::Host {
            destination,
            source,
            size,
        },
        borrowed_memory: [
            dst_description.map(|description| description.agent_base),
            src_description.map(|description| description.agent_base),
        ],
    })
}

fn resolve_sync_gpu_copy(
    runtime: &HsaState,
    destination: usize,
    source: usize,
    size: usize,
    host_destination: Result<usize, Status>,
    host_source: Result<usize, Status>,
) -> Result<ResolvedSyncGpuCopy, Status> {
    match (host_destination, host_source) {
        (Err(_), Ok(host_source)) => {
            let description =
                copy_pointer_description(runtime, destination, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (destination, destination_base) = gpu_copy_address(
                runtime,
                Some(&description),
                destination,
                size,
                agent,
                ACCESS_PERMISSION_WO,
            )?;
            let source_description = copy_pointer_description(runtime, source, size)?;
            let source_base = source_description
                .as_ref()
                .map(|description| description.agent_base);
            let source = prepare_host_copy_source(
                runtime,
                source_description.as_ref(),
                source,
                host_source,
                size,
                HostSourceTiming::Immediate,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::FromHost {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    destination,
                    source,
                },
                borrowed_memory: [Some(destination_base), source_base],
            })
        }
        (Ok(host_destination), Err(_)) => {
            let description =
                copy_pointer_description(runtime, source, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (source, source_base) = gpu_copy_address(
                runtime,
                Some(&description),
                source,
                size,
                agent,
                ACCESS_PERMISSION_RO,
            )?;
            let destination_description = copy_pointer_description(runtime, destination, size)?;
            let destination_base = destination_description
                .as_ref()
                .map(|description| description.agent_base);
            let destination = prepare_host_copy_destination(
                runtime,
                destination_description.as_ref(),
                destination,
                host_destination,
                size,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::ToHost {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    destination,
                    source,
                    size,
                },
                borrowed_memory: [destination_base, Some(source_base)],
            })
        }
        (Err(_), Err(_)) => {
            let resolved = resolve_async_copy(
                runtime,
                destination,
                HsaAgent { handle: 0 },
                source,
                HsaAgent { handle: 0 },
                size,
                None,
            )?;
            let AsyncCopyRoute::Gpu {
                device,
                entry:
                    GpuCopyEntry::Linear {
                        destination,
                        source,
                        size,
                    },
                ..
            } = resolved.route
            else {
                return Err(INVALID_ALLOCATION);
            };
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::Direct {
                    device,
                    destination,
                    source,
                    size,
                },
                borrowed_memory: resolved.borrowed_memory,
            })
        }
        (Ok(_), Ok(_)) => Err(ERROR),
    }
}

fn copy_gpu(device: &Device) -> Result<rocddi::device::gpu::GpuDevice<'_>, CopyRouteFailure> {
    device
        .gpu()
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))
}

fn loaded_source_snapshot(address: usize, size: usize) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| OUT_OF_RESOURCES)?;
    bytes.resize(size, 0);
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    let host = crate::loader::loaded_host_range(&runtime.loader, address, size)?
        .ok_or(INVALID_ALLOCATION)?;
    // SAFETY: The registry lock retains the complete loaded mapping while the
    // source is snapshotted after the dependency signals have been satisfied.
    unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
    Ok(bytes)
}

fn with_host_copy_source<T>(
    source: &HostCopySource,
    copy: impl FnOnce(&[u8]) -> Result<T, CopyRouteFailure>,
) -> Result<T, CopyRouteFailure> {
    let snapshot = if let HostCopySource::LoadedAddress(address, size) = source {
        loaded_source_snapshot(*address, *size).map_err(CopyRouteFailure::retired)?
    } else {
        Vec::new()
    };
    let host = match source {
        HostCopySource::Address(address, size) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            unsafe { std::slice::from_raw_parts(*address as *const u8, *size) }
        }
        HostCopySource::LoadedAddress(_, _) => snapshot.as_slice(),
        HostCopySource::LoadedBytes(bytes) => bytes,
    };
    copy(host)
}

fn copy_from_host_route(
    device: &Device,
    destination: u64,
    source: &HostCopySource,
    stop: &AtomicBool,
) -> Result<(), CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    with_host_copy_source(source, |host| {
        // SAFETY: HSA checked the GPU destination and retains its backing. rocddi
        // copies host bytes into owned staging before native submission.
        unsafe { gpu.copy_from_host(destination, host, stop) }.map_err(Into::into)
    })
}

fn copy_to_host_destination(
    destination: &HostCopyDestination,
    size: usize,
    copy: impl FnOnce(&mut [u8]) -> Result<(), CopyRouteFailure>,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match destination {
        HostCopyDestination::Address(address) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            let host = unsafe { std::slice::from_raw_parts_mut(*address as *mut u8, size) };
            copy(host)?;
            fence(Ordering::SeqCst);
            Ok(SyncGpuCopyOutcome::Done)
        }
        HostCopyDestination::LoadedAddress(address) => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(size)
                .map_err(|_| CopyRouteFailure::retired(OUT_OF_RESOURCES))?;
            bytes.resize(size, 0);
            copy(&mut bytes)?;
            Ok(SyncGpuCopyOutcome::LoadedDestination {
                address: *address,
                bytes,
            })
        }
    }
}

fn copy_to_host_route(
    device: &Device,
    destination: &HostCopyDestination,
    source: u64,
    size: usize,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    copy_to_host_destination(destination, size, |host| {
        // SAFETY: HSA checked the GPU source and retains its backing. rocddi
        // writes the host buffer only after native retirement.
        unsafe { gpu.copy_to_host(host, source, stop) }.map_err(Into::into)
    })
}

fn execute_sync_gpu_copy(
    route: SyncGpuCopyRoute,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match route {
        SyncGpuCopyRoute::Direct {
            device,
            destination,
            source,
            size,
        } => {
            let gpu = copy_gpu(&device)?;
            // SAFETY: HSA checked both mapped GPU ranges before retaining
            // their backing through this synchronous call.
            unsafe { gpu.copy_linear(destination, source, size, stop) }
                .map_err(CopyRouteFailure::from)?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::FromHost {
            device,
            destination,
            source,
        } => {
            copy_from_host_route(&device, destination, &source, stop)?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::ToHost {
            device,
            destination,
            source,
            size,
        } => copy_to_host_route(&device, &destination, source, size, stop),
    }
}

fn write_loaded_copy_destination(address: usize, bytes: &[u8]) -> Status {
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    let host = match crate::loader::loaded_host_range(&runtime.loader, address, bytes.len()) {
        Ok(Some(host)) => host,
        Ok(None) | Err(_) => return INVALID_ALLOCATION,
    };
    // SAFETY: The registry lock retains the complete loaded mapping until
    // the copied bytes and their system-scope ordering are published.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len()) };
    fence(Ordering::SeqCst);
    SUCCESS
}

fn execute_gpu_entries(
    device: &Device,
    entries: impl ExactSizeIterator<Item = GpuCopyEntry>,
    controls: BatchCopyControls,
    sdma_engine: Option<u32>,
    timed: bool,
    stop: &AtomicBool,
) -> Result<Option<GpuCopyTimestamps>, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    let mut sequence = match sdma_engine {
        Some(engine) => GpuCopySequence::begin_on_sdma_engine(gpu, stop, engine),
        None => GpuCopySequence::begin(gpu, stop),
    }
    .map_err(CopyRouteFailure::from)?;
    if timed {
        sequence.enable_timing().map_err(CopyRouteFailure::from)?;
    }
    let count = entries.len();
    for (ordinal, entry) in entries.enumerate() {
        let options = GpuCopyOptions {
            wait: if ordinal == 0 { controls.wait } else { None },
            signal: if ordinal + 1 == count {
                controls.signal
            } else {
                None
            },
            ..GpuCopyOptions::default()
        };
        let outcome = match entry {
            GpuCopyEntry::Rect(rect) => {
                // SAFETY: Preparation validated every selected row and keeps
                // its mapped backing live until native retirement.
                unsafe { sequence.copy_rect(rect) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Linear {
                destination,
                source,
                size,
            } => {
                // SAFETY: Copy preparation checked both mapped GPU ranges and
                // retains their owners until native retirement is proved.
                unsafe { sequence.copy_linear_with_options(destination, source, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Swap {
                address_a,
                address_b,
                size,
            } => {
                // SAFETY: Preparation verified read and write access to both
                // ranges on the selected GPU and retained their owners.
                unsafe { sequence.swap_linear_with_options(address_a, address_b, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Indirect {
                destination,
                source,
                size,
                indirect_source,
                indirect_destination,
            } => {
                // SAFETY: Preparation checked access to every direct range and
                // indirect pointer slot. The caller owns the resolved targets
                // through completion, including after an uncertain submission.
                unsafe {
                    sequence.copy_linear_with_options(
                        destination,
                        source,
                        size,
                        GpuCopyOptions {
                            indirect_source,
                            indirect_destination,
                            ..options
                        },
                    )
                }
                .map(|()| SyncGpuCopyOutcome::Done)
                .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::FromHost {
                destination,
                source,
            } => with_host_copy_source(&source, |host| {
                // SAFETY: HSA checked the GPU destination and retains its
                // backing. rocddi owns the staged copy of these host bytes.
                unsafe { sequence.copy_from_host(destination, host) }
                    .map_err(CopyRouteFailure::from)
            })
            .map(|()| SyncGpuCopyOutcome::Done),
            GpuCopyEntry::ToHost {
                destination,
                source,
                size,
            } => copy_to_host_destination(&destination, size, |host| {
                // SAFETY: HSA checked the GPU source and retains its backing.
                // rocddi writes the host slice after native retirement.
                unsafe { sequence.copy_to_host(host, source) }.map_err(CopyRouteFailure::from)
            }),
        }?;
        if let SyncGpuCopyOutcome::LoadedDestination { address, bytes } = outcome {
            let status = write_loaded_copy_destination(address, &bytes);
            if status != SUCCESS {
                return Err(CopyRouteFailure::retired(status));
            }
        }
    }
    if timed {
        sequence
            .finish_timing()
            .map(Some)
            .map_err(CopyRouteFailure::from)
    } else {
        Ok(None)
    }
}

fn system_copy_tick(device: Option<&Device>) -> Result<u64, CopyRouteFailure> {
    let device = device.ok_or_else(|| CopyRouteFailure::retired(ERROR))?;
    let counters = crate::runtime::clock_counters(device)
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))?;
    if counters.system == 0 {
        return Err(CopyRouteFailure::retired(ERROR));
    }
    Ok(counters.system)
}

fn gpu_copy_profile(index: usize, ticks: Option<GpuCopyTimestamps>) -> Option<AsyncCopyProfile> {
    ticks.map(|ticks| AsyncCopyProfile {
        clock: AsyncCopyClock::Gpu(index),
        start: ticks.start,
        end: ticks.end,
    })
}

fn execute_copy_route(
    route: AsyncCopyRoute,
    sdma_engine: Option<u32>,
    profile_enabled: bool,
    host_clock: Option<&Device>,
    stop: &AtomicBool,
) -> Result<Option<AsyncCopyProfile>, CopyRouteFailure> {
    match route {
        AsyncCopyRoute::Noop => Ok(None),
        AsyncCopyRoute::Host {
            destination,
            source,
            size,
        } => {
            if stop.load(Ordering::Acquire) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            let start = profile_enabled
                .then(|| system_copy_tick(host_clock))
                .transpose()?;
            let needs_loaded_mapping = matches!(&source, HostCopySource::LoadedAddress(..))
                || matches!(&destination, HostCopyDestination::LoadedAddress(..));
            let guard = if needs_loaded_mapping {
                match lock() {
                    Ok(guard) => Some(guard),
                    Err(_) => return Err(CopyRouteFailure::retired(ERROR)),
                }
            } else {
                None
            };
            let runtime = guard.as_ref().and_then(|guard| guard.as_ref());
            let loaded_host = |address| {
                runtime.and_then(|runtime| {
                    crate::loader::loaded_host_range(&runtime.loader, address, size)
                        .ok()
                        .flatten()
                })
            };
            let source_host = match &source {
                HostCopySource::Address(address, _) => *address,
                HostCopySource::LoadedAddress(address, _) => {
                    let Some(host) = loaded_host(*address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
                HostCopySource::LoadedBytes(bytes) => bytes.as_ptr() as usize,
            };
            let destination_host = match destination {
                HostCopyDestination::Address(address) => address,
                HostCopyDestination::LoadedAddress(address) => {
                    let Some(host) = loaded_host(address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
            };
            // SAFETY: The caller keeps ordinary host ranges live through
            // completion. HsaState-owned memory is borrowed, and the registry
            // lock keeps loaded mappings live for this copy. ptr::copy also
            // permits overlapping ranges.
            unsafe { std::ptr::copy(source_host as *const u8, destination_host as *mut u8, size) };
            fence(Ordering::SeqCst);
            drop(guard);
            let end = profile_enabled
                .then(|| system_copy_tick(host_clock))
                .transpose()?;
            let profile = start.zip(end).map(|(start, end)| AsyncCopyProfile {
                clock: AsyncCopyClock::System,
                start,
                end,
            });
            if profile.is_some_and(|profile| profile.end < profile.start) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            Ok(profile)
        }
        AsyncCopyRoute::Gpu {
            device,
            index,
            entry,
        } => execute_gpu_entries(
            &device,
            std::iter::once(entry),
            BatchCopyControls::default(),
            sdma_engine,
            profile_enabled,
            stop,
        )
        .map(|ticks| gpu_copy_profile(index, ticks)),
        AsyncCopyRoute::GpuBatch {
            device,
            index,
            entries,
            controls,
        } => execute_gpu_entries(
            &device,
            entries.into_iter(),
            controls,
            None,
            profile_enabled,
            stop,
        )
        .map(|ticks| gpu_copy_profile(index, ticks)),
    }
}

mod async_copy;
#[cfg(test)]
pub(super) use async_copy::*;
