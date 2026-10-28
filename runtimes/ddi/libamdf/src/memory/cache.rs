// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host and GPU cache transitions for AMDF memory access pairs.

use super::*;
use rocddi::host::cache::CachePolicy;

/// One producer or consumer site used to derive cache-transition requirements.
#[derive(Clone, Copy)]
pub(super) struct Site {
    /// Public memory resource participating in the transition.
    pub(super) memory: *mut Memory,
    /// Whether the participant is the host rather than a device.
    pub(super) host: bool,
    /// Access mode required at this participant.
    pub(super) access: u32,
    /// Whether host and device accesses are coherent here.
    pub(super) host_coherent: bool,
    /// Native or host cacheability classification.
    pub(super) cacheability: u32,
    /// Release operation before control passes away from this site.
    pub(super) release: amdf_cache_transition_t,
    /// Acquire operation before this site consumes the data.
    pub(super) acquire: amdf_cache_transition_t,
}

pub(super) fn host_transitions(
    instance: &Instance,
    cacheability: amdf_host_cacheability_t,
) -> Result<
    (
        Option<CacheMaintenance>,
        u32,
        amdf_cache_transition_t,
        amdf_cache_transition_t,
    ),
    u64,
> {
    let (maintenance, line, flush) = match cacheability {
        AMDF_HOST_CACHEABILITY_WRITE_BACK => {
            let maintenance = instance.host_write_back_cache.ok_or(UNSUPPORTED)?;
            let line = maintenance.line_size_bytes().ok_or(INTERNAL)?;
            (
                Some(maintenance),
                line,
                amdf_cache_transition_t {
                    kind: AMDF_CACHE_TRANSITION_KIND_RANGE,
                    executor: AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                    host_operation: AMDF_HOST_CACHE_OPERATION_FLUSH,
                    host_instruction: AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
                    host_fence_before: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                    host_fence_after: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                    range_granularity: u64::from(line),
                    ..Default::default()
                },
            )
        }
        AMDF_HOST_CACHEABILITY_WRITE_COMBINED => {
            let maintenance = CacheMaintenance::qualify(CachePolicy::WriteCombined)
                .map_err(|error| native(&error))?;
            (
                Some(maintenance),
                0,
                amdf_cache_transition_t {
                    kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                    executor: AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                    host_operation: AMDF_HOST_CACHE_OPERATION_FLUSH,
                    host_fence_after: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                    ..Default::default()
                },
            )
        }
        AMDF_HOST_CACHEABILITY_UNCACHED => (None, 0, no_transition()),
        _ => (None, 0, amdf_cache_transition_t::default()),
    };
    let mut invalidate = flush;
    if invalidate.executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT {
        invalidate.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
    }
    Ok((maintenance, line, flush, invalidate))
}

pub(super) fn describe_pair(
    producer: &Site,
    consumer: &Site,
) -> Result<amdf_memory_pair_info_t, u64> {
    // Coherent GPU access can eliminate CPU line maintenance on WB mappings,
    // but not the caller's publication or completion ordering. WC views retain
    // their fences even when the peer snoops host caches.
    let release = if producer.host
        && producer.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK
        && consumer.host_coherent
    {
        no_transition()
    } else {
        producer.release
    };
    let acquire = if consumer.host
        && consumer.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK
        && producer.host_coherent
    {
        no_transition()
    } else {
        consumer.acquire
    };
    if producer.access & AMDF_MEMORY_ACCESS_WRITE == 0
        || consumer.access & AMDF_MEMORY_ACCESS_READ == 0
        || release.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN
        || acquire.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN
    {
        return Err(UNSUPPORTED);
    }
    Ok(amdf_memory_pair_info_t {
        flags: AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE
            | if release.kind == AMDF_CACHE_TRANSITION_KIND_NONE
                && acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE
            {
                AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN
            } else {
                0
            },
        release,
        acquire,
        ..Default::default()
    })
}

pub(super) fn device_site(
    memory: *mut Memory,
    memory_class: u32,
    flags: u64,
    access: u32,
    family: &amdf_queue_family_info_t,
    local_wc_single_owner: bool,
) -> Site {
    // SYSTEM uses the family-qualified cache protocol. GFX1201 PM4, AQL, and
    // SDMA also qualify host-visible WC LOCAL storage with one owner VM;
    // peer-device LOCAL visibility needs separate evidence.
    let system =
        memory_class == AMDF_MEMORY_CLASS_SYSTEM && flags & AMDF_MEMORY_FLAG_HOST_COHERENT != 0;
    let qualified_local_family = (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
        && family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1
        && family.format_features == AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR)
        || (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
            && family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1
            && family.format_features
                == AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR
                    | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM)
        || (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            && family.format_version == 1
            && family.format_features == 0);
    let local = memory_class == AMDF_MEMORY_CLASS_LOCAL
        && flags & AMDF_MEMORY_FLAG_HOST_VISIBLE != 0
        && local_wc_single_owner
        && qualified_local_family;
    let qualified = (system || local)
        && family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL != 0;
    let transition = |operation, supported| {
        if qualified && family.cache_operations & supported != 0 {
            amdf_cache_transition_t {
                kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                executor: AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
                operation,
                ..Default::default()
            }
        } else {
            amdf_cache_transition_t::default()
        }
    };
    Site {
        memory,
        host: false,
        access,
        host_coherent: flags & AMDF_MEMORY_FLAG_HOST_COHERENT != 0,
        cacheability: AMDF_HOST_CACHEABILITY_UNKNOWN,
        release: transition(
            AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
        ),
        acquire: transition(
            AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        ),
    }
}

unsafe fn profile_site(
    scope: &Scope,
    profile: &amdf_memory_profile_t,
    capabilities: &[amdf_memory_access_capabilities_t],
    accesses: &[amdf_memory_device_access_t],
    required_flags: amdf_memory_flags_t,
    registered_host_cacheability: amdf_host_cacheability_t,
    site: amdf_memory_profile_site_t,
) -> Result<Site, u64> {
    if site.reserved != 0 {
        return Err(INVALID);
    }
    match site.kind {
        AMDF_MEMORY_SITE_KIND_DEVICE => {
            let value = unsafe { site.value.device };
            let index = usize::try_from(value.access_ordinal).map_err(|_| RANGE)?;
            let access = accesses.get(index).ok_or(RANGE)?;
            let capability = capabilities.get(index).ok_or(INTERNAL)?;
            let device = unsafe { object(access.device.cast::<Device>())? };
            let family =
                crate::instance::family(device.native.endpoint(), value.queue_family_ordinal)?;
            Ok(device_site(
                std::ptr::null_mut(),
                profile.memory_class,
                capability.guaranteed_flags | access.requirements.flags | required_flags,
                access.requirements.access,
                &family,
                accesses.len() == 1
                    && matches!(
                        device.native.endpoint().host_local_cacheability,
                        Some(memory::device::HostCacheability::WriteCombined)
                    ),
            ))
        }
        AMDF_MEMORY_SITE_KIND_HOST | AMDF_MEMORY_SITE_KIND_FILE_IO => {
            let access = unsafe { site.value.host_access };
            if access == 0
                || access & !(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE) != 0
            {
                return Err(INVALID);
            }
            if required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE == 0
                || profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP == 0
                || access & !profile.host_mapping.supported_access != 0
                || (site.kind == AMDF_MEMORY_SITE_KIND_FILE_IO
                    && profile.host_mapping.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO == 0)
            {
                return Err(UNSUPPORTED);
            }
            let cacheability = if profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0 {
                registered_host_cacheability
            } else if profile.memory_class == AMDF_MEMORY_CLASS_LOCAL {
                match unsafe { (*scope.endpoint).native.host_local_cacheability } {
                    Some(memory::device::HostCacheability::WriteBack) => {
                        AMDF_HOST_CACHEABILITY_WRITE_BACK
                    }
                    Some(memory::device::HostCacheability::WriteCombined) => {
                        AMDF_HOST_CACHEABILITY_WRITE_COMBINED
                    }
                    None => return Err(UNSUPPORTED),
                }
            } else {
                AMDF_HOST_CACHEABILITY_WRITE_BACK
            };
            let (_, _, release, acquire) =
                host_transitions(unsafe { &*scope.instance }, cacheability)?;
            Ok(Site {
                memory: std::ptr::null_mut(),
                host: true,
                access: if access & AMDF_MEMORY_MAP_FLAG_READ != 0 {
                    AMDF_MEMORY_ACCESS_READ
                } else {
                    0
                } | if access & AMDF_MEMORY_MAP_FLAG_WRITE != 0 {
                    AMDF_MEMORY_ACCESS_WRITE
                } else {
                    0
                },
                host_coherent: true,
                cacheability,
                release,
                acquire,
            })
        }
        _ => Err(INVALID),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "profile qualification validates one atomic caller contract"
)]
pub(crate) unsafe extern "C" fn scope_pair_info(
    pointer: *mut amdf_memory_scope_t,
    query: *const amdf_memory_profile_pair_query_t,
    out: *mut amdf_memory_pair_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let scope = object(pointer.cast::<Scope>())?;
        let query = input(query, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY)?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO)?;
        if query.reserved != 0
            || query.external_memory_reserved != 0
            || query.required_flags & !BACKING_FLAGS != 0
            || query.external_memory_type > AMDF_EXTERNAL_MEMORY_TYPE_COUNT
        {
            return Err(INVALID);
        }
        let accesses = array(query.accesses, query.access_count)?;
        if !unique_devices(accesses) {
            return Err(INVALID);
        }
        let mut staged =
            OwnerVec::try_with_allocator((*scope.instance).host_allocator, accesses.len())
                .map_err(|_| EXHAUSTED)?;
        for access in accesses {
            requirements(access.requirements)?;
            let device = object(access.device.cast::<Device>())?;
            if (*device.endpoint).instance != scope.instance {
                return Err(INVALID);
            }
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: linux_interop::supports_system_dma_buf_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) = joint_profile(scope, query.memory_profile_ordinal, &staged)?;
        if query.required_flags & !profile.supported_flags != 0 {
            return Err(UNSUPPORTED);
        }
        let registered = profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0;
        if registered {
            if query.registered_host_cacheability
                != profile.registration.registered_host_cacheability
            {
                return Err(UNSUPPORTED);
            }
        } else if query.registered_host_cacheability != AMDF_HOST_CACHEABILITY_UNKNOWN {
            return Err(INVALID);
        }
        if query.external_memory_type == AMDF_EXTERNAL_MEMORY_TYPE_NONE {
            if has_identity(query.external_memory_provenance.words) {
                return Err(INVALID);
            }
        } else {
            let count =
                usize::try_from(profile.external_memory_support_count).map_err(|_| INTERNAL)?;
            let support = profile
                .external_memory_support
                .get(..count)
                .and_then(|values| {
                    values
                        .iter()
                        .find(|value| value.r#type == query.external_memory_type)
                })
                .ok_or(UNSUPPORTED)?;
            if support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT == 0 {
                return Err(UNSUPPORTED);
            }
            if support.provenance.words != query.external_memory_provenance.words {
                return Err(UNSUPPORTED);
            }
        }
        let producer = profile_site(
            scope,
            &profile,
            &capabilities,
            accesses,
            query.required_flags,
            query.registered_host_cacheability,
            query.producer,
        )?;
        let consumer = profile_site(
            scope,
            &profile,
            &capabilities,
            accesses,
            query.required_flags,
            query.registered_host_cacheability,
            query.consumer,
        )?;
        out.publish(describe_pair(&producer, &consumer)?);
        Ok(())
    })
}

unsafe fn site(pointer: *const amdf_memory_site_t) -> Result<Site, u64> {
    let site = unsafe { input(pointer, AMDF_STRUCTURE_TYPE_MEMORY_SITE)? };
    if site.reserved != 0 {
        return Err(INVALID);
    }
    match site.kind {
        AMDF_MEMORY_SITE_KIND_HOST | AMDF_MEMORY_SITE_KIND_FILE_IO => {
            let host = unsafe { site.value.host };
            let mapping = unsafe { object(host.mapping.cast::<Mapping>())? };
            if site.kind == AMDF_MEMORY_SITE_KIND_FILE_IO
                && mapping.info.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO == 0
            {
                return Err(UNSUPPORTED);
            }
            let memory = match mapping.owner {
                MappingOwner::Memory(memory) if host.source_memory.is_null() => memory,
                MappingOwner::Reservation(reservation) => {
                    if host.source_memory.is_null() {
                        return Err(INVALID);
                    }
                    let (source, _) = unsafe {
                        host_mapping_source(host.source_memory, (*reservation).instance)?
                    };
                    if !source.is_remappable() {
                        return Err(UNSUPPORTED);
                    }
                    host.source_memory.cast::<Memory>()
                }
                MappingOwner::Memory(_) => return Err(INVALID),
            };
            Ok(Site {
                memory,
                host: true,
                access: if mapping.info.flags & AMDF_MEMORY_MAP_FLAG_READ != 0 {
                    AMDF_MEMORY_ACCESS_READ
                } else {
                    0
                } | if mapping.info.flags & AMDF_MEMORY_MAP_FLAG_WRITE != 0 {
                    AMDF_MEMORY_ACCESS_WRITE
                } else {
                    0
                },
                host_coherent: true,
                cacheability: mapping.info.cacheability,
                release: mapping.info.flush,
                acquire: mapping.info.invalidate,
            })
        }
        AMDF_MEMORY_SITE_KIND_DEVICE => {
            let device = unsafe { site.value.device };
            let memory = unsafe { object(device.memory.cast::<Memory>())? };
            if device.access_ordinal >= memory.info.access_count {
                return Err(RANGE);
            }
            let access = memory
                .access
                .get(device.access_ordinal as usize)
                .ok_or(INTERNAL)?;
            let family = unsafe {
                crate::instance::family(
                    (*access.device).native.endpoint(),
                    device.queue_family_ordinal,
                )?
            };
            Ok(device_site(
                device.memory.cast(),
                memory.info.memory_class,
                memory.info.flags | access.info.flags,
                access.info.access,
                &family,
                memory.info.access_count == 1
                    && memory.cacheability == AMDF_HOST_CACHEABILITY_WRITE_COMBINED,
            ))
        }
        _ => Err(INVALID),
    }
}

pub(crate) unsafe extern "C" fn pair_info(
    producer: *const amdf_memory_site_t,
    consumer: *const amdf_memory_site_t,
    out: *mut amdf_memory_pair_info_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO)?;
            let producer = site(producer)?;
            let consumer = site(consumer)?;
            if (*(*producer.memory).scope).instance != (*(*consumer.memory).scope).instance {
                return Err(PRECONDITION);
            }
            if producer.memory != consumer.memory {
                let producer_id = (*producer.memory).info.physical_backing_id.words;
                let consumer_id = (*consumer.memory).info.physical_backing_id.words;
                // Distinct resources need positive evidence of shared backing.
                // Unknown identity leaves the relation unsupported; known,
                // different identities prove that its precondition is false.
                if producer_id == [0; 2] || consumer_id == [0; 2] {
                    return Err(UNSUPPORTED);
                }
                if producer_id != consumer_id {
                    return Err(PRECONDITION);
                }
            }
            if (*producer.memory).freeing || (*consumer.memory).freeing {
                return Err(PRECONDITION);
            }
            // Shared backing alone does not tell the caller how to release or
            // acquire device caches. Publish a pair only when both recipes are
            // complete; otherwise leave the caller's output unchanged.
            out.publish(describe_pair(&producer, &consumer)?);
            Ok(())
        }
    })
}
