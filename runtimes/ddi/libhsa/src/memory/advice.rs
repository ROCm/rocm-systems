// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA SVM policy translation without native driver flags or identifiers.

use super::*;
use rocddi::memory::advice::{
    AdviceAccess, AdviceChange, AdviceProperty, AdviceQuery, AdviceState, AdviceValue,
};

/// Maps scalar HSA policy tags to semantic properties.
fn property(attribute: u64) -> Option<AdviceProperty> {
    match attribute {
        AMD_SVM_ATTRIB_GLOBAL_FLAG => Some(AdviceProperty::Coherent),
        AMD_SVM_ATTRIB_READ_ONLY => Some(AdviceProperty::DeviceReadOnly),
        AMD_SVM_ATTRIB_HIVE_LOCAL => Some(AdviceProperty::HiveLocal),
        AMD_SVM_ATTRIB_READ_MOSTLY => Some(AdviceProperty::DeviceReadMostly),
        AMD_SVM_ATTRIB_GPU_EXEC => Some(AdviceProperty::DeviceExecutable),
        _ => None,
    }
}

/// Reserves frontend output storage before any publication.
fn storage<T>(count: usize) -> Result<Vec<T>, Status> {
    rocddi::memory::storage::try_vec_with_capacity(count).map_err(|_| OUT_OF_RESOURCES)
}

/// Qualifies a GPU agent with the selected provider.
fn device_target(
    runtime: &HsaState,
    service: &MemoryAdvice,
    agent: HsaAgent,
) -> Result<rocddi::memory::advice::AdviceTarget, Status> {
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    service
        .target(&runtime.gpus[index].device)
        .map_err(map_error)
}

/// Applies HSA policy after validating the complete frontend batch.
pub(crate) fn set(
    runtime: &HsaState,
    address: u64,
    size: u64,
    pairs: &[HsaAmdSvmAttributePair],
) -> Result<(), Status> {
    let selected = pairs
        .iter()
        .filter(|pair| {
            matches!(
                pair.attribute,
                AMD_SVM_ATTRIB_PREFERRED_LOCATION
                    | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                    | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE
                    | AMD_SVM_ATTRIB_AGENT_NO_ACCESS
            )
        })
        .map(|pair| HsaAgent { handle: pair.value })
        .find(|agent| runtime.gpu_index(*agent).is_some());
    let service = svm_service(runtime, selected)?;
    let mut changes = storage(pairs.len() + 1)?;
    let mut agents = storage(pairs.len())?;
    let mut scalar_attributes = 0_u16;
    let mut host_access = None;
    for pair in pairs {
        if property(pair.attribute).is_some()
            || matches!(
                pair.attribute,
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY | AMD_SVM_ATTRIB_PREFERRED_LOCATION
            )
        {
            let bit = 1_u16 << pair.attribute;
            if scalar_attributes & bit != 0 {
                return Err(INCOMPATIBLE_ARGUMENTS);
            }
            scalar_attributes |= bit;
        }
        if let Some(property) = property(pair.attribute) {
            let enabled = if pair.attribute == AMD_SVM_ATTRIB_GLOBAL_FLAG {
                match pair.value {
                    AMD_SVM_GLOBAL_FLAG_FINE_GRAINED => true,
                    AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED => false,
                    _ => return Err(INVALID_ARGUMENT),
                }
            } else {
                pair.value != 0
            };
            changes.push(AdviceChange::Property(property, enabled));
            continue;
        }
        match pair.attribute {
            AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                let page = u128::from(rocddi::memory::host::host_page_size().map_err(map_error)?);
                let exponent = u32::try_from(pair.value.min(SVM_MAX_MIGRATION_GRANULARITY))
                    .map_err(|_| INVALID_ARGUMENT)?;
                changes.push(AdviceChange::MigrationGranularity(
                    page.checked_shl(exponent).ok_or(INVALID_ARGUMENT)?,
                ));
            }
            AMD_SVM_ATTRIB_PREFERRED_LOCATION => changes.push(AdviceChange::PreferredLocation(
                svm_location_from_agent(runtime, &service, HsaAgent { handle: pair.value }, true)?,
            )),
            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
            | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE
            | AMD_SVM_ATTRIB_AGENT_NO_ACCESS => {
                let agent = HsaAgent { handle: pair.value };
                if !runtime.is_agent(agent) {
                    return Err(INVALID_AGENT);
                }
                if agents.contains(&agent.handle) {
                    return Err(INCOMPATIBLE_ARGUMENTS);
                }
                agents.push(agent.handle);
                if runtime.is_cpu_agent(agent) {
                    // All CPU aliases address one host property; any permitted
                    // host access wins over another CPU alias denying access.
                    host_access = Some(
                        host_access.unwrap_or(false)
                            || pair.attribute != AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
                    );
                } else {
                    changes.push(AdviceChange::Access(
                        device_target(runtime, &service, agent)?,
                        match pair.attribute {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE => AdviceAccess::Accessible,
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE => {
                                AdviceAccess::AccessibleInPlace
                            }
                            _ => AdviceAccess::NoAccess,
                        },
                    ));
                }
            }
            _ => return Err(INVALID_ARGUMENT),
        }
    }
    if let Some(enabled) = host_access {
        changes.push(AdviceChange::Property(AdviceProperty::HostAccess, enabled));
    }
    service.set(address, size, &changes).map_err(map_error)
}

/// Stages every HSA result before publishing caller-owned output fields.
///
/// # Safety
/// `pairs` addresses `count` writable ABI pairs. Every tag and the value of
/// each `ACCESS_QUERY` are initialized; other input values may be uninitialized.
pub(crate) unsafe fn get(
    runtime: &HsaState,
    address: u64,
    size: u64,
    pairs: *mut HsaAmdSvmAttributePair,
    count: usize,
) -> Result<(), Status> {
    let mut inputs = storage(count)?;
    for index in 0..count {
        // SAFETY: The caller supplies the complete array. Read only initialized
        // fields, without forming a reference to an uninitialized ABI pair.
        let pair = unsafe { pairs.add(index) };
        let attribute = unsafe { (&raw const (*pair).attribute).read() };
        let agent = if attribute == AMD_SVM_ATTRIB_ACCESS_QUERY {
            Some(HsaAgent {
                handle: unsafe { (&raw const (*pair).value).read() },
            })
        } else {
            None
        };
        inputs.push((attribute, agent));
    }
    let selected = inputs
        .iter()
        .filter_map(|(_, agent)| *agent)
        .find(|agent| runtime.gpu_index(*agent).is_some());
    let service = svm_service(runtime, selected)?;
    let mut queries = storage(count)?;
    for &(attribute, agent) in &inputs {
        queries.push(if let Some(property) = property(attribute) {
            AdviceQuery::Property(property)
        } else {
            match attribute {
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => AdviceQuery::PreferredLocation,
                AMD_SVM_ATTRIB_PREFETCH_LOCATION => AdviceQuery::PrefetchLocation,
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => AdviceQuery::MigrationGranularity,
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    let agent = agent.ok_or(INVALID_AGENT)?;
                    if runtime.is_cpu_agent(agent) {
                        AdviceQuery::Property(AdviceProperty::HostAccess)
                    } else {
                        AdviceQuery::Access(device_target(runtime, &service, agent)?)
                    }
                }
                _ => return Err(INVALID_ARGUMENT),
            }
        });
    }
    let values = service.query(address, size, &queries).map_err(map_error)?;
    let mut outputs = storage(count)?;
    for ((attribute, _), value) in inputs.into_iter().zip(values) {
        outputs.push(match (attribute, value) {
            (AMD_SVM_ATTRIB_GLOBAL_FLAG, AdviceValue::Property(state)) => match state {
                AdviceState::Enabled => AMD_SVM_GLOBAL_FLAG_FINE_GRAINED,
                AdviceState::Disabled => AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED,
                AdviceState::Mixed => AMD_SVM_GLOBAL_FLAG_INDETERMINATE,
            },
            (AMD_SVM_ATTRIB_ACCESS_QUERY, AdviceValue::Property(state)) => {
                if state == AdviceState::Enabled {
                    AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                } else {
                    AMD_SVM_ATTRIB_AGENT_NO_ACCESS
                }
            }
            (AMD_SVM_ATTRIB_ACCESS_QUERY, AdviceValue::Access(access)) => match access {
                AdviceAccess::Accessible => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE,
                AdviceAccess::AccessibleInPlace => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE,
                AdviceAccess::NoAccess => AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
            },
            (_, AdviceValue::Property(state)) => u64::from(state == AdviceState::Enabled),
            (_, AdviceValue::Location(location)) => {
                svm_agent_from_location(runtime, location)?.handle
            }
            (_, AdviceValue::MigrationGranularity(bytes)) => {
                let page = u128::from(rocddi::memory::host::host_page_size().map_err(map_error)?);
                if bytes < page || bytes % page != 0 {
                    return Err(ERROR);
                }
                u64::from((bytes / page).ilog2())
            }
            _ => return Err(ERROR),
        });
    }
    for (index, output) in outputs.into_iter().enumerate() {
        // SAFETY: All validation and translation succeeded. ACCESS_QUERY writes
        // only the tag, preserving its agent input; other queries write value.
        let pair = unsafe { pairs.add(index) };
        if unsafe { (&raw const (*pair).attribute).read() } == AMD_SVM_ATTRIB_ACCESS_QUERY {
            unsafe { (&raw mut (*pair).attribute).write(output) };
        } else {
            unsafe { (&raw mut (*pair).value).write(output) };
        }
    }
    Ok(())
}
