// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! KFD encoding of semantic process-range policy and migration requests.

use super::{LinuxKfdDriver, error, native_error, uapi};
use crate::driver::{Driver, MemoryAdviceOperations};
use crate::memory::advice::{
    AdviceAccess, AdviceCapabilities, AdviceChange, AdviceLocation, AdviceProperty, AdviceQuery,
    AdviceState, AdviceTarget, AdviceValue, PrefetchCompletion,
};
use crate::memory::storage::try_vec_with_capacity;
use crate::{Error, ErrorKind};

fn flag(property: AdviceProperty) -> u32 {
    match property {
        AdviceProperty::HostAccess => uapi::SVM_FLAG_HOST_ACCESS,
        AdviceProperty::Coherent => uapi::SVM_FLAG_COHERENT,
        AdviceProperty::HiveLocal => uapi::SVM_FLAG_HIVE_LOCAL,
        AdviceProperty::DeviceReadOnly => uapi::SVM_FLAG_GPU_READ_ONLY,
        AdviceProperty::DeviceExecutable => uapi::SVM_FLAG_GPU_EXECUTE,
        AdviceProperty::DeviceReadMostly => uapi::SVM_FLAG_GPU_READ_MOSTLY,
    }
}

fn property_state(set: u32, clear: u32, bit: u32) -> Result<AdviceState, Error> {
    match (set & bit != 0, clear & bit != 0) {
        (true, false) => Ok(AdviceState::Enabled),
        (false, true) => Ok(AdviceState::Disabled),
        (false, false) => Ok(AdviceState::Mixed),
        (true, true) => Err(error(
            ErrorKind::DriverContract,
            "contradictory KFD SVM flag reduction",
        )),
    }
}

fn attribute(attribute_type: u32, value: u32) -> uapi::SvmAttribute {
    uapi::SvmAttribute {
        attribute_type,
        value,
    }
}

fn checked_value(value: uapi::SvmAttribute, expected: u32) -> Result<u32, Error> {
    if value.attribute_type != expected {
        return Err(error(
            ErrorKind::DriverContract,
            "KFD changed an SVM query type",
        ));
    }
    Ok(value.value)
}

fn access_type(access: AdviceAccess) -> u32 {
    match access {
        AdviceAccess::Accessible => uapi::SVM_ATTR_ACCESS,
        AdviceAccess::AccessibleInPlace => uapi::SVM_ATTR_ACCESS_IN_PLACE,
        AdviceAccess::NoAccess => uapi::SVM_ATTR_NO_ACCESS,
    }
}

impl LinuxKfdDriver {
    fn decode_advice_queries(
        &self,
        queries: &[AdviceQuery],
        native: &[uapi::SvmAttribute],
    ) -> Result<Vec<AdviceValue>, Error> {
        let mut values = try_vec_with_capacity(queries.len())?;
        let flags = queries
            .iter()
            .any(|query| matches!(query, AdviceQuery::Property(_)));
        let (set, clear) = if flags {
            (
                checked_value(native[0], uapi::SVM_ATTR_SET_FLAGS)?,
                checked_value(native[1], uapi::SVM_ATTR_CLEAR_FLAGS)?,
            )
        } else {
            (0, 0)
        };
        let mut index = if flags { 2 } else { 0 };
        for query in queries {
            if let AdviceQuery::Property(property) = *query {
                values.push(AdviceValue::Property(property_state(
                    set,
                    clear,
                    flag(property),
                )?));
                continue;
            }
            let returned = native[index];
            index += 1;
            values.push(match *query {
                AdviceQuery::PreferredLocation => {
                    AdviceValue::Location(self.advice_location_from_native(checked_value(
                        returned,
                        uapi::SVM_ATTR_PREFERRED_LOCATION,
                    )?)?)
                }
                AdviceQuery::PrefetchLocation => {
                    AdviceValue::Location(self.advice_location_from_native(checked_value(
                        returned,
                        uapi::SVM_ATTR_PREFETCH_LOCATION,
                    )?)?)
                }
                AdviceQuery::Access(target) => {
                    if returned.value != self.advice_gpu_id(target)? {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM access query target",
                        ));
                    }
                    AdviceValue::Access(match returned.attribute_type {
                        uapi::SVM_ATTR_ACCESS => AdviceAccess::Accessible,
                        uapi::SVM_ATTR_ACCESS_IN_PLACE => AdviceAccess::AccessibleInPlace,
                        uapi::SVM_ATTR_NO_ACCESS => AdviceAccess::NoAccess,
                        _ => {
                            return Err(error(
                                ErrorKind::DriverContract,
                                "invalid KFD SVM access response",
                            ));
                        }
                    })
                }
                AdviceQuery::MigrationGranularity => {
                    let exponent = checked_value(returned, uapi::SVM_ATTR_GRANULARITY)?;
                    if exponent > 63 {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD SVM granularity exceeds its supported limit",
                        ));
                    }
                    let bytes = u128::from(crate::memory::host_page_size()?)
                        .checked_shl(exponent)
                        .ok_or_else(|| {
                            error(ErrorKind::DriverContract, "KFD SVM granularity overflows")
                        })?;
                    values.push(AdviceValue::MigrationGranularity(bytes));
                    continue;
                }
                AdviceQuery::Property(_) => {
                    return Err(error(ErrorKind::Internal, "property query was not reduced"));
                }
            });
        }
        Ok(values)
    }

    fn advice_gpu_id(&self, target: AdviceTarget) -> Result<u32, Error> {
        if target.provider != self.driver_instance() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "foreign KFD advice target",
            ));
        }
        self.with_context_resources(|resources| resources.bindings.svm_gpu_id(target.endpoint))
    }

    fn advice_location_to_native(&self, location: AdviceLocation) -> Result<u32, Error> {
        match location {
            AdviceLocation::System => Ok(uapi::SVM_LOCATION_SYSTEM),
            AdviceLocation::UnspecifiedOrMixed => Ok(uapi::SVM_LOCATION_UNDEFINED),
            AdviceLocation::Device(target) => self.advice_gpu_id(target),
        }
    }

    fn advice_location_from_native(&self, value: u32) -> Result<AdviceLocation, Error> {
        match value {
            uapi::SVM_LOCATION_SYSTEM => Ok(AdviceLocation::System),
            uapi::SVM_LOCATION_UNDEFINED => Ok(AdviceLocation::UnspecifiedOrMixed),
            gpu_id => self
                .with_context_resources(|resources| resources.bindings.svm_identity(gpu_id))
                .map(|endpoint| {
                    AdviceLocation::Device(AdviceTarget {
                        provider: self.driver_instance(),
                        endpoint,
                    })
                }),
        }
    }

    fn advice_native(
        &self,
        address: u64,
        size: u64,
        operation: u32,
        native: &mut [uapi::SvmAttribute],
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if native.is_empty() {
            return Ok(());
        }
        self.kfd()?
            .svm_attributes(address, size, operation, native)
            .map_err(|source| native_error("AMDKFD_IOC_SVM", source))
    }
}

impl MemoryAdviceOperations for LinuxKfdDriver {
    fn advice_capabilities(&self) -> Result<AdviceCapabilities, Error> {
        self.ensure_open()?;
        let page = crate::memory::host_page_size()?;
        Ok(AdviceCapabilities {
            prefetch_completion: PrefetchCompletion::Attempt,
            range_alignment: page,
            minimum_migration_granularity: u128::from(page),
            maximum_migration_granularity: u128::from(page)
                .checked_shl(63)
                .ok_or_else(|| error(ErrorKind::DriverContract, "KFD SVM page size overflows"))?,
        })
    }

    fn advice_set(&self, address: u64, size: u64, changes: &[AdviceChange]) -> Result<(), Error> {
        let page = self.advice_capabilities()?.minimum_migration_granularity;
        let mut native = try_vec_with_capacity(changes.len() + 2)?;
        let mut set = 0;
        let mut clear = 0;
        for change in changes {
            match *change {
                AdviceChange::Property(property, enabled) => {
                    let bit = flag(property);
                    if enabled {
                        set |= bit;
                        clear &= !bit;
                    } else {
                        clear |= bit;
                        set &= !bit;
                    }
                }
                AdviceChange::PreferredLocation(location) => native.push(attribute(
                    uapi::SVM_ATTR_PREFERRED_LOCATION,
                    self.advice_location_to_native(location)?,
                )),
                AdviceChange::Access(target, access) => {
                    native.push(attribute(access_type(access), self.advice_gpu_id(target)?));
                }
                AdviceChange::MigrationGranularity(bytes) => {
                    if bytes < page || !bytes.is_power_of_two() || bytes / page > (1_u128 << 63) {
                        return Err(error(
                            ErrorKind::InvalidArgument,
                            "invalid KFD migration granularity",
                        ));
                    }
                    native.push(attribute(
                        uapi::SVM_ATTR_GRANULARITY,
                        (bytes / page).ilog2(),
                    ));
                }
            }
        }
        if clear != 0 {
            native.push(attribute(uapi::SVM_ATTR_CLEAR_FLAGS, clear));
        }
        if set != 0 {
            native.push(attribute(uapi::SVM_ATTR_SET_FLAGS, set));
        }
        self.advice_native(address, size, uapi::SVM_OP_SET_ATTR, &mut native)
    }

    fn advice_query(
        &self,
        address: u64,
        size: u64,
        queries: &[AdviceQuery],
    ) -> Result<Vec<AdviceValue>, Error> {
        let mut native = try_vec_with_capacity(queries.len() + 2)?;
        let flags = queries
            .iter()
            .any(|query| matches!(query, AdviceQuery::Property(_)));
        if flags {
            native.push(attribute(uapi::SVM_ATTR_SET_FLAGS, 0));
            native.push(attribute(uapi::SVM_ATTR_CLEAR_FLAGS, 0));
        }
        for query in queries {
            let encoded = match *query {
                AdviceQuery::Property(_) => continue,
                AdviceQuery::PreferredLocation => attribute(
                    uapi::SVM_ATTR_PREFERRED_LOCATION,
                    uapi::SVM_LOCATION_UNDEFINED,
                ),
                AdviceQuery::PrefetchLocation => attribute(
                    uapi::SVM_ATTR_PREFETCH_LOCATION,
                    uapi::SVM_LOCATION_UNDEFINED,
                ),
                AdviceQuery::Access(target) => {
                    attribute(uapi::SVM_ATTR_ACCESS, self.advice_gpu_id(target)?)
                }
                AdviceQuery::MigrationGranularity => attribute(uapi::SVM_ATTR_GRANULARITY, 0),
            };
            native.push(encoded);
        }
        self.advice_native(address, size, uapi::SVM_OP_GET_ATTR, &mut native)?;
        self.decode_advice_queries(queries, &native)
    }

    fn advice_prefetch(
        &self,
        address: u64,
        size: u64,
        location: AdviceLocation,
        completion: PrefetchCompletion,
    ) -> Result<(), Error> {
        // KFD's synchronous ioctl waits for submitted copy fences, but its
        // migration trigger can suppress placement failures and use fallback
        // locations. GET PREFETCH_LOCATION reports intent, not residency.
        if completion == PrefetchCompletion::Migration {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD does not prove completed placement",
            ));
        }
        let value = self.advice_location_to_native(location)?;
        self.advice_native(
            address,
            size,
            uapi::SVM_OP_SET_ATTR,
            &mut [attribute(uapi::SVM_ATTR_PREFETCH_LOCATION, value)],
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn native_queries_stage_interleaved_values_and_reject_late_malformed_data() {
        let driver = LinuxKfdDriver::with_context_lifetime(
            crate::session::DriverContextLifetime::Session,
            crate::os::ProcessIdentity::capture().unwrap(),
        );
        let queries = [
            AdviceQuery::Property(AdviceProperty::Coherent),
            AdviceQuery::PreferredLocation,
            AdviceQuery::Property(AdviceProperty::DeviceReadOnly),
            AdviceQuery::MigrationGranularity,
        ];
        let mut native = [
            attribute(uapi::SVM_ATTR_SET_FLAGS, uapi::SVM_FLAG_COHERENT),
            attribute(uapi::SVM_ATTR_CLEAR_FLAGS, uapi::SVM_FLAG_GPU_READ_ONLY),
            attribute(
                uapi::SVM_ATTR_PREFERRED_LOCATION,
                uapi::SVM_LOCATION_UNDEFINED,
            ),
            attribute(uapi::SVM_ATTR_GRANULARITY, 2),
        ];
        assert_eq!(
            driver.decode_advice_queries(&queries, &native).unwrap(),
            [
                AdviceValue::Property(AdviceState::Enabled),
                AdviceValue::Location(AdviceLocation::UnspecifiedOrMixed),
                AdviceValue::Property(AdviceState::Disabled),
                AdviceValue::MigrationGranularity(
                    u128::from(crate::memory::host_page_size().unwrap()) * 4
                )
            ]
        );
        native[3].attribute_type = uapi::SVM_ATTR_ACCESS;
        assert_eq!(
            driver
                .decode_advice_queries(&queries, &native)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );
        native[3] = attribute(uapi::SVM_ATTR_GRANULARITY, 64);
        assert_eq!(
            driver
                .decode_advice_queries(&queries, &native)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );
    }

    #[test]
    fn native_granularity_exceeds_hsa_set_policy_without_losing_byte_units() {
        let driver = LinuxKfdDriver::with_context_lifetime(
            crate::session::DriverContextLifetime::Session,
            crate::os::ProcessIdentity::capture().unwrap(),
        );
        let page = u128::from(crate::memory::host_page_size().unwrap());
        for exponent in [19, 27, 63] {
            assert_eq!(
                driver
                    .decode_advice_queries(
                        &[AdviceQuery::MigrationGranularity],
                        &[attribute(uapi::SVM_ATTR_GRANULARITY, exponent)]
                    )
                    .unwrap(),
                [AdviceValue::MigrationGranularity(page << exponent)]
            );
        }
        // This rejection requires neither an active GPU nor a native ioctl.
        assert_eq!(
            driver
                .advice_prefetch(
                    0x10000,
                    4096,
                    AdviceLocation::System,
                    PrefetchCompletion::Migration
                )
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn semantic_properties_follow_uapi_bits_and_range_reductions() {
        for (property, bit) in [
            (AdviceProperty::HostAccess, 0x01),
            (AdviceProperty::Coherent, 0x02),
            (AdviceProperty::HiveLocal, 0x04),
            (AdviceProperty::DeviceReadOnly, 0x08),
            (AdviceProperty::DeviceExecutable, 0x10),
            (AdviceProperty::DeviceReadMostly, 0x20),
        ] {
            assert_eq!(flag(property), bit);
            assert_eq!(property_state(bit, 0, bit).unwrap(), AdviceState::Enabled);
            assert_eq!(property_state(0, bit, bit).unwrap(), AdviceState::Disabled);
            assert_eq!(property_state(0, 0, bit).unwrap(), AdviceState::Mixed);
            assert_eq!(
                property_state(bit, bit, bit).unwrap_err().kind(),
                ErrorKind::DriverContract
            );
        }
    }
}
