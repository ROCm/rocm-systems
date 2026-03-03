// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

use amdsmi::{AmdsmiProcessorHandle, AmdsmiResult, AmdsmiStatusT};
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::registry::Unit;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Hash, PartialEq, Eq, EncodeLabelSet, Debug)]
pub struct AmdsmiLabels {
    pub socket_id: String,
    pub processor_id: String,
    pub bdf: String,
}
#[derive(Clone)]
pub struct AmdsmiInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub unit: Unit,
}
#[derive(Clone)]
pub struct AmdsmiMetric {
    pub info: AmdsmiInfo,
    pub value: i64,
    pub status: AmdsmiStatusT,
}

pub type AmdsmiCollector = Box<dyn Fn(AmdsmiProcessorHandle) -> Vec<AmdsmiMetric> + Send + Sync>;

pub struct AmdsmiGauge<T, F>
where
    F: Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync,
{
    pub info: AmdsmiInfo,
    pub func: F,
}

impl<T, F> AmdsmiGauge<T, F>
where
    F: Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync,
{
    pub fn new(name: &'static str, description: &'static str, unit: Unit, func: F) -> Self {
        Self {
            info: AmdsmiInfo {
                name,
                description,
                unit,
            },
            func,
        }
    }
}

pub struct AmdsmiGaugeGroup<T, F>
where
    F: Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync + Clone,
{
    pub name: &'static str, /* name of the group */
    pub info_list: Vec<(
        AmdsmiInfo,
        Arc<dyn Fn(&T) -> i64 + Send + Sync>, /* field accessor */
    )>,
    pub func: F,
}

impl<T, F> AmdsmiGaugeGroup<T, F>
where
    F: Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync + Clone,
{
    pub fn new(
        name: &'static str, /* name of the group */
        info_list: Vec<(
            &'static str,                         /* name */
            &'static str,                         /* description */
            Unit,                                 /* unit */
            Arc<dyn Fn(&T) -> i64 + Send + Sync>, /* field accessor */
        )>,
        func: F,
    ) -> Self {
        Self {
            name,
            info_list: info_list
                .into_iter()
                .map(|(name, description, unit, accessor)| {
                    (
                        AmdsmiInfo {
                            name,
                            description,
                            unit,
                        },
                        accessor,
                    )
                })
                .collect(),
            func,
        }
    }
}

/// A macro to define a unit as `Unit::Other`.
///
/// This macro takes a unit string as an argument and generates the appropriate `Unit` value.
///
/// # Parameters
/// - `$unit_str`: The unit string to define the unit.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// # use amdsmi_metric::*;
///
/// let unit = metric_unit_other!("W");
/// assert_eq!(unit, Unit::Other("W".to_string()));
/// ```
macro_rules! metric_unit_other {
    ($unit_str:expr) => {
        Unit::Other($unit_str.to_string())
    };
}
