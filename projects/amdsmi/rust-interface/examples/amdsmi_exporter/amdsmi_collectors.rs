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

use crate::amdsmi_metric::*;
use crate::collectors::collectors_create;
use amdsmi::*;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;

pub struct AmdsmiCollectors {
    pub collectors: HashMap<&'static str, AmdsmiCollector>,
    pub gpu_filter: Option<Vec<String>>,
    pub whitelist: Option<Vec<String>>,
    pub blacklist: Option<Vec<String>>,
}

impl Default for AmdsmiCollectors {
    fn default() -> Self {
        AmdsmiCollectors::new()
    }
}

impl AmdsmiCollectors {
    pub fn new() -> Self {
        let mut instance = AmdsmiCollectors {
            collectors: HashMap::new(),
            gpu_filter: None,
            whitelist: None,
            blacklist: None,
        };
        collectors_create(&mut instance);
        instance
    }

    pub fn set_gpu_filter(&mut self, filters: Option<Vec<String>>) -> &mut Self {
        self.gpu_filter = filters;
        self
    }

    pub fn set_field_filter(
        &mut self,
        whitelist: Option<Vec<String>>,
        blacklist: Option<Vec<String>>,
    ) -> &mut Self {
        self.whitelist = whitelist;
        self.blacklist = blacklist;
        // Re-generate the collectors after setting the field filters
        self.collectors.clear();
        collectors_create(self);
        self
    }

    pub fn add_metric<T, F>(&mut self, collector: AmdsmiGauge<T, F>) -> &mut Self
    where
        T: 'static + Into<i64> + Display + Copy,
        F: 'static + Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync,
    {
        let name = collector.info.name;
        let description = collector.info.description;
        let unit = collector.info.unit;
        let func = collector.func;

        // Apply whitelist and blacklist filters before inserting the collector
        if let Some(ref whitelist) = self.whitelist {
            if !whitelist.contains(&name.to_string()) {
                return self;
            }
        }
        if let Some(ref blacklist) = self.blacklist {
            if blacklist.contains(&name.to_string()) {
                return self;
            }
        }
        self.collectors.insert(
            name,
            Box::new(move |processor_handle| match func(processor_handle) {
                Ok(value) => vec![AmdsmiMetric {
                    info: AmdsmiInfo {
                        name,
                        description,
                        unit: unit.clone(),
                    },
                    value: value.into(),
                    status: AmdsmiStatusT::AmdsmiStatusSuccess,
                }],
                Err(e) => vec![AmdsmiMetric {
                    info: AmdsmiInfo {
                        name,
                        description,
                        unit: unit.clone(),
                    },
                    value: 0,
                    status: e,
                }],
            }),
        );
        self
    }

    pub fn add_metric_group<T, F>(&mut self, collector: &AmdsmiGaugeGroup<T, F>) -> &mut Self
    where
        T: 'static + Debug,
        F: 'static + Fn(AmdsmiProcessorHandle) -> AmdsmiResult<T> + Send + Sync + Clone,
    {
        let name = collector.name;
        let func = collector.func.clone();

        // Apply whitelist and blacklist filters before the Box::new
        let filtered_info_list: Vec<_> = collector
            .info_list
            .iter()
            .filter(|(info, _)| {
                if let Some(ref whitelist) = self.whitelist {
                    if !whitelist.contains(&info.name.to_string()) {
                        return false;
                    }
                }
                if let Some(ref blacklist) = self.blacklist {
                    if blacklist.contains(&info.name.to_string()) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // If the filtered_info_list is empty, return early
        if filtered_info_list.is_empty() {
            return self;
        }

        self.collectors.insert(
            name,
            Box::new(move |processor_handle| {
                let mut results = Vec::new();

                match func(processor_handle) {
                    Ok(value) => {
                        for (info, accessor) in &filtered_info_list {
                            results.push(AmdsmiMetric {
                                info: AmdsmiInfo {
                                    name: info.name,
                                    description: info.description,
                                    unit: info.unit.clone(),
                                },
                                value: accessor(&value),
                                status: AmdsmiStatusT::AmdsmiStatusSuccess,
                            });
                        }
                    }
                    Err(e) => {
                        for (info, _) in &filtered_info_list {
                            results.push(AmdsmiMetric {
                                info: AmdsmiInfo {
                                    name: info.name,
                                    description: info.description,
                                    unit: info.unit.clone(),
                                },
                                value: 0,
                                status: e,
                            });
                        }
                    }
                }
                results
            }),
        );
        self
    }

    pub fn print_supported(&self) {
        // Initialize the AMD SMI library
        if let Err(e) = amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus) {
            eprintln!("Failed to initialize AMD SMI: {}", e);
            return;
        }

        // Get socket handles
        let socket_handles = match amdsmi_get_socket_handles() {
            Ok(handles) => handles,
            Err(e) => {
                eprintln!("Failed to get socket handles: {}", e);
                return;
            }
        };

        for (socket_index, socket_handle) in socket_handles.iter().enumerate() {
            let socket_id = socket_index.to_string();

            // Get processor handles for each socket handle
            let processor_handles = match amdsmi_get_processor_handles(*socket_handle) {
                Ok(handles) => handles,
                Err(e) => {
                    eprintln!("Failed to get processor handles: {}", e);
                    continue;
                }
            };

            for (processor_index, processor_handle) in processor_handles.iter().enumerate() {
                let processor_id = processor_index.to_string();

                let bdf = match amdsmi_get_gpu_device_bdf(*processor_handle) {
                    Ok(bdf) => bdf.to_string(),
                    Err(e) => {
                        eprintln!("Failed to get GPU device BDF: {}", e);
                        continue;
                    }
                };

                println!("========================================");
                println!("GPU Information:");
                println!("  Socket ID:      {}", socket_id);
                println!("  Processor ID:   {}", processor_id);
                println!("  BDF:            {}", bdf);
                println!("----------------------------------------");
                println!("Supported Metrics:");
                println!(
                    "{:<30} | {:<10} | {:<10} | {:<50}",
                    "Metric ID", "Value", "Unit", "Help"
                );
                println!("{:-<30}-+-{:-<10}-+-{:-<10}-+-{:-<50}", "", "", "", "");

                // Collect metrics
                for (_, collector) in &self.collectors {
                    let results = collector(*processor_handle);
                    for metric in results {
                        let value_str = if metric.status == AmdsmiStatusT::AmdsmiStatusSuccess {
                            metric.value.to_string()
                        } else {
                            "NA".to_string()
                        };

                        println!(
                            "{:<30} | {:<10} | {:<10} | {:<50}",
                            metric.info.name,
                            value_str,
                            metric.info.unit.as_str(),
                            metric.info.description
                        );
                    }
                }
                println!("========================================\n");
            }
        }

        // Shutdown the AMD SMI library
        if let Err(e) = amdsmi_shut_down() {
            eprintln!("Failed to shutdown AMD SMI: {}", e);
        }
    }

    pub fn run_collect(&mut self) -> Registry {
        // Initialize the AMD SMI library
        if let Err(e) = amdsmi_init(AmdsmiInitFlagsT::AmdsmiInitAmdGpus) {
            eprintln!("Failed to initialize AMD SMI: {}", e);
            return Registry::default();
        }

        let mut registry = Registry::default();

        // Get socket handles
        let socket_handles = match amdsmi_get_socket_handles() {
            Ok(handles) => handles,
            Err(e) => {
                eprintln!("Failed to get socket handles: {}", e);
                return registry;
            }
        };

        for (socket_index, socket_handle) in socket_handles.iter().enumerate() {
            let socket_id = socket_index.to_string();

            // Get processor handles for each socket handle
            let processor_handles = match amdsmi_get_processor_handles(*socket_handle) {
                Ok(handles) => handles,
                Err(e) => {
                    eprintln!("Failed to get processor handles: {}", e);
                    continue;
                }
            };

            for (processor_index, processor_handle) in processor_handles.iter().enumerate() {
                let processor_id = processor_index.to_string();

                let bdf = match amdsmi_get_gpu_device_bdf(*processor_handle) {
                    Ok(bdf) => bdf.to_string(),
                    Err(e) => {
                        eprintln!("Failed to get GPU device BDF: {}", e);
                        continue;
                    }
                };

                let labels = AmdsmiLabels {
                    socket_id: socket_id.clone(),
                    processor_id: processor_id.clone(),
                    bdf,
                };

                // Only for GPUs which match the provided gpu filter argument
                if let Some(ref filters) = self.gpu_filter {
                    if !filters.iter().any(|filter| {
                        filter == &labels.bdf
                            || filter == &format!("{}.{}", socket_id, processor_id)
                    }) {
                        continue;
                    }
                }

                // Collect metrics
                for (_, collector) in &self.collectors {
                    let results = collector(*processor_handle);
                    for metric in results {
                        if metric.status == AmdsmiStatusT::AmdsmiStatusSuccess {
                            let metric_family =
                                Family::<AmdsmiLabels, Gauge>::new_with_constructor(Gauge::default);
                            if metric.info.unit.as_str().is_empty() {
                                registry.register(
                                    metric.info.name,
                                    metric.info.description,
                                    metric_family.clone(),
                                );
                            } else {
                                registry.register_with_unit(
                                    metric.info.name,
                                    metric.info.description,
                                    metric.info.unit.clone(),
                                    metric_family.clone(),
                                );
                            }
                            metric_family.get_or_create(&labels).set(metric.value);
                        } else {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "Failed to collect metric {}: {} for socket {} processor {}",
                                metric.info.name,
                                metric.status,
                                labels.socket_id,
                                labels.processor_id
                            );
                        }
                    }
                }
            }
        }

        // Shutdown the AMD SMI library
        if let Err(e) = amdsmi_shut_down() {
            eprintln!("Failed to shutdown AMD SMI: {}", e);
        }

        registry
    }
}

/// A macro to add a metric collector.
///
/// This macro creates an `AmdsmiGauge` instance and adds it to the collectors.
///
/// # Parameters
/// - `$collectors`: The collectors to add the metric to.
/// - `$name`: The name of the metric.
/// - `$description`: The description of the metric.
/// - `$unit`: The unit of the metric.
/// - `$func`: The function to collect the metric.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// # use amdsmi_collectors::*;
///
/// # fn main() {
/// let mut collectors = AmdsmiCollectors::default();
/// add_metric!(
///     collectors,
///     "AMDSMI_FI_TEMPERATURE_EDGE",
///     "Currnt edge temperature",
///     Unit::Celsius,
///     amdsmi_func!(amdsmi_get_temp_metric, AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeEdge, AmdsmiTemperatureMetricT::AmdsmiTempCurrent)
/// );
/// # }
/// ```
macro_rules! add_metric {
    ($collectors:expr, $name:expr, $description:expr, $unit:expr, $func:expr) => {
        let collector = AmdsmiGauge::new($name, $description, $unit, $func);
        $collectors.add_metric(collector);
    };
}

/// A macro to add a single metric from a struct to the collectors.
///
/// This macro creates an `AmdsmiGauge` instance from a field of struct and adds it to the collectors.
///
/// # Parameters
/// - `$collectors`: The collectors to add the metric to.
/// - `$name`: The name of the metric.
/// - `$description`: The description of the metric.
/// - `$unit`: The unit of the metric.
/// - `$func`: The function to collect the metric.
/// - `$field`: The struct field from which the metric is collected.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// # use amdsmi_collectors::*;
///
/// # fn main() {
/// let mut collectors = AmdsmiCollectors::default();
/// add_metric_field!(
///     collectors,
///     "AMDSMI_FI_CURRENT_SOCKET_POWER",
///     "Current Socket Power",
///     Unit::Watts,
///     amdsmi_get_power_info,
///     current_socket_power
/// );
/// # }
/// ```
#[allow(unused_macros)]
macro_rules! add_metric_field {
    ($collectors:expr, $name:expr, $description:expr, $unit:expr, $func:expr, $field:ident) => {
        let collector = AmdsmiGauge::new($name, $description, $unit, |handle| {
            $func(handle).map(|info| info.$field.into())
        });
        $collectors.add_metric(collector);
    };
}
/// A macro to add a group of metrics within a single struct.
///
/// This macro creates an `AmdsmiGaugeGroup` instance and adds it to the collectors.
///
/// # Parameters
/// - `$collectors`: The collectors to add the metric group to.
/// - `$group_name`: The name of the metric group.
/// - `$func`: The function to collect the metrics.
/// - `$t`: The type of the struct containing the metrics.
/// - `$name`: The name of each metric.
/// - `$description`: The description of each metric.
/// - `$unit`: The unit of each metric.
/// - `$field`: The struct field from which each metric is collected.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// # use amdsmi_collectors::*;
///
/// # fn main() {
/// let mut collectors = AmdsmiCollectors::default();
/// add_metric_group!(
///     collectors,
///     "AmdsmiPowerInfoMetrics",
///     amdsmi_get_power_info,
///     AmdsmiPowerInfoT,
///     ("AMDSMI_FI_CURRENT_SOCKET_POWER", "Current Socket Power", Unit::Watts, current_socket_power),
///     ("AMDSMI_FI_AVERAGE_SOCKET_POWER", "Average Socket Power", Unit::Watts, average_socket_power),
///     ("AMDSMI_FI_GFX_VOLTAGE", "GFX Voltage", Unit::Volts, gfx_voltage),
///     ("AMDSMI_FI_SOC_VOLTAGE", "SOC Voltage", Unit::Volts, soc_voltage),
///     ("AMDSMI_FI_MEM_VOLTAGE", "Memory Voltage", Unit::Volts, mem_voltage),
///     ("AMDSMI_FI_POWER_LIMIT", "Power Limit", Unit::Watts, power_limit)
/// );
/// # }
/// ```
#[allow(unused_macros)]
macro_rules! add_metric_group {
    ($collectors:expr, $group_name:expr, $func:expr, $t:ty, $(($name:expr, $description:expr, $unit:expr, $field:ident)),+ $(,)?) => {
        let info_list = vec![
            $(
                (
                    $name,
                    $description,
                    $unit,
                    Arc::new(|info: &$t| info.$field as i64) as Arc<dyn Fn(&$t) -> i64 + Send + Sync>,
                ),
            )+
        ];
        let mut collector = AmdsmiGaugeGroup::new($group_name, info_list, $func);
        $collectors.add_metric_group(&mut collector);
    };
}
/// A macro to create a closure for an AMD SMI function with dynamic arguments.
///
/// This macro generates a closure that calls the specified function with the provided arguments.
///
/// # Parameters
/// - `$func`: The function to create the closure for.
/// - `$arg`: The arguments to pass to the function.
///
/// # Example
///
/// ```rust
/// # use amdsmi::*;
/// # use amdsmi_collectors::*;
///
/// # fn main() {
/// let func = amdsmi_func!(amdsmi_get_temp_metric, AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeEdge, AmdsmiTemperatureMetricT::AmdsmiTempCurrent);
/// # }
/// ``
#[allow(unused_macros)]
macro_rules! amdsmi_func {
    ($func:expr, $($arg:expr),*) => {
        move |handle| $func(handle, $($arg),*)
    };
}
