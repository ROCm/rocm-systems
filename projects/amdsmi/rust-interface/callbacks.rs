// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use bindgen::callbacks::EnumVariantValue;

#[derive(Debug)]
pub struct UpperCamelCaseCallbacks;

impl bindgen::callbacks::ParseCallbacks for UpperCamelCaseCallbacks {
    fn item_name(&self, original_item_name: &str) -> Option<String> {
        if original_item_name.starts_with("amdsmi")
            && (original_item_name.ends_with("_t") || original_item_name.contains("_t_"))
        {
            Some(convert_to_upper_camel_case(original_item_name))
        } else {
            match original_item_name {
                "amdsmi_processor_handle"
                | "amdsmi_socket_handle"
                | "amdsmi_node_handle"
                | "processor_type_t"
                | "amd_metrics_table_header_t" => {
                    Some(convert_to_upper_camel_case(original_item_name))
                }
                _ => None,
            }
        }
    }

    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        Some(convert_to_upper_camel_case(original_variant_name))
    }
}

fn convert_to_upper_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c.to_ascii_lowercase());
        }
    }

    result
}
