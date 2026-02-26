#pragma once
#include <string>

bool is_sriov_virtual_function();
bool is_running_in_vm();
bool is_running_in_container();
bool is_virtualization_enabled();

std::string is_running_in_vm_reason();
