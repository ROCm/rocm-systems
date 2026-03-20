// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "util/detect_virtualization.hpp"

// ---------------------------------------------------------------------------
// is_sriov_virtual_function
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, SriovVfReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_sriov_virtual_function());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_running_in_vm
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, VmDetectionReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_running_in_vm());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_running_in_container
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, ContainerDetectionReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_running_in_container());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_virtualization_enabled - consistency with sub-checks
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, MasterCheckConsistentWithSubChecks) {
    // Call each sub-function once, then verify the master result agrees.
    bool sriov     = is_sriov_virtual_function();
    bool vm        = is_running_in_vm();

    EXPECT_EQ(is_virtualization_enabled(), sriov || vm);
}

void print_hostname_entry() {
    char buf[256] = {};
    FILE* pipe = popen("cat /etc/hosts | grep $(hostname)", "r");
    if (!pipe) return;
    while (fgets(buf, sizeof(buf), pipe))
        std::cout << buf;
    pclose(pipe);
}

void print_bios_vendor() {
    char buf[256] = {};
    FILE* pipe = popen("cat /sys/class/dmi/id/sys_vendor", "r");
    if (!pipe) return;
    while (fgets(buf, sizeof(buf), pipe))
        std::cout << buf;
    pclose(pipe);
}

void print_kvm_signature() {
    char buf[256] = {};
    FILE* pipe = popen("cat /sys/class/dmi/id/bios_version", "r");
    if (!pipe) return;
    while (fgets(buf, sizeof(buf), pipe))
        std::cout << buf;
    pclose(pipe);
}

void print_gpu_pass_through(){
    char buf[256] = {};
    FILE* pipe = popen("lspci -nn | grep -i -E \"vga|3d|display\"", "r");
    if (!pipe) return;
    while (fgets(buf, sizeof(buf), pipe))
        std::cout << buf;
    pclose(pipe);
}

void print_kfd_ioctl_header() {
    // First, find and print the full path/name of the amdgpu* directory
    char buf[256] = {};
    FILE* pipe = popen("ls -d /usr/src/amdgpu*/", "r");
    if (!pipe) {
        std::cout << "amdgpu src dir: (not found)" << std::endl;
        return;
    }
    std::string amdgpu_dir;
    while (fgets(buf, sizeof(buf), pipe))
        amdgpu_dir += buf;
    pclose(pipe);

    // Trim trailing newline
    if (!amdgpu_dir.empty() && amdgpu_dir.back() == '\n')
        amdgpu_dir.pop_back();

    std::cout << "amdgpu src dir: " << amdgpu_dir << std::endl;

    // Now cat the kfd_ioctl.h from that directory
    std::string cmd = "cat " + amdgpu_dir + "/include/uapi/linux/kfd_ioctl.h";
    pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cout << "kfd_ioctl.h: (not readable)" << std::endl;
        return;
    }
    memset(buf, 0, sizeof(buf));
    while (fgets(buf, sizeof(buf), pipe))
        std::cout << buf;
    pclose(pipe);
}

// ---------------------------------------------------------------------------
// Expect false on bare-metal / non-VF environment
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, SriovVfReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_sriov_virtual_function());
}

TEST(DetectVirtualization, VmDetectionReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_running_in_vm());
    std::string reason = is_running_in_vm_reason();
    if (!reason.empty()) {
        std::cout << "VM detection reason: " << reason << std::endl;
        std::cout << "Hostname entry: " << std::endl;
        print_hostname_entry();
        std::cout << "BIOS vendor: " << std::endl;
        print_bios_vendor();
        std::cout << "KVM signature: " << std::endl;
        print_kvm_signature();
        std::cout << "GPU pass-through devices: " << std::endl;
        print_gpu_pass_through();
        std::cout << "KFD ioctl header: " << std::endl;
        print_kfd_ioctl_header();
    }
}

TEST(DetectVirtualization, ContainerDetectionReturnsFalseOnBareMetal) {
    std::cout << "KFD ioctl header: " << std::endl;
    print_kfd_ioctl_header();
    EXPECT_FALSE(is_running_in_container());
}

TEST(DetectVirtualization, VirtualizationEnabledReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_virtualization_enabled());
}