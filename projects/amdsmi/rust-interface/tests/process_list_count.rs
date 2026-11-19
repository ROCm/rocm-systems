// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use amdsmi::{AmdsmiProcInfoT, AmdsmiProcessorHandle, AmdsmiStatusT};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

// The fake library's running processes, how many start after each of the next
// STARTING_COUNTS counts, how many exit after the next count, and how many reads
// (calls with a capacity) it saw.
static RUNNING: AtomicU32 = AtomicU32::new(0);
static STARTS_PER_COUNT: AtomicU32 = AtomicU32::new(0);
static STARTING_COUNTS: AtomicU32 = AtomicU32::new(0);
static EXITS_AFTER_COUNT: AtomicU32 = AtomicU32::new(0);
static READS: AtomicU32 = AtomicU32::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

const FIRST_PID: u32 = 1000;

// Mirrors amdsmi_get_gpu_process_list() in amd_smi.cc: a capacity of 0, or no
// running process, is a count. Otherwise it writes up to the capacity, sets the
// count to the number running, and returns OUT_OF_RESOURCES when they don't fit.
#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_process_list(
    _handle: AmdsmiProcessorHandle,
    max_processes: *mut u32,
    list: *mut AmdsmiProcInfoT,
) -> AmdsmiStatusT {
    let running = RUNNING.load(Ordering::SeqCst);
    let capacity = *max_processes;
    if capacity == 0 || running == 0 {
        *max_processes = running;
        if STARTING_COUNTS.load(Ordering::SeqCst) > 0 {
            STARTING_COUNTS.fetch_sub(1, Ordering::SeqCst);
            RUNNING.fetch_add(STARTS_PER_COUNT.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        let exits = EXITS_AFTER_COUNT.swap(0, Ordering::SeqCst);
        RUNNING.fetch_sub(exits.min(RUNNING.load(Ordering::SeqCst)), Ordering::SeqCst);
        return AmdsmiStatusT::AmdsmiStatusSuccess;
    }
    READS.fetch_add(1, Ordering::SeqCst);
    if list.is_null() {
        return AmdsmiStatusT::AmdsmiStatusInval;
    }
    for i in 0..capacity.min(running) {
        let mut info: AmdsmiProcInfoT = std::mem::zeroed();
        info.pid = FIRST_PID + i;
        list.add(i as usize).write(info);
    }
    *max_processes = running;
    if capacity >= running {
        AmdsmiStatusT::AmdsmiStatusSuccess
    } else {
        AmdsmiStatusT::AmdsmiStatusOutOfResources
    }
}

fn set_up(running: u32, starts_per_count: u32, starting_counts: u32) {
    RUNNING.store(running, Ordering::SeqCst);
    STARTS_PER_COUNT.store(starts_per_count, Ordering::SeqCst);
    STARTING_COUNTS.store(starting_counts, Ordering::SeqCst);
    EXITS_AFTER_COUNT.store(0, Ordering::SeqCst);
    READS.store(0, Ordering::SeqCst);
}

fn handle() -> AmdsmiProcessorHandle {
    NonNull::dangling().as_ptr()
}

fn pids(list: &[AmdsmiProcInfoT]) -> Vec<u32> {
    list.iter().map(|process| process.pid).collect()
}

#[test]
fn process_list_stays_empty_when_a_process_starts_after_a_count_of_zero() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_up(0, 1, 1);
    let list = amdsmi::amdsmi_get_gpu_process_list(handle()).unwrap();
    assert!(list.is_empty(), "{} entries in an empty list", list.len());
}

#[test]
fn process_list_counts_again_when_a_process_starts_before_the_read() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_up(1, 1, 1);
    let list = amdsmi::amdsmi_get_gpu_process_list(handle()).unwrap();
    assert_eq!(pids(&list), vec![FIRST_PID, FIRST_PID + 1]);
    assert_eq!(READS.load(Ordering::SeqCst), 2);
}

#[test]
fn process_list_holds_the_processes_still_running_at_the_read() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_up(3, 0, 0);
    EXITS_AFTER_COUNT.store(2, Ordering::SeqCst);
    let list = amdsmi::amdsmi_get_gpu_process_list(handle()).unwrap();
    assert_eq!(pids(&list), vec![FIRST_PID]);
}

#[test]
fn process_list_is_empty_when_every_process_exits_before_the_read() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_up(2, 0, 0);
    EXITS_AFTER_COUNT.store(2, Ordering::SeqCst);
    let list = amdsmi::amdsmi_get_gpu_process_list(handle()).unwrap();
    assert!(list.is_empty(), "{} entries in an empty list", list.len());
}

#[test]
fn process_list_fails_after_three_reads_when_processes_keep_starting() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_up(1, 1, u32::MAX);
    let status = amdsmi::amdsmi_get_gpu_process_list(handle()).unwrap_err();
    assert_eq!(status, AmdsmiStatusT::AmdsmiStatusOutOfResources);
    assert_eq!(READS.load(Ordering::SeqCst), 3);
}
