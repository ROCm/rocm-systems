// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Reentrant reader admission for executable traversal callbacks.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Condvar, Mutex};

thread_local! {
    static READ_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct GateState {
    readers: usize,
    waiting_writers: usize,
    writer: bool,
}

/// Excludes executable create and destroy while callbacks visit handles.
/// A callback may start another traversal even when a writer is waiting.
pub(crate) struct ExecutableTraversalGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

pub(crate) struct TraversalRead<'a> {
    gate: &'a ExecutableTraversalGate,
    _same_thread: PhantomData<Rc<()>>,
}

pub(crate) struct TraversalWrite<'a> {
    gate: &'a ExecutableTraversalGate,
    _same_thread: PhantomData<Rc<()>>,
}

impl ExecutableTraversalGate {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                readers: 0,
                waiting_writers: 0,
                writer: false,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn read(&self) -> Result<TraversalRead<'_>, ()> {
        let depth = READ_DEPTH.with(Cell::get);
        let next_depth = depth.checked_add(1).ok_or(())?;
        let mut state = self.state.lock().map_err(|_| ())?;
        while state.writer || (state.waiting_writers != 0 && depth == 0) {
            state = self.changed.wait(state).map_err(|_| ())?;
        }
        state.readers = state.readers.checked_add(1).ok_or(())?;
        READ_DEPTH.with(|current| current.set(next_depth));
        Ok(TraversalRead {
            gate: self,
            _same_thread: PhantomData,
        })
    }

    pub(crate) fn write(&self) -> Result<TraversalWrite<'_>, ()> {
        // Mutating an executable from its own traversal callback cannot
        // finish while that callback still holds a stable handle snapshot.
        if READ_DEPTH.with(|depth| depth.get() != 0) {
            return Err(());
        }
        let mut state = self.state.lock().map_err(|_| ())?;
        state.waiting_writers = state.waiting_writers.checked_add(1).ok_or(())?;
        while state.writer || state.readers != 0 {
            state = self.changed.wait(state).map_err(|_| ())?;
        }
        state.waiting_writers -= 1;
        state.writer = true;
        Ok(TraversalWrite {
            gate: self,
            _same_thread: PhantomData,
        })
    }
}

impl Drop for TraversalRead<'_> {
    fn drop(&mut self) {
        let depth = READ_DEPTH.with(Cell::get);
        READ_DEPTH.with(|current| current.set(depth - 1));
        let mut state = self
            .gate
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.readers -= 1;
        if state.readers == 0 {
            self.gate.changed.notify_all();
        }
    }
}

impl Drop for TraversalWrite<'_> {
    fn drop(&mut self) {
        let mut state = self
            .gate
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.writer = false;
        self.gate.changed.notify_all();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn nested_traversal_passes_a_waiting_writer() {
        let gate = Arc::new(ExecutableTraversalGate::new());
        let outer = gate.read().unwrap();
        let writer_gate = gate.clone();
        let writer = std::thread::spawn(move || {
            let _guard = writer_gate.write().unwrap();
        });
        let start = Instant::now();
        while gate.state.lock().unwrap().waiting_writers == 0 {
            assert!(start.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }
        let nested = gate.read().unwrap();
        assert!(gate.write().is_err());
        drop(nested);
        drop(outer);
        writer.join().unwrap();
    }
}
