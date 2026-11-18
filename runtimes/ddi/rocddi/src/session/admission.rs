// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Session-wide admission independent of the driver registry's ownership.

use super::SessionState;
use crate::os::ProcessIdentity;
use crate::{Error, ErrorKind};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const OPEN: usize = 0;
const RETIRING: usize = 1;
const CLOSED: usize = 2;
const PHASE_MASK: usize = 3;
const OBLIGATION: usize = 4;

thread_local! {
    // Native transactions can call other core services. A nested call on the
    // same thread belongs to its already admitted outer transaction.
    static OPERATIONS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct Control {
    process: ProcessIdentity,
    // One atomic linearizes admission against retirement. Higher bits count
    // operations and resource leases; low bits encode SessionState.
    state: AtomicUsize,
    used: AtomicBool,
    attached: bool,
}

/// Shared lifetime authority retained by drivers without retaining the registry.
#[derive(Clone)]
pub(crate) struct SessionAdmission {
    control: Arc<Control>,
}

/// One admitted transaction, including nested calls and owner publication.
/// This guard must be released on its admitting thread.
pub(crate) struct SessionOperation {
    admission: SessionAdmission,
    _thread: PhantomData<Rc<()>>,
}

struct ResourceLease {
    admission: SessionAdmission,
}

/// Counted ownership for resources that do not retain their concrete driver.
#[derive(Clone)]
pub(crate) struct SessionLease {
    _owner: Arc<ResourceLease>,
}

impl SessionAdmission {
    /// Constructs standalone driver admission before installation in a session.
    #[cfg(test)]
    pub(crate) fn new(process: ProcessIdentity) -> Self {
        Self::create(process, false)
    }

    pub(super) fn for_session(process: ProcessIdentity) -> Self {
        Self::create(process, true)
    }

    fn create(process: ProcessIdentity, attached: bool) -> Self {
        Self {
            control: Arc::new(Control {
                process,
                state: AtomicUsize::new(OPEN),
                used: AtomicBool::new(false),
                attached,
            }),
        }
    }

    /// Binds a fresh, exclusively owned test driver before registry publication.
    pub(crate) fn bind(&mut self, session: &Self) -> Result<(), Error> {
        self.check_process()?;
        if Arc::ptr_eq(&self.control, &session.control) {
            return Ok(());
        }
        if self.control.attached
            || self.control.used.load(Ordering::Acquire)
            || self.state()? != SessionState::Open
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "session requires an unused, unbound driver",
            });
        }
        *self = session.clone();
        Ok(())
    }

    fn check_process(&self) -> Result<(), Error> {
        self.control.process.check().map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "session belongs to another process",
        })
    }

    pub(super) fn state(&self) -> Result<SessionState, Error> {
        self.check_process()?;
        Ok(
            match self.control.state.load(Ordering::Acquire) & PHASE_MASK {
                OPEN => SessionState::Open,
                RETIRING => SessionState::Retiring,
                _ => SessionState::Closed,
            },
        )
    }

    /// Admits one transaction, or joins its same-thread outer transaction.
    pub(crate) fn enter(&self) -> Result<SessionOperation, Error> {
        self.check_process()?;
        let identity = Arc::as_ptr(&self.control) as usize;
        OPERATIONS
            .try_with(|operations| {
                let mut operations = operations.borrow_mut();
                operations.try_reserve(1).map_err(|_| Error::Capacity {
                    resource: "session operation scope",
                })?;
                let nested = operations.contains(&identity);
                self.control
                    .state
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                        let phase = state & PHASE_MASK;
                        if phase == OPEN || (phase == RETIRING && nested) {
                            state.checked_add(OBLIGATION)
                        } else {
                            None
                        }
                    })
                    .map_err(|state| Error::Operation {
                        kind: if state & PHASE_MASK == OPEN {
                            ErrorKind::ResourceExhausted
                        } else {
                            ErrorKind::Unsupported
                        },
                        detail: "session cannot admit a new operation",
                    })?;
                self.control.used.store(true, Ordering::Release);
                operations.push(identity);
                Ok(SessionOperation {
                    admission: self.clone(),
                    _thread: PhantomData,
                })
            })
            .map_err(|_| Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "session admission is unavailable during thread teardown",
            })?
    }

    /// Retains a native borrower even when it does not own a driver reference.
    pub(crate) fn lease(&self) -> Result<SessionLease, Error> {
        let _operation = self.enter()?;
        self.control
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                state.checked_add(OBLIGATION)
            })
            .map_err(|_| Error::Capacity {
                resource: "session resource leases",
            })?;
        Ok(SessionLease {
            _owner: Arc::new(ResourceLease {
                admission: self.clone(),
            }),
        })
    }

    pub(crate) fn begin_retirement(&self) -> Result<(), Error> {
        self.check_process()?;
        let _ = self
            .control
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                if state & PHASE_MASK == OPEN {
                    Some(state | RETIRING)
                } else {
                    None
                }
            });
        Ok(())
    }

    pub(super) fn is_idle(&self) -> Result<bool, Error> {
        self.check_process()?;
        Ok(self.control.state.load(Ordering::Acquire) < OBLIGATION)
    }

    /// Closes admission only if no operation raced the ownership preflight.
    pub(super) fn retire_if_idle(&self) -> Result<bool, Error> {
        self.check_process()?;
        Ok(self
            .control
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state < OBLIGATION).then_some(if state == OPEN { RETIRING } else { state })
            })
            .is_ok())
    }

    pub(super) fn finish_retirement(&self) {
        // The caller owns the registry exclusively, discharged all drivers,
        // and proved that no operation or resource lease remains.
        self.control.state.store(CLOSED, Ordering::Release);
    }
}

impl Drop for SessionOperation {
    fn drop(&mut self) {
        if self.admission.check_process().is_err() {
            return;
        }
        let identity = Arc::as_ptr(&self.admission.control) as usize;
        let _ = OPERATIONS.try_with(|operations| {
            let mut operations = operations.borrow_mut();
            if let Some(index) = operations.iter().rposition(|entry| *entry == identity) {
                operations.remove(index);
            }
        });
        self.admission
            .control
            .state
            .fetch_sub(OBLIGATION, Ordering::AcqRel);
    }
}

impl Drop for ResourceLease {
    fn drop(&mut self) {
        if self.admission.check_process().is_ok() {
            self.admission
                .control
                .state
                .fetch_sub(OBLIGATION, Ordering::AcqRel);
        }
    }
}
