<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocddi safety boundaries

This document describes the ownership proof expected from the current Linux
KFD driver and its AMDF and HSA adapters. The public C ABIs, their caller
preconditions, and hardware behavior still require independent qualification.

## Native reachability

A resource may be released only after the last native user of its address or
handle is known to have stopped. The intended owner transitions are:

| State | Reachability and owner rule |
| --- | --- |
| Prepared | Allocate owner metadata and external backing before a native call can publish their addresses. No public handle exists. |
| Active | The core owns native ring, control, VM, and connection state. The adapter owns public handles and any signal, scratch, or allocator backing borrowed by native state. |
| Stopping | Stop producers and callback workers, then inactivate or destroy native state. Keep all backing while any step can fail or report unfinished work. |
| Released | Drop backing only after native cleanup proves that no device or worker can reach it. |
| Quarantined | An ambiguous CREATE, failed rollback, uncertain DESTROY, or unwind retains the complete dependency graph. Where recovery cannot prove release, the process intentionally retains that graph. |

The core's queue creation transaction accepts external dependencies before it
calls KFD. HSA passes inactive and error signal storage plus scratch; AMDF
passes a scratch-memory borrow. A proved rejection releases them. An ambiguous
CREATE, failed rollback, or unwind retains them with the native dependencies.
After successful creation, the adapter keeps the returned dependencies through
queue destruction. An unpublished queue with failed cleanup is retained with
its external owners. HSA stages the public queue record's capacity, doorbell
storage, event state, and CU-mask copy before CREATE. Until publication, a
successful native queue remains paired with its signal and scratch owners even
if frontend setup unwinds.

HSA scratch fault recovery has a separate transaction. It retains the old
scratch and stages each candidate allocation in the queue record before
rewriting its control block. A successful stopped-queue update makes the
candidate current before releasing the inactive signal. A failed or unwound
update leaves the candidate in the record because firmware may have observed
its address. Candidates remain owned through native queue teardown.

HSA batch queue creation reads descriptor input fields without referencing the
output-only queue field, which may be uninitialized. It copies a caller CU mask
before writing that output; CU-mask updates likewise copy the caller mask
before changing native queue control or the stored mask.

Host `MAP_FIXED` mappings reserve an interval under the address reservation's
lock before entering the driver. That lock also covers native map and unmap.
An overlapping request is rejected without replacing a live mapping; failed
unmap keeps the interval occupied. Both the address reservation and physical
backing remain owned while a mapping can refer to them.
Device mappings likewise reserve an interval before native submission and
release it only after unmap succeeds. Occupancy is scoped to the driver's
native device VM so distinct address spaces may map the same reservation
address. An ambiguous map marks the native reservation unusable before the
tentative interval is removed. A failed host map must restore the reservation
before the core releases its tentative interval. The same driver-generic
owners enforce these rules for the current KFD driver and a fake driver.

Async HSA signal registrations retain their signal storage. Callback-triggered
shutdown requests worker exit and defers cleanup until another thread can join
the workers. The async dispatcher stops before invoking another queued callback
after a callback requests shutdown. Queue event workers keep their join handle
when they cannot join themselves. Public signal and queue index fast paths
require the caller to keep each handle live for the entire operation.
Their temporary Rust references cannot escape the access helpers. A
single-signal wait holds its borrow until that call returns, as required by
the public handle contract. Multi-signal waits keep validated raw slot
pointers while the runtime registration retains the corresponding storage.
Runtime-owned async registrations retain their own references. Soft queues
retain the caller-supplied doorbell signal while its handle remains in the
queue header.
Memory-fault notification stops calling handlers when one requests shutdown;
the worker then exits without treating the interrupted notification as an
unhandled fault. Shutdown events still notify every registered handler.
The core owns each KFD signal-event page passed to event creation. It marks
the page offered after native preflight, just before ioctl dispatch. KFD may
install an offered page before CREATE_EVENT returns an error, so that page
remains live through KFD process teardown even after a failed first event.
Explicit transfer detaches its allocation metadata; failed transfer or unwind
retains the complete owner. HSA retries without the page handle only after
the first call offered the page. A pre-dispatch failure leaves a new page
reusable or releasable. If no event succeeds after an offer, HSA quarantines
its runtime registry: a later runtime cannot reuse the page handle from the
old KFD session.
SVM prefetch workers retain every dependency and their completion signal,
including repeated handles, until the worker exits. A failed submission rolls
back the references; final shutdown joins the workers before releasing signal
storage.

AMD multi-signal waits validate handles under the runtime lock and retain each
live signal slot, including duplicate handles, until the wait returns. Invalid
handles are never dereferenced, and destruction marks retained waits for exit.
The wait entry points copy all caller input arrays before writing an optional
satisfying-value result. The C ABI permits the output to overlap an input, so
the wait must use owned comparison values after it starts producing results.
An IPC reattach creates a new retirement marker, so an earlier wait still sees
the destruction even if the same numeric handle becomes public again.
Shutdown requests cancellation and waits for active multi-signal waits to
release their references before freeing signal storage. The same in-flight
tracker covers HSA SPM and clock queries after they release the registry lock.
Shutdown waits for those native calls before releasing the device VM.

HSA logging accepts a null stream and writes through Rust's stderr
implementation after releasing the runtime registry lock. A non-null C
`FILE*` stream returns not-supported, so no C stdio import or retained
caller-owned stream is involved. An in-flight token keeps shutdown from
releasing the runtime while a prepared log write completes. Shutdown disables
subsequent prepared log records before worker and queue cleanup.

## Raw address contracts

Safe Rust must not be able to authorize retained device access to temporary
caller memory. The core marks host registration, raw queue creation and scratch
replacement, SPM destination replacement, trap-handler installation, and
kernel command submission as `unsafe`. The matching private driver methods
also require explicit unsafe calls. Owned allocation requests use a separate
safe driver method whose request type cannot contain a borrowed host address.
Registration enters through an unsafe driver method. The KFD USERPTR variant
requires an unsafe borrowed-pages token, and DRM registration has the same
explicit raw-address contract. Their safety sections specify the backing,
synchronization, and release frontier required of the adapter.
Queue destruction is also `unsafe`: a progress sample cannot exclude a
producer publishing afterward. An adapter must retire producer mappings and
stop publication before native backing can be released.
Implicit drop of a live native queue retains its backing for KFD process
teardown. An adapter must also retain any external signal or scratch backing
if it cannot prove native destruction succeeded.
An address returned as a number is information; passing it to a native engine
is the operation that requires a retained owner or an unsafe contract.

SPM destination replacement is an explicit raw contract: KFD may write after
the call returns and may copy into the previous destination while replacing
it. On failure, the old and new buffers remain caller-owned until a later
successful unset or replacement, or conclusive device teardown. A short-lived
Rust slice cannot represent this obligation.
HSA validates the SPM agent under the runtime registry lock and clones its
activated device before calling KFD. Acquire, release, and destination-update
ioctls run after that lock is released. An in-flight token delays teardown,
while the device clone keeps its VM live through the call. No result is
published into the runtime registry afterward.

## Shared process connection

The session owns a registry of driver instances. A passive endpoint names its
driver instance; activation retains that exact instance with the resulting
device state. Resource owners retain their concrete driver's cleanup state.
Session shutdown first checks that no device or resource still borrows any
driver instance. If one driver shutdown fails, the registry retains unfinished
instances for retry and never replays a successful shutdown. Linux KFD
memory and GPU operations accept only activated KFD state.

The workspace-root shared image loads both public ABIs from one shared
object. Sessions with `DriverContextLifetime::Process` share the primary KFD
context, one Linux KFD connection, and the exact DRM VM bindings acquired for
each GPU. A process owner allocated with the Rust system allocator retains
these driver handles through process exit; no frontend callback allocator is
captured by that owner. A session joins the process owner before any KFD call
that might enable runtime state. The last joined session disables KFD runtime
enablement under the owner's lock. Failure leaves the session joined so its
destruction can be retried. The shared image is marked `NODELETE`, including
after both public ABIs shut down, because a later activation must reuse the
same DRM file object.

The AMDF static archive has an independent process owner if linked into an
application; it does not share native state with the shared image.
An inherited post-fork session is rejected before touching its old locks; the
Linux KFD driver publishes a fresh process owner in the child before accessing
its mutex.

The Linux descriptor interop code validates and duplicates raw C descriptors
before a frontend creates a Rust borrowed descriptor. It also duplicates
borrowed descriptors before driver import. Positioned reads leave the caller's
shared file offset unchanged, and closing a duplicate cannot close the
original descriptor.

## Foreign call boundary

Hand-authored foreign function declarations for Linux system and device calls
are confined to `src/driver/linux_kfd/{sys,drm,util,process_identity}.rs`.
The AMDF and HSA frontends export C ABI entry points and invoke
caller-provided callbacks. The core host allocator invokes AMDF
caller-provided allocation callbacks. These ABI calls do not import a native
library. HSA has no `libdrm_amdgpu` link and obtains
DRM facts from the Linux driver's raw ioctl implementation.
The Rust standard library still gives the shared image `libc` and
`libgcc_s` dependencies. This source organization describes where rocddi
authors call foreign functions; it does not establish GPU memory safety.

## Verification scope

The workspace tests exercise native failure injection, owner retention,
callback shutdown, overlapping host mappings, descriptor ownership, and
bounded progress sampling. A fake CPU driver test checks discovery,
activation, host and device allocation ownership, and failed cleanup retry.
Queue fault injection verifies the core transaction with an external owner.
Scripted DRM ioctls exercise ambiguous map, failed map wait, ambiguous unmap,
failed unmap wait, and failed GEM close. These tests assert that the native
reservation, backing, and GEM remain retained until cleanup is proved, and
that retries do not replay a submitted unmap. A scripted command submission
with an unexpected DRM sequence keeps the caller's command storage live until
its attached timeline point signals, including through a failed destroy.
The opt-in [GFX1201 C ABI probes](../tests/gpu/README.md) inject ambiguous
CREATE and failed rollback through both frontends. HSA verifies that the
separate inactive-signal and scratch KFD allocations remain live; AMDF
verifies that its public scratch memory remains busy. These probes do not
cover every native failure or allocator callback lifetime. Independent C/Rust
ABI probes compare AMDF and HSA layouts, and the HSA probe calls the local
versioned finalizer symbol.
Focused Miri tests exercise the custom host owners and buffer, including
panicking destructors and partial iteration, under tree borrows and alignment
checks. They cannot inspect device or firmware behavior. Linux AArch64 CI
cross-checks source compilation and executes the CPU-only suite on native
hardware. Those tests cannot establish GPU firmware ordering, cache coherence,
or recovery on a device. Those require the hardware and workload qualification
described by each frontend's support documentation.
