"""Test GDA queue-pair introspection API (C and Python bindings).

Validates :func:`rocshmem4py.rocshmem_query_qp_info`, which exposes GDA RC QP
metadata for external WQE construction (see ``rocshmem/qp_introspect.hpp``).
The C API returns false when the GDA backend is not active; the Python binding
returns ``None`` in that case.

Only the ionic vendor fields are implemented; mlx5 and bnxt are declared but
unimplemented and return false at the C layer.

Every test depends on the ``gda_qp`` fixture, which skips the module when
introspection is unavailable. That is deliberate: without it an implementation
that always returned ``None`` would satisfy this whole file -- the positive
assertions would be skipped and the negative ones ("invalid input yields None")
would hold trivially. Gating on a known-good QP means a stub cannot pass.

The API splits its failure modes, and the tests check both halves:
  * ``ValueError``    -- caller error (ctx_id <= 0, peer outside [0, n_pes)).
  * ``None``          -- not applicable here: backend is not GDA, provider
                         unsupported, or the ctx_id names no QP.
  * ``rocshmem_qp_introspect_available()`` answers the environment question up
    front, so callers need not infer capability from a null result.

Run under torchrun (conftest selects init_with_torch when torch is present):

    torchrun --nnodes=2 --node_rank=<0|1> --nproc_per_node=1 \
      --master_addr=<ip> --master_port=<port> \
      -m pytest python/rocshmem/tests/test_qp_introspect.py -v

conftest's mpi4py path works as well. Plain ``mpirun`` + ``rocshmem_init()``
needs an MPI that can create shared-memory windows, which is not available in
every container.
"""

import pytest
import rocshmem4py
from conftest import requires_multi_pe, requires_torch

# ctx_id 0 is the default context and is rejected by the C API; user contexts
# start at 1. See qp_introspect.hpp on why describing the default-context QP is
# unsafe for external WQE posting.
USER_CTX = 1


@pytest.fixture(scope="module")
def gda_qp():
    """A known-good QpInfo, or skip the module.

    Guarantees every other test runs against a live GDA backend, so none of them
    can pass vacuously.
    """
    qp = rocshmem4py.rocshmem_query_qp_info(0, USER_CTX)
    if qp is None:
        pytest.skip(
            "GDA introspection unavailable (rocshmem_query_qp_info returned "
            "None for peer 0, ctx 1) -- not the GDA backend, or built without it"
        )
    return qp


def test_qp_info_core_fields_populated(gda_qp):
    """Every vendor-neutral field is present and plausibly initialised."""
    assert isinstance(gda_qp, rocshmem4py.QpInfo)
    assert isinstance(gda_qp.vendor, rocshmem4py.QpInfoVendor)

    assert gda_qp.sq_buf > 0, "sq_buf should be a valid device address"
    assert gda_qp.sq_prod > 0, "sq_prod should be a valid device address"
    assert gda_qp.cq_buf > 0, "cq_buf should be a valid device address"
    assert gda_qp.base_heap > 0, "base_heap should be a valid device address"
    assert gda_qp.sq_depth > 0, "sq_depth should be positive"
    assert gda_qp.cq_depth > 0, "cq_depth should be positive"
    assert gda_qp.lkey != 0, "lkey should be set"
    assert gda_qp.rkey != 0, "rkey should be set"
    assert gda_qp.qpn != 0, "qpn should be set"


def test_qp_info_distinct_per_peer_and_context(gda_qp):
    """Each (peer, ctx) names its own QP, so the descriptors must not collide.

    A plausible implementation bug is to ignore ctx_id, or to return the same
    connection for every peer; either would show up as duplicate qpn/sq_buf.
    """
    n_pes = rocshmem4py.rocshmem_n_pes()
    seen = {}
    for peer in range(n_pes):
        for ctx_id in (1, 2):
            qp = rocshmem4py.rocshmem_query_qp_info(peer, ctx_id)
            if qp is None:
                continue
            key = (qp.qpn, qp.sq_buf)
            assert key not in seen, (
                f"(peer={peer}, ctx={ctx_id}) returned the same QP as "
                f"{seen[key]}: qpn={qp.qpn} sq_buf={qp.sq_buf:#x}"
            )
            seen[key] = (peer, ctx_id)

    assert len(seen) >= 2, "expected several distinct QPs across peers/contexts"


@requires_torch
@requires_multi_pe
def test_rkey_matches_peer_lkey(gda_qp):
    """The rkey this PE holds for a peer must equal that peer's own lkey.

    This is the one assertion here that checks the values are *correct* rather
    than merely present: the rkey is what a remote NIC quotes when writing into
    our memory region, so if the key exchange were wrong the numbers would not
    line up -- and no amount of non-zero checking would notice.
    """
    import torch.distributed as dist

    me = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()

    mine = {"pe": me, "lkey": gda_qp.lkey, "rkey_for": {}}
    for peer in range(n_pes):
        qp = rocshmem4py.rocshmem_query_qp_info(peer, USER_CTX)
        if qp is not None:
            mine["rkey_for"][peer] = qp.rkey

    gathered = [None] * n_pes
    dist.all_gather_object(gathered, mine)
    by_pe = {g["pe"]: g for g in gathered if g is not None}

    checked = 0
    for peer, rkey in mine["rkey_for"].items():
        if peer == me or peer not in by_pe:
            continue
        assert rkey == by_pe[peer]["lkey"], (
            f"pe{me} holds rkey={rkey:#x} for pe{peer}, but pe{peer} reports "
            f"lkey={by_pe[peer]['lkey']:#x} -- key exchange is inconsistent"
        )
        checked += 1

    assert checked > 0, "no remote peer to check against"


def test_vendor_fields_raise_attribute_error_on_mismatch(gda_qp):
    """Vendor-specific properties raise AttributeError on the wrong vendor.

    QpInfo is a tagged union: only the arm named by ``vendor`` is defined, so the
    accessors gate on the tag rather than returning whatever bytes are there.
    AttributeError (not RuntimeError) is required -- ``hasattr`` swallows only
    AttributeError, so any other type makes ``hasattr(qp, "mlx5_dbrec")`` raise
    instead of answering False.
    """
    if gda_qp.vendor != rocshmem4py.QpInfoVendor.IONIC:
        pytest.skip(f"vendor {gda_qp.vendor} is implemented; test needs updating")

    for field in ("ionic_db", "ionic_dbval", "ionic_sq_mask",
                  "ionic_cq_mask", "ionic_udma_idx"):
        assert hasattr(gda_qp, field), f"ionic QpInfo should expose {field}"

    for field in ("mlx5_dbrec", "mlx5_bf", "bnxt_dbr"):
        with pytest.raises(AttributeError, match=field.split("_")[0]):
            getattr(gda_qp, field)
        assert not hasattr(gda_qp, field), (
            f"hasattr({field}) must be False on an ionic QpInfo; if this fails "
            "the accessor is raising something other than AttributeError"
        )


def test_introspection_reports_available(gda_qp):
    """The capability predicate agrees with the fixture that got a real QP."""
    assert rocshmem4py.rocshmem_qp_introspect_available() is True


def test_available_implies_a_supported_provider(gda_qp):
    """available() must track the provider, not merely the backend.

    Checking only for a GDA backend would report True on mlx5 or bnxt, where
    query_qp_info then returns None -- the caller is told to go ahead and gets
    nothing back, with no way to see why. available() is therefore true only for
    providers this API implements, and provider() reports what was detected so a
    caller can tell "no GDA here" from "GDA on a NIC we do not support yet".
    """
    provider = rocshmem4py.rocshmem_qp_introspect_provider()
    assert provider != rocshmem4py.QpInfoVendor.UNKNOWN, (
        "a QP was obtained, so a provider must have been detected"
    )
    assert provider == gda_qp.vendor, "provider() disagrees with the QP's vendor tag"

    if rocshmem4py.rocshmem_qp_introspect_available():
        assert rocshmem4py.rocshmem_query_qp_info(0, USER_CTX) is not None, (
            "available() promised introspection but query_qp_info refused it"
        )


def test_default_context_raises(gda_qp):
    """ctx_id <= 0 is a caller error: that QP is the one rocSHMEM drives itself.

    The default context's producer index, send-queue lock and cached doorbell
    position are private to rocSHMEM, and an external descriptor builder updates
    none of them. Measured on ionic, external posts alone leave collectives
    working, but a workload that also polls completions on that shared queue
    does not finish, where the same workload on a user context does. Rejected
    rather than left to documentation. It raises rather than returning None
    because it is a mistake in the call, not a property of the environment.
    """
    for ctx_id in (0, -1):
        with pytest.raises(ValueError, match="ctx_id"):
            rocshmem4py.rocshmem_query_qp_info(0, ctx_id)


def test_out_of_range_peer_raises(gda_qp):
    """A peer outside [0, n_pes) is a caller error and raises."""
    n_pes = rocshmem4py.rocshmem_n_pes()
    for peer in (n_pes + 10, -1):
        with pytest.raises(ValueError, match="peer"):
            rocshmem4py.rocshmem_query_qp_info(peer, USER_CTX)


def test_unnamed_context_returns_none(gda_qp):
    """A ctx_id past the configured context count names no QP -> None.

    Distinct from the ValueError cases: the argument is well-formed, there just
    is no such queue pair in this configuration. That is a property of the
    environment, so it is reported the same way an unsupported backend is.
    """
    assert rocshmem4py.rocshmem_query_qp_info(0, 999) is None
