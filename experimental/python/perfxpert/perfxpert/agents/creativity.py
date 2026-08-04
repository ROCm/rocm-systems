"""Bounded creative freedom for Layer-2 specialists (RFC 0001).

PerfXpert's specialists select from vetted YAML catalogs. That is what makes
their advice defensible, and it is also why they cannot say anything about a
workload the catalogs do not cover. This module adds a second, clearly
separated lane: a model may *propose* a novel technique, but a proposal is
never a recommendation.

The separation is structural rather than advisory:

- The vetted lane keeps its existing shape and stays deterministic. Nothing
  here can add to, reorder, or reword it.
- The exploratory lane carries proposals the runtime built. The model emits
  an :class:`ExploratoryProposalDraft`; the runtime validates it, resolves
  every evidence reference against a manifest of tool calls that actually
  happened, checks the target kernel against measured kernels, stamps the
  fields it owns, and only then constructs the final proposal.

A proposal therefore cannot cite a tool that was never called, name a kernel
that was never measured, or assign itself an identity or provenance.

This module is inert until a session resolves to the ``exploratory`` tier,
which :func:`resolve_tier` grants only when the deployment ceiling, a live
session, and the agent's own capability all independently permit it. Any one
of them withheld -- including the default configuration -- yields ``strict``,
and a strict session never reaches proposal construction at all.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from contextvars import ContextVar
from enum import Enum
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from perfxpert.agents import schemas
from perfxpert.agents.framework import AgentCapability

_LOG = logging.getLogger(__name__)

# Confidence ceiling for an unverified proposal. Not a claim that novel ideas
# are unlikely — it stops an unmeasured proposal from numerically
# impersonating a proven recommendation. Lane separation, not this number, is
# the real safety mechanism.
MAX_EXPLORATORY_CONFIDENCE = 0.5

# Bounds the blast radius of a verbose or adversarial model.
MAX_PROPOSALS_PER_SPECIALIST = 3

PROPOSAL_ID_PREFIX = "pxp-exp-"


class CreativityTier(str, Enum):
    """Session ceiling on model-generated content."""

    STRICT = "strict"
    EXPLORATORY = "exploratory"


class ProposalRejected(ValueError):
    """A draft failed validation and must not become a proposal."""


def resolve_tier(
    configured_max: CreativityTier,
    *,
    airgap: bool,
    capability: AgentCapability,
) -> CreativityTier:
    """Resolve the effective tier for one agent in one session.

    Every input must permit exploration for it to be enabled. Air-gap runs
    stay strict because there is no model to propose anything and decision
    parity with live runs is an invariant.
    """
    if configured_max is not CreativityTier.EXPLORATORY:
        return CreativityTier.STRICT
    if airgap:
        return CreativityTier.STRICT
    if capability is not AgentCapability.ADDITIVE_EXPLORATION:
        return CreativityTier.STRICT
    return CreativityTier.EXPLORATORY


# -- Evidence manifest ----------------------------------------------------


class EvidenceManifest:
    """The tool calls and kernels a proposal is allowed to refer to.

    Built by the runtime from what actually ran. A draft citing anything
    outside it is rejected, so a model cannot invent supporting evidence.
    """

    def __init__(
        self,
        *,
        tool_calls: Optional[Iterable[str]] = None,
        kernels: Optional[Iterable[str]] = None,
        catalog_entries: Optional[Iterable[str]] = None,
    ) -> None:
        self.tool_calls = frozenset(tool_calls or ())
        self.kernels = frozenset(kernels or ())
        self.catalog_entries = frozenset(catalog_entries or ())

    def permits_reference(self, kind: str, ref: str) -> bool:
        if kind == "tool":
            return ref in self.tool_calls
        if kind == "kernel":
            return ref in self.kernels
        if kind == "catalog":
            return ref in self.catalog_entries
        return False

    def permits_kernel(self, kernel: Optional[str]) -> bool:
        if kernel is None:
            return True
        return kernel in self.kernels

    def is_empty(self) -> bool:
        return not (self.tool_calls or self.kernels or self.catalog_entries)


# -- Deterministic identity -----------------------------------------------


def _digest_text(text: str) -> str:
    """Full-length digest of a text artefact, or empty when there is none."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_json(value: Any) -> str:
    """Full-length digest of a JSON-serialisable artefact, or empty."""
    if value is None:
        return ""
    try:
        payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workload_fingerprint(*parts: Any) -> str:
    """Stable digest of the workload a set of proposals was made about.

    Two runs over the same workload produce the same value, and two different
    workloads do not. What goes in is up to the caller, but it must be
    measured input rather than anything the model can influence, or the
    identity it feeds becomes forgeable.
    """
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_proposal_id(
    draft: "schemas.ExploratoryProposalDraft",
    *,
    specialist: str,
    trace_fingerprint: str = "",
) -> str:
    """Derive a stable ID from proposal content and the workload it targets.

    Content-addressed so the same proposal for the same workload keeps one
    identity across runs, which is what lets Recommendation deduplicate and
    what a human reviewer promotes against. The model does not supply it.

    The workload is part of the material because the same sentence about two
    different workloads is two different claims: without it, a proposal made
    against one trace would collapse into a proposal made against another and
    a reviewer's decision on one would silently apply to the other.
    """
    material = {
        "specialist": specialist,
        "title": draft.title,
        "hypothesis": draft.hypothesis,
        "mechanism": draft.mechanism,
        "target_kernel": draft.target_kernel,
        "trace_fingerprint": trace_fingerprint,
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{PROPOSAL_ID_PREFIX}{digest}"


# -- Draft validation -----------------------------------------------------


def validate_draft(
    draft: "schemas.ExploratoryProposalDraft",
    *,
    manifest: EvidenceManifest,
) -> None:
    """Raise :class:`ProposalRejected` if a draft may not become a proposal."""
    if not draft.evidence:
        raise ProposalRejected("proposal cites no evidence")

    for item in draft.evidence:
        if not manifest.permits_reference(item.kind, item.ref):
            raise ProposalRejected(
                f"evidence {item.kind}:{item.ref!r} was not produced during this run"
            )

    if not manifest.permits_kernel(draft.target_kernel):
        raise ProposalRejected(
            f"target kernel {draft.target_kernel!r} is not among the measured kernels"
        )

    if draft.confidence > MAX_EXPLORATORY_CONFIDENCE:
        raise ProposalRejected(
            f"confidence {draft.confidence} exceeds the exploratory ceiling "
            f"{MAX_EXPLORATORY_CONFIDENCE}"
        )


def build_proposal(
    draft: "schemas.ExploratoryProposalDraft",
    *,
    specialist: str,
    manifest: EvidenceManifest,
    provenance: "schemas.ProposalProvenance",
) -> "schemas.ExploratoryProposal":
    """Validate a draft and construct the runtime-owned proposal."""
    validate_draft(draft, manifest=manifest)
    return schemas.ExploratoryProposal(
        proposal_id=compute_proposal_id(
            draft,
            specialist=specialist,
            trace_fingerprint=provenance.trace_fingerprint,
        ),
        status="exploratory",
        specialist=specialist,
        title=draft.title,
        target_kernel=draft.target_kernel,
        hypothesis=draft.hypothesis,
        mechanism=draft.mechanism,
        evidence=list(draft.evidence),
        expected_effects=list(draft.expected_effects),
        verification=draft.verification,
        assumptions=list(draft.assumptions),
        failure_modes=list(draft.failure_modes),
        confidence=draft.confidence,
        provenance=provenance,
    )


def build_proposals(
    drafts: Sequence[Any],
    *,
    specialist: str,
    manifest: EvidenceManifest,
    provenance: "schemas.ProposalProvenance",
) -> Tuple[List["schemas.ExploratoryProposal"], List[str]]:
    """Build every acceptable proposal, reporting why the others were dropped.

    One bad draft must not discard the rest, and a rejection must be
    explainable rather than silent, so rejections come back alongside the
    accepted proposals.
    """
    accepted: List[schemas.ExploratoryProposal] = []
    rejected: List[str] = []
    seen_ids = set()

    for raw in drafts[:MAX_PROPOSALS_PER_SPECIALIST]:
        try:
            draft = (
                raw
                if isinstance(raw, schemas.ExploratoryProposalDraft)
                else schemas.ExploratoryProposalDraft(**raw)
            )
        except Exception as exc:
            rejected.append(f"malformed draft: {exc}")
            continue
        try:
            proposal = build_proposal(
                draft,
                specialist=specialist,
                manifest=manifest,
                provenance=provenance,
            )
        except ProposalRejected as exc:
            rejected.append(str(exc))
            continue
        if proposal.proposal_id in seen_ids:
            continue
        seen_ids.add(proposal.proposal_id)
        accepted.append(proposal)

    if len(drafts) > MAX_PROPOSALS_PER_SPECIALIST:
        rejected.append(
            f"discarded {len(drafts) - MAX_PROPOSALS_PER_SPECIALIST} draft(s) over the "
            f"{MAX_PROPOSALS_PER_SPECIALIST}-proposal cap"
        )
    return accepted, rejected


def dedupe_proposals(
    proposals: Iterable["schemas.ExploratoryProposal"],
) -> List["schemas.ExploratoryProposal"]:
    """Deduplicate by server-generated ID only.

    Deliberately not by title or confidence: those are model-supplied, so
    comparing them would let a model suppress another specialist's proposal
    by mimicking it.
    """
    seen = set()
    out: List[schemas.ExploratoryProposal] = []
    for proposal in proposals:
        if proposal.proposal_id in seen:
            continue
        seen.add(proposal.proposal_id)
        out.append(proposal)
    return out


# -- Runtime entry point --------------------------------------------------


# The ceiling the running session resolved at build time. A session reads
# configuration once and publishes the result here, so every agent in that
# session is judged against the same ceiling. Without it each call re-reads
# config and a mid-session edit could move a specialist from strict to
# exploratory between its own gate check and its proposal extraction.
_ACTIVE_CEILING: ContextVar[Optional[CreativityTier]] = ContextVar(
    "perfxpert_creativity_ceiling", default=None
)


@contextlib.contextmanager
def active_ceiling(ceiling: Optional[CreativityTier]) -> Iterator[None]:
    """Pin the creativity ceiling for every agent run in this context."""
    token = _ACTIVE_CEILING.set(ceiling)
    try:
        yield
    finally:
        _ACTIVE_CEILING.reset(token)


def configured_ceiling() -> CreativityTier:
    """Ceiling for the current session, from the session or else configuration.

    Deliberately not a function argument anywhere it is used: the ceiling is
    a deployment decision, and a parameter would give a calling model a
    handle to raise its own limit. A session's captured value wins over a
    fresh config read so the ceiling cannot move while a session is in
    flight; only calls made outside any session read configuration directly.
    """
    session_ceiling = _ACTIVE_CEILING.get()
    if session_ceiling is not None:
        return session_ceiling
    try:
        from perfxpert.config import load_config

        return CreativityTier(load_config().agent_creativity)
    except Exception:  # pragma: no cover - unreadable config falls back safely
        return CreativityTier.STRICT


def effective_tier(agent: Any, *, airgap: bool) -> CreativityTier:
    return resolve_tier(
        configured_ceiling(), airgap=airgap, capability=agent.capability
    )


def manifest_from_run(
    agent: Any,
    raw: dict,
    *,
    kernels: Iterable[str] = (),
    catalog_entries: Iterable[str] = (),
) -> EvidenceManifest:
    """Build the manifest from what this run actually did.

    Only tools the agent both declares and called are admissible. The SDK
    reports sanitised names (dots rewritten to underscores), so those are
    mapped back to the declared names before matching.
    """
    declared = {t.name for t in agent.tools}
    by_sanitized = {name.replace(".", "_"): name for name in declared}

    called = set()
    for entry in raw.get("tool_calls") or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name in declared:
            called.add(name)
        elif name in by_sanitized:
            called.add(by_sanitized[name])

    return EvidenceManifest(
        tool_calls=called, kernels=kernels, catalog_entries=catalog_entries
    )


def proposals_from_response(
    agent: Any,
    raw: dict,
    *,
    specialist: str,
    airgap: bool,
    manifest: EvidenceManifest,
    provider: str = "",
    field_path: Sequence[str] = ("exploratory_proposals",),
    tier: Optional[CreativityTier] = None,
    trace_fingerprint: str = "",
    catalog: Any = None,
) -> List["schemas.ExploratoryProposal"]:
    """Turn a model response's drafts into validated proposals.

    ``field_path`` is where this agent's schema puts the lane. Most declare it
    at the top level; Diff is at its output field cap and nests it under
    ``kernel_deltas``, and a draft the model emits somewhere the schema does
    not declare is discarded by output validation before it gets here.

    ``tier`` is the caller's already-resolved tier. Callers that gate on the
    tier before invoking the model should pass the value they gated on, so the
    decision to consult the model and the decision to keep its proposals are
    one decision rather than two reads that could disagree.

    Returns an empty list whenever the effective tier is strict, which is
    every current configuration. Any failure here yields no proposals rather
    than propagating: the exploratory lane is an addition, so it must never
    be able to break a run that would otherwise have succeeded.
    """
    resolved = tier if tier is not None else effective_tier(agent, airgap=airgap)
    if resolved is not CreativityTier.EXPLORATORY:
        return []

    node: Any = raw.get("structured_output") or {}
    for key in field_path:
        if not isinstance(node, dict):
            return []
        node = node.get(key)
    drafts = node or []
    if not isinstance(drafts, list) or not drafts:
        return []

    # Every field here is stamped from what the runtime knows, never from the
    # response. Together they let a reviewer reproduce the conditions a
    # proposal was made under: which model said it, about which workload,
    # under which fence text, against which catalog.
    provenance = schemas.ProposalProvenance(
        provider=provider,
        model=str(raw.get("model", "") or ""),
        trace_fingerprint=trace_fingerprint,
        fence_sha256=_digest_text(getattr(agent, "fence_text", "") or ""),
        catalog_sha256=_digest_json(catalog),
    )
    try:
        accepted, rejected = build_proposals(
            drafts,
            specialist=specialist,
            manifest=manifest,
            provenance=provenance,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("creativity: proposal construction failed (%s)", exc)
        return []

    for reason in rejected:
        _LOG.info("creativity: rejected %s proposal — %s", specialist, reason)
    return accepted


__all__ = [
    "AgentCapability",
    "CreativityTier",
    "active_ceiling",
    "configured_ceiling",
    "effective_tier",
    "workload_fingerprint",
    "manifest_from_run",
    "proposals_from_response",
    "EvidenceManifest",
    "ProposalRejected",
    "MAX_EXPLORATORY_CONFIDENCE",
    "MAX_PROPOSALS_PER_SPECIALIST",
    "PROPOSAL_ID_PREFIX",
    "build_proposal",
    "build_proposals",
    "compute_proposal_id",
    "dedupe_proposals",
    "resolve_tier",
    "validate_draft",
]
