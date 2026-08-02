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

This module is inert until a session resolves to the ``exploratory`` tier;
:func:`resolve_tier` returns ``strict`` for every current configuration.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from perfxpert.agents import schemas
from perfxpert.agents.framework import AgentCapability

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


def compute_proposal_id(draft: "schemas.ExploratoryProposalDraft", *, specialist: str) -> str:
    """Derive a stable ID from proposal content.

    Content-addressed so the same proposal for the same workload keeps one
    identity across runs, which is what lets Recommendation deduplicate and
    what a human reviewer promotes against. The model does not supply it.
    """
    material = {
        "specialist": specialist,
        "title": draft.title,
        "hypothesis": draft.hypothesis,
        "mechanism": draft.mechanism,
        "target_kernel": draft.target_kernel,
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
        proposal_id=compute_proposal_id(draft, specialist=specialist),
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


__all__ = [
    "AgentCapability",
    "CreativityTier",
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
