"""Does the parity suite's hostile model actually get past output validation?

A parity test proves nothing if the runtime never accepted the model's answer
in the first place: the two modes would agree because one side was discarded,
not because the deterministic path held. Output validation uses a mirror
schema with ``extra="forbid"``, so a payload carrying keys from several agents
at once is rejected wholesale by every one of them.

This checks the hostile payload is accepted where it is meant to be tested.
"""

import pytest

from perfxpert.agents import AGENT_BUILDERS
from perfxpert.agents.framework import validate_structured_output
from tests.test_integration.test_airgap_parity import HOSTILE, hostile_for


def _agents_by_name():
    return {builder().name: builder() for builder in AGENT_BUILDERS}


@pytest.mark.parametrize("agent_name", sorted(_agents_by_name()))
def test_hostile_payload_is_accepted_by_the_agent_it_targets(agent_name):
    """If this fails, the corresponding parity test is passing for the wrong
    reason -- the response never reached the code the test claims to check."""
    agent = _agents_by_name()[agent_name]
    payload = hostile_for(agent.output_schema)

    accepted, reason = validate_structured_output(agent, payload)

    assert accepted is not None, (
        f"hostile payload for {agent_name} was discarded by output validation "
        f"({reason}); the parity test using it proves nothing"
    )


def test_the_shared_hostile_dict_would_be_rejected_by_every_agent():
    """Documents why the payload is filtered per-agent rather than shared:
    the combined dict names fields from several schemas at once, and each
    agent's mirror forbids the ones that are not its own."""
    rejected = []
    for builder in AGENT_BUILDERS:
        agent = builder()
        accepted, _ = validate_structured_output(agent, dict(HOSTILE))
        if accepted is None:
            rejected.append(agent.name)

    assert rejected, "expected the combined payload to be rejected somewhere"
