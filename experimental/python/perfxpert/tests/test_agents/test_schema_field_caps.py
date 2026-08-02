"""CI guardrail: input ≤10 fields, output ≤5 fields per agent (spec §2).

Schemas are read off the agent definitions rather than a hand-written name
list, so a newly added agent is covered without touching this file.
"""

import pytest
from pydantic import BaseModel

from perfxpert.agents import AGENT_BUILDERS


def _agent_id(builder):
    return builder.__name__


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_declared_schemas_are_pydantic_models(builder):
    """The framework validates against these, so they must be real models."""
    agent = builder()
    for role, cls in (("input", agent.input_schema), ("output", agent.output_schema)):
        assert isinstance(cls, type) and issubclass(cls, BaseModel), (
            f"{agent.name} declares a non-Pydantic {role}_schema: {cls!r}"
        )


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_input_has_at_most_10_fields(builder):
    agent = builder()
    cls = agent.input_schema
    field_count = len(cls.model_fields)
    assert field_count <= 10, (
        f"{agent.name} input {cls.__name__} has {field_count} fields (cap is 10)"
    )


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_output_has_at_most_5_fields(builder):
    agent = builder()
    cls = agent.output_schema
    field_count = len(cls.model_fields)
    assert field_count <= 5, (
        f"{agent.name} output {cls.__name__} has {field_count} fields (cap is 5)"
    )
