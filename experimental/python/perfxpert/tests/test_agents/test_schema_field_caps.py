"""CI guardrail: input ≤10 fields, output ≤5 fields per agent (spec §2)."""

import pytest

from perfxpert.agents import schemas


INPUT_SCHEMAS = [
    "RootInput",
    "AnalysisInput",
    "RecommendationInput",
    "CorrectnessInput",
    "ComputeSpecialistInput",
    "MemorySpecialistInput",
    "LatencySpecialistInput",
]

OUTPUT_SCHEMAS = [
    "RootOutput",
    "AnalysisOutput",
    "RecommendationOutput",
    "CorrectnessOutput",
    "ComputeSpecialistOutput",
    "MemorySpecialistOutput",
    "LatencySpecialistOutput",
]


@pytest.mark.parametrize("name", INPUT_SCHEMAS)
def test_input_has_at_most_10_fields(name):
    cls = getattr(schemas, name)
    field_count = len(cls.model_fields)
    assert field_count <= 10, f"{name} has {field_count} fields (cap is 10)"


@pytest.mark.parametrize("name", OUTPUT_SCHEMAS)
def test_output_has_at_most_5_fields(name):
    cls = getattr(schemas, name)
    field_count = len(cls.model_fields)
    assert field_count <= 5, f"{name} has {field_count} fields (cap is 5)"
