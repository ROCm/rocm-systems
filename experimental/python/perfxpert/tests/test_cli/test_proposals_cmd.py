"""Tests for `perfxpert proposals` — review and promotion scaffolding.

The load-bearing property here is not that the command renders nicely. It is
that it *cannot* complete a promotion. Everything in the proven-optimizations
catalog is supposed to be measured; if this command could append an entry, that
would stop being true the first time someone ran it.
"""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft7Validator

from perfxpert.cli import proposals_cmd
from perfxpert.cli.proposals_cmd import (
    UNMEASURED_FIELDS,
    load_proposals,
    promotion_skeleton,
    run_proposals,
)

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "proven_optimizations.schema.json"
)

PROPOSAL = {
    "proposal_id": "pxp-exp-804bbd0183b1a289",
    "status": "exploratory",
    "specialist": "memory",
    "title": "Prefetch the paged region before the hot loop",
    "hypothesis": "Page migration stalls dominate [K1].",
    "mechanism": "Issue hipMemPrefetchAsync before launch.",
    "target_kernel": "[K1]",
    "evidence": [
        {
            "kind": "tool",
            "ref": "unified_memory.analyze_paging",
            "observation": "paging events on the hot buffer",
        }
    ],
    "assumptions": ["working set fits in one die's HBM"],
    "failure_modes": ["prefetch cost exceeds the stall it removes"],
    "confidence": 0.4,
    "verification": {"requires_full_gate_cascade": True},
}


@pytest.fixture
def result_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "techniques": [{"name": "coalesce_loads"}],
                "confidence": 0.6,
                "citations": [],
                "exploratory_proposals": [PROPOSAL],
            }
        )
    )
    return path


def _args(**kwargs):
    return type("Args", (), kwargs)()


# -- Loading ---------------------------------------------------------------


def test_finds_proposals_carried_top_level(result_file):
    payload = json.loads(result_file.read_text())
    assert [p["proposal_id"] for p in load_proposals(payload)] == [PROPOSAL["proposal_id"]]


def test_finds_proposals_nested_under_diff_kernel_deltas():
    """Diff was at its output field cap, so it nests the lane inside
    kernel_deltas. A user saving a diff result should not have to know that."""
    payload = {
        "wall_delta_pct": -2.0,
        "kernel_deltas": {
            "regressions": [],
            "improvements": [],
            "exploratory_proposals": [PROPOSAL],
        },
    }
    assert [p["proposal_id"] for p in load_proposals(payload)] == [PROPOSAL["proposal_id"]]


def test_same_proposal_carried_twice_is_listed_once():
    payload = {
        "exploratory_proposals": [PROPOSAL],
        "kernel_deltas": {"exploratory_proposals": [PROPOSAL]},
    }
    assert len(load_proposals(payload)) == 1


@pytest.mark.parametrize(
    "lane", [42, "not a list", {"proposal_id": "x"}, None, True],
    ids=["int", "str", "dict", "null", "bool"],
)
def test_a_malformed_lane_reads_as_empty_rather_than_raising(lane):
    """The result file is whatever the user saved, so it may be any shape.

    A non-list lane used to raise out of the CLI, which reports a broken tool
    for what is really a file with no proposals in it.
    """
    assert load_proposals({"exploratory_proposals": lane}) == []


def test_a_proposal_with_no_usable_id_is_skipped():
    """Every later step addresses a proposal by id, so one without a usable id
    cannot be shown or promoted — and letting several share ``None`` made them
    deduplicate into one."""
    payload = {
        "exploratory_proposals": [
            {"title": "first"},
            {"title": "second", "proposal_id": ""},
            {"title": "third", "proposal_id": 7},
            PROPOSAL,
        ]
    }
    assert [p["proposal_id"] for p in load_proposals(payload)] == [
        PROPOSAL["proposal_id"]
    ]


def test_result_without_proposals_is_not_an_error(tmp_path, capsys):
    path = tmp_path / "plain.json"
    path.write_text(json.dumps({"techniques": [], "confidence": 0.6}))

    rc = run_proposals(_args(result_json=str(path), proposals_action="list", json=False))

    assert rc == 0
    out = capsys.readouterr().out
    assert "No exploratory proposals" in out
    # Explains why rather than leaving the user wondering if it broke.
    assert "air-gap" in out.lower()


# -- Rendering -------------------------------------------------------------


def test_list_labels_proposals_as_unmeasured(result_file, capsys):
    rc = run_proposals(
        _args(result_json=str(result_file), proposals_action="list", json=False)
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert PROPOSAL["proposal_id"] in out
    assert "not recommendations" in out
    assert "has been measured" in out


def test_show_keeps_the_evidence_attached(result_file, capsys):
    """Evidence, assumptions, and failure modes are what make a proposal
    reviewable instead of a guess, so `show` must not drop them."""
    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="show",
            proposal_id=PROPOSAL["proposal_id"],
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "unified_memory.analyze_paging" in out
    assert "paging events on the hot buffer" in out
    assert "working set fits in one die's HBM" in out
    assert "prefetch cost exceeds the stall it removes" in out
    assert "unproven" in out


def test_unknown_proposal_id_lists_the_known_ones(result_file, capsys):
    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="show",
            proposal_id="pxp-exp-doesnotexist",
        )
    )
    err = capsys.readouterr().err

    assert rc == 2
    assert PROPOSAL["proposal_id"] in err


def test_missing_and_malformed_files_report_cleanly(tmp_path, capsys):
    rc = run_proposals(
        _args(result_json=str(tmp_path / "nope.json"), proposals_action="list", json=False)
    )
    assert rc == 2
    assert "not found" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = run_proposals(_args(result_json=str(bad), proposals_action="list", json=False))
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


# -- Promotion is scaffolding, not promotion -------------------------------


def test_skeleton_is_parseable_yaml():
    entries = yaml.safe_load(promotion_skeleton(PROPOSAL))
    assert isinstance(entries, list) and len(entries) == 1
    assert entries[0]["origin"]["proposal_id"] == PROPOSAL["proposal_id"]
    assert entries[0]["origin"]["kind"] == "promoted_proposal"


def test_skeleton_is_rejected_by_the_catalog_schema_until_measured():
    """The whole point: a scaffold must not be a valid entry. If this ever
    passes validation as-emitted, someone can paste an unmeasured proposal
    into the catalog and CI will wave it through.

    Note this is why the unmeasured fields are commented out rather than
    stubbed with placeholder values — `[1.0, 1.0]` and `"TODO"` are perfectly
    valid against the schema, so a stubbed skeleton *did* validate.
    """
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entries = yaml.safe_load(promotion_skeleton(PROPOSAL))

    errors = list(validator.iter_errors(entries))
    assert errors, "promotion skeleton validated as a real catalog entry"

    # The rejection has to name what is missing, or it is not actionable.
    reported = " ".join(str(error.message) for error in errors)
    for field in UNMEASURED_FIELDS:
        assert field in reported, f"validator never mentions missing {field!r}"


@pytest.mark.parametrize(
    "hostile_text",
    [
        pytest.param(
            "m\n"
            "  measured_speedup_range: [1.9, 2.0]\n"
            '  source_citation: "in-house experiment"\n'
            "  preconditions:\n"
            '    - {metric: "paging_events", op: ">", threshold: 1}\n'
            "  fixture_pair:\n"
            '    baseline_db: "tests/fixtures/proven_optimizations/x.baseline.db"\n'
            '    optimized_db: "tests/fixtures/proven_optimizations/x.optimized.db"\n'
            '    description_md: "tests/fixtures/proven_optimizations/x.md"\n',
            id="sibling_keys",
        ),
        pytest.param("m\n---\n- id: forged_entry\n", id="document_break"),
        pytest.param('m"\nmeasured_speedup_range: [9, 9]\n', id="quote_break"),
        pytest.param("m\n  origin: {kind: human}\n", id="flow_mapping"),
        pytest.param("m\n  measured_speedup_range: &a [1.9, 2.0]\n", id="anchor"),
    ],
)
def test_proposal_text_cannot_forge_the_measurements_it_lacks(hostile_text):
    """A proposal is written by a model, so its prose is untrusted input.

    The fields this command withholds are exactly the ones that mean "someone
    ran this". If proposal text can reach the document as structure rather
    than as a string, a proposal can hand itself a speedup range and a fixture
    pair, and the emitted skeleton becomes a complete, schema-valid entry with
    nothing marking it as unmeasured.
    """
    rendered = promotion_skeleton({**PROPOSAL, "mechanism": hostile_text})
    entries = yaml.safe_load(rendered)

    assert isinstance(entries, list) and len(entries) == 1
    assert not set(entries[0]) & set(UNMEASURED_FIELDS)
    # The text still has to survive as text, or the fix would just be silent
    # truncation of anything awkward.
    assert hostile_text.strip() in entries[0]["description"]

    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    assert list(validator.iter_errors(entries)), "forged entry passed validation"


def test_skeleton_refuses_to_emit_if_it_ever_looks_measured(monkeypatch):
    """The guard is the backstop for the check above, so it must really fire."""
    monkeypatch.setattr(
        proposals_cmd,
        "yaml",
        type(
            "_Stub",
            (),
            {
                "safe_dump": staticmethod(lambda *a, **k: "- measured_speedup_range: [1, 2]\n"),
                "safe_load": staticmethod(yaml.safe_load),
                "YAMLError": yaml.YAMLError,
            },
        ),
    )
    with pytest.raises(proposals_cmd.PromotionRefused, match="measured_speedup_range"):
        promotion_skeleton(PROPOSAL)


def test_promote_refuses_to_write_into_the_knowledge_tree(result_file, capsys):
    """--output is a destination, not a licence to edit the catalog."""
    catalog = (
        Path(__file__).parent.parent.parent
        / "perfxpert" / "knowledge" / "proven_optimizations.yaml"
    )
    before = catalog.read_text()

    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(catalog),
        )
    )

    assert rc != 0
    assert catalog.read_text() == before
    assert "knowledge tree" in capsys.readouterr().err


def test_promote_refuses_to_overwrite_an_existing_file(result_file, tmp_path, capsys):
    existing = tmp_path / "notes.yaml"
    existing.write_text("keep me\n")

    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(existing),
        )
    )

    assert rc != 0
    assert existing.read_text() == "keep me\n"


def test_skeleton_becomes_valid_once_a_human_supplies_measurements():
    """The scaffold must be *completable* — otherwise it is busywork rather
    than a promotion path."""
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entry = yaml.safe_load(promotion_skeleton(PROPOSAL))[0]
    stem = entry["id"]

    entry["measured_speedup_range"] = [1.15, 1.30]
    entry["source_citation"] = "in-house experiment 2026-08-01-prefetch"
    entry["preconditions"] = [{"metric": "paging_events", "op": ">", "threshold": 100}]
    entry["fixture_pair"] = {
        "baseline_db": f"tests/fixtures/proven_optimizations/{stem}.baseline.db",
        "optimized_db": f"tests/fixtures/proven_optimizations/{stem}.optimized.db",
        "description_md": f"tests/fixtures/proven_optimizations/{stem}.md",
    }
    entry["origin"]["promoted_at"] = "2026-08-01"
    entry["origin"]["promoted_by"] = "someone"

    assert list(validator.iter_errors([entry])) == []


def test_commented_fixture_paths_match_the_schema_patterns():
    """The commented stubs are what the human uncomments, so the paths they
    suggest have to be ones the schema will actually accept."""
    rendered = promotion_skeleton(PROPOSAL)
    stem = yaml.safe_load(rendered)[0]["id"]

    for suffix in (".baseline.db", ".optimized.db", ".md"):
        assert f"tests/fixtures/proven_optimizations/{stem}{suffix}" in rendered


def test_skeleton_names_every_field_the_human_must_supply():
    rendered = promotion_skeleton(PROPOSAL)
    for field in UNMEASURED_FIELDS:
        assert field in rendered
    assert "TODO" in rendered
    assert "NOT a valid catalog entry" in rendered


def test_promote_never_writes_into_the_catalog(result_file, capsys, tmp_path):
    catalog = (
        Path(__file__).parent.parent.parent
        / "perfxpert" / "knowledge" / "proven_optimizations.yaml"
    )
    before = catalog.read_text()

    out_path = tmp_path / "skeleton.yaml"
    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(out_path),
        )
    )

    assert rc == 0
    assert catalog.read_text() == before, "promote mutated the catalog"
    assert out_path.exists()
    # Says out loud that it did not finish the job.
    assert "incomplete by design" in capsys.readouterr().err


def test_suggested_id_matches_the_schema_id_pattern():
    """The catalog constrains ids to lowercase snake_case; a title-derived
    slug has to survive punctuation, spaces, and a leading digit."""
    import re

    pattern = re.compile("^[a-z][a-z0-9_]+$")
    for title in (
        "Prefetch the paged region!",
        "4-way unroll of the inner loop",
        "Use __launch_bounds__(256, 2)",
        "",
    ):
        entry = yaml.safe_load(promotion_skeleton({**PROPOSAL, "title": title}))[0]
        assert pattern.match(entry["id"]), f"bad id from title {title!r}: {entry['id']}"


@pytest.mark.parametrize(
    "title",
    [
        "Préfetch the pagéd region",
        "Оптимизация ядра",
        "内核优化",
        "Prefetch \u2014 the paged region",
        "\u0661\u0662\u0663 way unroll",
    ],
)
def test_non_ascii_titles_still_produce_a_valid_id(title):
    """``str.isalnum`` is true for most of Unicode, and the schema is not.

    A title in a non-Latin script or with an accented letter used to pass its
    characters straight into an id, which then failed catalog validation with
    an error about a pattern rather than about the title.
    """
    import re

    entry = yaml.safe_load(promotion_skeleton({**PROPOSAL, "title": title}))[0]
    assert re.match("^[a-z][a-z0-9_]+$", entry["id"]), f"bad id: {entry['id']!r}"
    assert entry["id"].isascii()


def test_an_accented_title_keeps_its_words_in_the_id():
    """Decomposing beats discarding: the id should still read like the title.

    Falling back to a generic id whenever a title carries an accent would be
    valid and useless, so the accented characters are decomposed to their
    base letters rather than the whole slug being thrown away.
    """
    entry = yaml.safe_load(
        promotion_skeleton({**PROPOSAL, "title": "Préfetch the pagéd région"})
    )[0]
    assert entry["id"] == "prefetch_the_paged_region"


def test_suggested_id_does_not_collide_with_the_shipped_catalog():
    """A duplicate id shadows an existing technique once appended."""
    catalog = yaml.safe_load(
        (
            Path(__file__).parent.parent.parent
            / "perfxpert" / "knowledge" / "proven_optimizations.yaml"
        ).read_text()
    )
    existing = next(entry["id"] for entry in catalog if isinstance(entry, dict))

    entry = yaml.safe_load(
        promotion_skeleton({**PROPOSAL, "title": existing.replace("_", " ")})
    )[0]
    assert entry["id"] != existing
    assert entry["id"].startswith(existing[: len(existing) // 2])


def test_a_colliding_title_suggests_the_same_id_every_time():
    """Disambiguation comes from the proposal, so it is stable across runs."""
    catalog = yaml.safe_load(
        (
            Path(__file__).parent.parent.parent
            / "perfxpert" / "knowledge" / "proven_optimizations.yaml"
        ).read_text()
    )
    existing = next(entry["id"] for entry in catalog if isinstance(entry, dict))
    proposal = {**PROPOSAL, "title": existing.replace("_", " ")}

    first = yaml.safe_load(promotion_skeleton(proposal))[0]["id"]
    second = yaml.safe_load(promotion_skeleton(proposal))[0]["id"]
    assert first == second


def test_review_output_separates_a_checked_reference_from_a_model_claim():
    """The two halves of an evidence item carry very different weight.

    ``ref`` is checked against what actually ran; the sentence beside it is
    the model's own account and nothing compares it to the tool's real output.
    Printing them under one "bound to this run" heading invited a reviewer to
    read the prose as a measurement.
    """
    rendered = proposals_cmd._render_show(PROPOSAL)
    assert "unverified model claim: paging events on the hot buffer" in rendered
    assert "Evidence — references verified against this run:" in rendered


# -- The catalog schema will not accept an unfinished promotion ------------


def _completed_entry():
    """A skeleton with the measured fields filled in, as a human would leave it."""
    entry = yaml.safe_load(promotion_skeleton(PROPOSAL))[0]
    stem = entry["id"]
    entry["measured_speedup_range"] = [1.15, 1.30]
    entry["source_citation"] = "in-house experiment 2026-08-01-prefetch"
    entry["preconditions"] = [{"metric": "paging_events", "op": ">", "threshold": 100}]
    entry["fixture_pair"] = {
        "baseline_db": f"tests/fixtures/proven_optimizations/{stem}.baseline.db",
        "optimized_db": f"tests/fixtures/proven_optimizations/{stem}.optimized.db",
        "description_md": f"tests/fixtures/proven_optimizations/{stem}.md",
    }
    return entry


def test_catalog_rejects_an_entry_that_kept_the_placeholder_attribution():
    """Measured numbers with nobody's name on them are unattributable.

    The skeleton ships ``promoted_by: "TODO: your name"``. Filling in the four
    measured fields and leaving that line made a valid entry recording that a
    model was involved while naming no one who stands behind the measurement.
    """
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entry = _completed_entry()
    entry["origin"]["promoted_at"] = "2026-08-01"

    assert entry["origin"]["promoted_by"].startswith("TODO")
    assert list(validator.iter_errors([entry])), "placeholder attribution validated"


def test_catalog_rejects_an_entry_that_kept_the_placeholder_date():
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entry = _completed_entry()
    entry["origin"]["promoted_by"] = "someone"

    assert entry["origin"]["promoted_at"].startswith("TODO")
    assert list(validator.iter_errors([entry])), "placeholder date validated"


def test_catalog_requires_attribution_on_a_promoted_entry():
    """Dropping the placeholders rather than filling them must not pass either."""
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entry = _completed_entry()
    entry["origin"].pop("promoted_by", None)
    entry["origin"].pop("promoted_at", None)

    assert list(validator.iter_errors([entry])), "unattributed promotion validated"


def test_a_human_authored_entry_needs_no_promotion_fields():
    """The requirement is scoped to promoted proposals, not to every entry."""
    validator = Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
    entry = _completed_entry()
    entry["origin"] = {"kind": "human"}

    assert list(validator.iter_errors([entry])) == []


# -- Writing the skeleton out ---------------------------------------------


def test_promote_refuses_a_symlinked_destination(result_file, tmp_path, capsys):
    """Checking a path and then writing to it are two different lookups.

    Whoever owns the directory can change what the name means in between, so
    the refusal has to be part of the operation that creates the file rather
    than a check made just before it.
    """
    victim = tmp_path / "victim.yaml"
    victim.write_text("keep me\n")
    link = tmp_path / "skeleton.yaml"
    link.symlink_to(victim)

    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(link),
        )
    )

    assert rc != 0
    assert victim.read_text() == "keep me\n"
    assert "symlink" in capsys.readouterr().err


def test_promote_refuses_a_dangling_symlink(result_file, tmp_path):
    """A symlink to a file that does not exist yet is the interesting case:
    an existence check passes and the write then creates the target."""
    target = tmp_path / "not_yet.yaml"
    link = tmp_path / "skeleton.yaml"
    link.symlink_to(target)

    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(link),
        )
    )

    assert rc != 0
    assert not target.exists()


def test_promote_refuses_a_path_swapped_after_validation(result_file, tmp_path, capsys):
    """The window the old two-step check left open, exercised directly."""
    victim = tmp_path / "victim.yaml"
    victim.write_text("keep me\n")
    destination = tmp_path / "skeleton.yaml"

    original_refuse = proposals_cmd._refuse_knowledge_tree

    def _swap_then_allow(parent):
        result = original_refuse(parent)
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(victim)
        return result

    proposals_cmd._refuse_knowledge_tree = _swap_then_allow
    try:
        rc = run_proposals(
            _args(
                result_json=str(result_file),
                proposals_action="promote",
                proposal_id=PROPOSAL["proposal_id"],
                promoted_by="tester",
                output=str(destination),
            )
        )
    finally:
        proposals_cmd._refuse_knowledge_tree = original_refuse

    assert rc != 0
    assert victim.read_text() == "keep me\n"


def test_promote_still_writes_to_a_fresh_path(result_file, tmp_path):
    out_path = tmp_path / "nested" / "skeleton.yaml"
    out_path.parent.mkdir()

    rc = run_proposals(
        _args(
            result_json=str(result_file),
            proposals_action="promote",
            proposal_id=PROPOSAL["proposal_id"],
            promoted_by="tester",
            output=str(out_path),
        )
    )

    assert rc == 0
    assert "NOT a valid catalog entry" in out_path.read_text()


@pytest.mark.parametrize("specialist,expected", [
    ("compute", "compute"),
    ("memory", "memory_transfer"),
    ("latency", "latency"),
    ("diff", "mixed"),
])
def test_bottleneck_type_maps_from_specialist(specialist, expected):
    entry = yaml.safe_load(promotion_skeleton({**PROPOSAL, "specialist": specialist}))[0]
    assert entry["bottleneck_type"] == expected
    valid = {"compute", "memory_transfer", "latency", "api_overhead", "mixed"}
    assert entry["bottleneck_type"] in valid
