"""Doc AI tests — the confidence gate.

A date extracted from a blurry photo decides which tier a complaint is maintainable
at. So the rule is: trust a field only when Doc AI says it is confident, and ask
about everything else. These tests pin that rule.
"""

import pytest

import documents


def extraction(result, annotations=None, status="completed"):
    return {"result": result, "annotations": annotations or {}, "status": status}


def annotated(**scores):
    return {field: {"confidence": score, "sources": []} for field, score in scores.items()}


def test_high_confidence_fields_become_facts():
    out = documents.to_facts(extraction(
        {"institution": "State Bank", "letter_date": "2026-06-01"},
        annotated(institution=0.97, letter_date=0.91),
    ))
    assert out["facts"] == {"institution": "State Bank", "last_communication_on": "2026-06-01"}
    assert out["confirm"] == []


def test_low_confidence_fields_are_confirmed_not_assumed():
    out = documents.to_facts(extraction(
        {"letter_date": "2026-06-01"},
        annotated(letter_date=0.41),
    ))
    assert out["facts"] == {}
    assert [c["fact"] for c in out["confirm"]] == ["last_communication_on"]
    assert out["confirm"][0]["confidence"] == 0.41


def test_missing_confidence_is_treated_as_uncertain():
    """No score reported is not the same as a high score. Fail towards asking."""
    out = documents.to_facts(extraction({"institution": "State Bank"}, {}))
    assert out["facts"] == {}
    assert out["confirm"][0]["field"] == "institution"


@pytest.mark.parametrize("score,expected_fact", [(0.75, True), (0.74, False)])
def test_threshold_boundary(score, expected_fact):
    out = documents.to_facts(extraction({"institution": "X"}, annotated(institution=score)))
    assert bool(out["facts"]) is expected_fact


def test_empty_values_are_ignored_entirely():
    out = documents.to_facts(extraction(
        {"institution": "", "letter_date": None, "amount_inr": 1200},
        annotated(institution=0.99, letter_date=0.99, amount_inr=0.99),
    ))
    assert out["facts"] == {"amount_inr": 1200}


def test_fields_we_do_not_map_are_reported_not_dropped_silently():
    out = documents.to_facts(extraction(
        {"reference_number": "CMP/2026/991", "is_final_reply": True},
        annotated(reference_number=0.99, is_final_reply=0.99),
    ))
    assert set(out["ignored"]) == {"reference_number", "is_final_reply"}
    assert out["facts"] == {}


def test_letter_date_maps_to_the_institutions_last_communication():
    """The letter's date is when the bank last spoke — that is the date the 90-day
    clock runs from, so mapping it wrongly would move a legal deadline."""
    assert documents.FACT_MAP["letter_date"] == "last_communication_on"


def test_schema_is_valid_for_the_api():
    """Every field needs a type and a non-empty description or the API rejects it."""
    schema = documents.EXTRACT_SCHEMA
    assert schema["type"] == "object"
    assert schema["properties"]
    for name, spec in schema["properties"].items():
        assert spec.get("type"), name
        assert spec.get("description", "").strip(), name


def test_summarise_handles_an_empty_extraction():
    assert "could not read" in documents.summarise(extraction({}))


def test_summarise_reports_what_it_found():
    line = documents.summarise(extraction(
        {"institution": "State Bank", "letter_date": "2026-06-01", "is_final_reply": True}
    ))
    assert "State Bank" in line and "2026-06-01" in line and "final reply" in line


def test_failed_job_yields_no_facts():
    out = documents.to_facts(extraction({}, {}, status="failed"))
    assert out["facts"] == {} and out["confirm"] == []


# --------------------------------------------- the fact sheet must not be clobbered


def test_a_later_message_does_not_erase_what_the_document_read(tmp_path, monkeypatch):
    """The bug this pins: Doc AI read the letter correctly, then the user's next
    message replaced the whole fact sheet with {} and every extracted field vanished.
    From the outside that is indistinguishable from "it cannot read my documents"."""
    import store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.init()

    case_id = store.create_case(owner_token="t")
    store.merge_facts(case_id, {"institution": "State Bank of Maharashtra",
                                "amount_inr": 9840,
                                "last_communication_on": "2026-06-25"})

    # what the intake paths now do on first contact
    store.merge_facts(case_id, {"institution": None})

    facts = store.get_case(case_id)["facts"]
    assert facts["last_communication_on"] == "2026-06-25", "document facts were erased"
    assert facts["amount_inr"] == 9840
    assert facts["institution"] == "State Bank of Maharashtra", "None must not overwrite"


def test_merge_facts_still_updates_a_known_field(tmp_path, monkeypatch):
    import store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t2.db")
    store.init()
    case_id = store.create_case(owner_token="t")

    store.merge_facts(case_id, {"institution": "Old Bank"})
    store.merge_facts(case_id, {"institution": "New Bank"})

    assert store.get_case(case_id)["facts"]["institution"] == "New Bank"
