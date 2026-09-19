"""Release qualification must be supported by report bytes, not green labels."""

import hashlib
import json
from pathlib import Path

import pytest

from intramind_runtime.release_gate import (
    INVENTORY_IDS,
    REQUIRED_MEASUREMENTS,
    TEST_GATES,
    main,
    violations,
)


def _reference(path: Path, content: str) -> dict:
    path.write_text(content)
    return {"report_path": path.name, "report_sha256": hashlib.sha256(content.encode()).hexdigest()}


def _junit(tmp_path: Path, content: str | None = None) -> dict:
    content = content or (
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="feature" name="resume" /></testsuite></testsuites>'
    )
    return {
        **_reference(tmp_path / "tests.xml", content),
        "status": "passed",
        "format": "junit_xml",
        "required_testcases": ["feature::resume"],
    }


@pytest.fixture
def bundle(tmp_path: Path) -> dict:
    """Synthetic reports for validator tests; these never qualify a deployment."""
    profile = {"profile_id": "test-only", "fingerprint_sha256": "a" * 64}
    measurements = {
        gate: {metric: minimum for metric, (minimum, _) in metrics.items()}
        for gate, metrics in REQUIRED_MEASUREMENTS.items()
    }
    measurements["gpu_capacity"]["memory_headroom_bytes"] = 1024
    policy = {
        "schema_version": 1,
        "source": "test fixture, not release evidence",
        "production_capacity_profile": profile,
        "gates": {
            gate: {
                metric: {"unit": "fixture-unit", "minimum": value}
                for metric, value in values.items()
            }
            for gate, values in measurements.items()
        },
    }
    policy["gates"]["gpu_capacity"]["memory_headroom_bytes"]["minimum"] = 512
    policy_ref = _reference(tmp_path / "policy.json", json.dumps(policy))
    evidence = {gate: _junit(tmp_path) for gate in TEST_GATES}
    for gate, values in measurements.items():
        report = {
            "schema_version": 1,
            "gate": gate,
            "status": "passed",
            "production_capacity_profile": profile,
            "policy_sha256": policy_ref["report_sha256"],
            "measurements": values,
        }
        evidence[gate] = {
            **_reference(tmp_path / f"{gate}.json", json.dumps(report)),
            "status": "passed",
            "format": "qualification_json",
        }
    inventory = [{"id": key, "disposition": "retained_native"} for key in sorted(INVENTORY_IDS)]
    inventory[0].update(
        disposition="migrated",
        evidence={"gate": "feature_regression", "testcases": ["feature::resume"]},
    )
    return {
        "artifact_backend": "minio",
        "production_capacity_profile": profile,
        "qualification_policy": policy_ref,
        "evidence": evidence,
        "inventory": inventory,
    }


def test_a_green_unit_suite_cannot_authorize_celery_removal():
    result = violations({"inventory": [], "artifact_backend": "minio", "evidence": {}})
    assert any("soak_72h" in message for message in result)
    assert any("offhost_restore" in message for message in result)
    assert any("celery_drained" in message for message in result)
    assert any("43" in message for message in result)


def test_complete_synthetic_evidence_bundle_passes(tmp_path: Path, bundle: dict) -> None:
    assert violations(bundle, base_dir=tmp_path) == []


@pytest.mark.parametrize("tamper", [True, False])
def test_missing_or_modified_report_blocks_green_metadata(
    tmp_path: Path, bundle: dict, tamper: bool
) -> None:
    path = tmp_path / "tests.xml"
    if tamper:
        path.write_text("tampered")
    else:
        path.unlink()
    failures = violations(bundle, base_dir=tmp_path)
    expected = "SHA256 mismatch" if tamper else "cannot be read"
    assert any(f"feature_regression: report {expected}" in failure for failure in failures)


@pytest.mark.parametrize(
    "content",
    [
        '<testsuite tests="0" />',
        '<testsuite><testcase classname="feature" name="resume"><failure /></testcase></testsuite>',
        '<testsuite><testcase classname="feature" name="resume"><error /></testcase></testsuite>',
        '<testsuite><testcase classname="feature" name="resume"><skipped /></testcase></testsuite>',
        '<testsuite tests="2"><testcase classname="feature" name="resume" /></testsuite>',
        '<testsuite><testcase classname="feature" name="resume" status="notrun" /></testsuite>',
        '<testsuite><testcase classname="feature" name="resume" /><error /></testsuite>',
        '<testsuite><testcase classname="feature" name="not_requested" /></testsuite>',
        '<testsuite><testcase name="same" /><testcase name="same" /></testsuite>',
        '<!DOCTYPE testsuite [<!ENTITY a "x">]><testsuite />',
        '{"status": "passed"}',
    ],
)
def test_failed_skipped_empty_or_inconsistent_junit_blocks_release(
    tmp_path: Path, bundle: dict, content: str
) -> None:
    bundle["evidence"]["feature_regression"] = _junit(tmp_path, content)
    assert any(
        error.startswith("feature_regression:") for error in violations(bundle, base_dir=tmp_path)
    )


def test_required_testcase_coverage_cannot_be_omitted(tmp_path: Path, bundle: dict) -> None:
    bundle["evidence"]["feature_regression"].pop("required_testcases")
    assert any("required_testcases" in error for error in violations(bundle, base_dir=tmp_path))


@pytest.mark.parametrize(
    ("gate", "metric", "value"),
    [
        ("gpu_capacity", "samples", 0),
        ("gpu_capacity", "oom_count", 1),
        ("gpu_capacity", "memory_headroom_bytes", 256),
        ("overload", "samples_at_5x", 0),
        ("soak_72h", "duration_seconds", 71 * 3600),
        ("offhost_restore", "rpo_seconds", 901),
        ("offhost_restore", "rto_seconds", 14401),
        ("offhost_restore", "offhost_restore_completed", 0),
        ("celery_drained", "active_business_tasks", 1),
        ("seven_days_full_routing", "duration_seconds", 6 * 24 * 3600),
        ("seven_days_full_routing", "minimum_routing_percent", 99),
        ("gpu_capacity", "samples", float("nan")),
    ],
)
def test_green_qualification_checks_measured_results_and_policy(
    tmp_path: Path, bundle: dict, gate: str, metric: str, value: float
) -> None:
    reference = bundle["evidence"][gate]
    path = tmp_path / reference["report_path"]
    report = json.loads(path.read_text())
    report["measurements"][metric] = value
    reference.update(_reference(path, json.dumps(report)))
    assert any(error.startswith(f"{gate}:") for error in violations(bundle, base_dir=tmp_path))


def test_policy_hash_is_verified_and_not_just_copied(tmp_path: Path, bundle: dict) -> None:
    (tmp_path / "policy.json").write_text('{"source": "tampered"}')
    assert any(
        "qualification_policy: report SHA256 mismatch" in error
        for error in violations(bundle, base_dir=tmp_path)
    )


def test_wrong_gpu_fingerprint_does_not_qualify(tmp_path: Path, bundle: dict) -> None:
    reference = bundle["evidence"]["gpu_capacity"]
    path = tmp_path / reference["report_path"]
    report = json.loads(path.read_text())
    report["production_capacity_profile"]["fingerprint_sha256"] = "b" * 64
    reference.update(_reference(path, json.dumps(report)))
    assert any(
        "gpu_capacity: capacity profile mismatch" in error
        for error in violations(bundle, base_dir=tmp_path)
    )


def test_migration_claim_requires_a_verified_feature_case(tmp_path: Path, bundle: dict) -> None:
    bundle["inventory"][0]["evidence"]["testcases"] = ["feature::not_executed"]
    assert any(
        "A01: migration evidence" in error for error in violations(bundle, base_dir=tmp_path)
    )


def test_cli_resolves_reports_relative_to_manifest(
    tmp_path: Path, bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(bundle))
    unrelated = tmp_path / "other-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr("sys.argv", ["release_gate", str(manifest_path)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0


def test_audited_inventory_cannot_be_replaced_by_arbitrary_ids(
    tmp_path: Path, bundle: dict
) -> None:
    bundle["inventory"][0]["id"] = "unrelated"
    assert any("43" in error for error in violations(bundle, base_dir=tmp_path))


@pytest.mark.parametrize("manifest", [[], {"inventory": None}, {"evidence": ["invalid"]}])
def test_bad_manifest_is_blocked_without_crashing(tmp_path: Path, manifest: object) -> None:
    assert violations(manifest, base_dir=tmp_path)
