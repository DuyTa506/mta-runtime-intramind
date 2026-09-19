"""Verify a release evidence bundle without claiming independent attestation.

Paths resolve relative to the manifest. Every evidence entry has ``status``,
``format``, ``report_path`` and ``report_sha256``. Test gates use ``junit_xml``
and a nonempty ``required_testcases`` list of ``classname::name`` identities.
Migrated inventory items reference those cases via
``evidence={"gate": "feature_regression", "testcases": [...]}``.

Operational gates use ``qualification_json`` reports containing schema_version=1,
gate, status, production_capacity_profile, policy_sha256 and numeric measurements.
The manifest's qualification_policy is another verified file reference. Its JSON
contains schema_version=1, source, production_capacity_profile and gates. Each
gate maps metric names to {unit, minimum?, maximum?}; at least one bound is required.
The capacity profile is {profile_id, fingerprint_sha256} in all three documents.

The gate checks bytes, executed cases, measurements and declared policy bounds.
It cannot prove test quality, authenticity of measurements, deployment identity,
or that a declared fingerprint describes the actual GPU. Those require review and
trusted qualification runners. No existing report is silently grandfathered in.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree

REQUIRED_EVIDENCE = (
    "ledger_fault_injection",
    "workflow_replay",
    "feature_regression",
    "api_security",
    "minio_contract",
    "gpu_capacity",
    "overload",
    "soak_72h",
    "offhost_restore",
    "canary",
    "celery_drained",
    "seven_days_full_routing",
)

TEST_GATES = frozenset(REQUIRED_EVIDENCE[:5])
INVENTORY_IDS = frozenset(
    f"{prefix}{number:02}"
    for prefix, count in (("A", 12), ("B", 8), ("D", 10), ("H", 8), ("O", 3), ("X", 2))
    for number in range(1, count + 1)
)
# Accepted duration/recovery targets and correctness conditions, not GPU/SLO guesses.
REQUIRED_MEASUREMENTS = {
    "gpu_capacity": {"samples": (1, None), "oom_count": (0, 0), "memory_headroom_bytes": (1, None)},
    "overload": {
        "samples_at_2x": (1, None),
        "samples_at_5x": (1, None),
        "lost_accepted_runs": (0, 0),
        "unbounded_queue_events": (0, 0),
    },
    "soak_72h": {
        "duration_seconds": (72 * 3600, None),
        "workflows_completed": (1, None),
        "lost_accepted_runs": (0, 0),
        "ledger_invariant_violations": (0, 0),
        "oom_count": (0, 0),
    },
    "offhost_restore": {
        "rpo_seconds": (0, 900),
        "rto_seconds": (0, 14400),
        "offhost_restore_completed": (1, 1),
        "restored_artifact_checks": (1, None),
        "broken_artifact_refs": (0, 0),
    },
    "canary": {
        "runs_at_5_percent": (1, None),
        "runs_at_25_percent": (1, None),
        "runs_at_50_percent": (1, None),
        "runs_at_100_percent": (1, None),
    },
    "celery_drained": {
        "queued_business_tasks": (0, 0),
        "active_business_tasks": (0, 0),
        "reserved_business_tasks": (0, 0),
    },
    "seven_days_full_routing": {
        "duration_seconds": (7 * 24 * 3600, None),
        "minimum_routing_percent": (100, 100),
        "workflows_completed": (1, None),
    },
}
MAX_REPORT_BYTES = 16 * 1024 * 1024


class InvalidEvidence(ValueError):
    """An evidence artifact cannot qualify a release."""


def _read_report(reference: object, base_dir: Path) -> bytes:
    if not isinstance(reference, dict):
        raise InvalidEvidence("report reference required")
    name, digest = reference.get("report_path"), reference.get("report_sha256")
    if not isinstance(name, str) or not name:
        raise InvalidEvidence("report_path required")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise InvalidEvidence("valid report_sha256 required")
    try:
        with (base_dir / name).open("rb") as report:
            content = report.read(MAX_REPORT_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise InvalidEvidence("report cannot be read") from exc
    if not content or len(content) > MAX_REPORT_BYTES:
        raise InvalidEvidence("report must contain 1 byte to 16 MiB")
    if hashlib.sha256(content).hexdigest() != digest:
        raise InvalidEvidence("report SHA256 mismatch")
    return content


def _json_report(content: bytes) -> dict:
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidEvidence("invalid JSON report") from exc
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise InvalidEvidence("JSON object with schema_version=1 required")
    return data


def _case_names(value: object) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise InvalidEvidence("nonempty testcase list required")
    if len(set(value)) != len(value):
        raise InvalidEvidence("testcase identities must be distinct")
    return set(value)


def _junit_cases(content: bytes, required: object) -> set[str]:
    try:
        required_cases = _case_names(required)
    except InvalidEvidence as exc:
        raise InvalidEvidence(f"required_testcases: {exc}") from exc
    try:
        xml = content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise InvalidEvidence("JUnit report must use UTF-8") from exc
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise InvalidEvidence("JUnit declarations are not supported")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise InvalidEvidence("invalid JUnit XML") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise InvalidEvidence("JUnit testsuite/testsuites root required")
    cases = list(root.iter("testcase"))
    if not cases:
        raise InvalidEvidence("JUnit contains no executed tests")
    if any(element.tag in {"failure", "error", "skipped"} for element in root.iter()):
        raise InvalidEvidence("JUnit contains failures/errors/skips")
    identities = set()
    for case in cases:
        if (
            not case.get("name")
            or case.get("status", "run") not in {"run", "passed", "success", "completed"}
            or case.get("result", "completed") != "completed"
        ):
            raise InvalidEvidence(
                "every JUnit testcase must pass; failures/errors/skips block release"
            )
        identities.add(f"{case.get('classname', '')}::{case.get('name')}")
    if len(identities) != len(cases):
        raise InvalidEvidence("duplicate JUnit testcase identities")
    for suite in root.iter():
        if suite.tag not in {"testsuite", "testsuites"}:
            continue
        totals = {
            "tests": len(list(suite.iter("testcase"))),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "disabled": 0,
        }
        for name, expected in totals.items():
            if name in suite.attrib:
                try:
                    actual = int(suite.attrib[name])
                except ValueError as exc:
                    raise InvalidEvidence("invalid JUnit counters") from exc
                if actual != expected:
                    raise InvalidEvidence("JUnit counters disagree with testcase outcomes")
    if not required_cases <= identities:
        raise InvalidEvidence("required testcase coverage is missing")
    return required_cases


def _finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_profile(profile: object) -> bool:
    return (
        isinstance(profile, dict)
        and isinstance(profile.get("profile_id"), str)
        and bool(profile["profile_id"])
        and isinstance(profile.get("fingerprint_sha256"), str)
        and bool(re.fullmatch(r"[a-f0-9]{64}", profile["fingerprint_sha256"]))
    )


def _qualify(key: str, content: bytes, policy: dict, policy_hash: str, profile: dict) -> None:
    report = _json_report(content)
    if report.get("gate") != key or report.get("status") != "passed":
        raise InvalidEvidence("report gate/status mismatch")
    if report.get("production_capacity_profile") != profile:
        raise InvalidEvidence("capacity profile mismatch")
    if report.get("policy_sha256") != policy_hash:
        raise InvalidEvidence("qualification policy hash mismatch")
    limits = policy.get("gates", {}).get(key)
    measured = report.get("measurements")
    if not isinstance(limits, dict) or not isinstance(measured, dict):
        raise InvalidEvidence("policy thresholds and measured values required")
    for metric, (minimum, maximum) in REQUIRED_MEASUREMENTS[key].items():
        value = measured.get(metric)
        if metric not in limits or not _finite_number(value):
            raise InvalidEvidence(f"{metric}: measurement and policy threshold required")
        if minimum is not None and value < minimum or maximum is not None and value > maximum:
            raise InvalidEvidence(f"{metric}: accepted release requirement not met")
    for metric, bound in limits.items():
        value = measured.get(metric)
        if not _finite_number(value) or not isinstance(bound, dict) or not bound.get("unit"):
            raise InvalidEvidence(f"{metric}: numeric measurement and policy unit required")
        checks = {name: bound[name] for name in ("minimum", "maximum") if name in bound}
        if not checks or any(not _finite_number(limit) for limit in checks.values()):
            raise InvalidEvidence(f"{metric}: finite policy bounds required")
        if "minimum" in checks and "maximum" in checks and checks["minimum"] > checks["maximum"]:
            raise InvalidEvidence(f"{metric}: contradictory policy bounds")
        if value < checks.get("minimum", -math.inf) or value > checks.get("maximum", math.inf):
            raise InvalidEvidence(f"{metric}: measured value exceeds qualified policy bounds")


def violations(manifest: object, *, base_dir: Path | None = None) -> list[str]:
    """Return every blocked gate; relative references use the manifest directory."""
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    base_dir = base_dir or Path.cwd()
    failures = []
    tasks = manifest.get("inventory", [])
    if not isinstance(tasks, list):
        tasks = []
    ids = [item.get("id") for item in tasks if isinstance(item, dict)]
    if (
        len(tasks) != 43
        or any(not isinstance(item, str) for item in ids)
        or set(ids) != INVENTORY_IDS
    ):
        failures.append("all 43 distinct inventory items must be accounted for")
    profile = manifest.get("production_capacity_profile")
    if not _valid_profile(profile):
        failures.append("measured production capacity profile with fingerprint required")
    policy = None
    policy_ref = manifest.get("qualification_policy")
    try:
        policy = _json_report(_read_report(policy_ref, base_dir))
        if not isinstance(policy.get("source"), str) or not policy["source"]:
            raise InvalidEvidence("policy source required")
        if not _valid_profile(profile) or policy.get("production_capacity_profile") != profile:
            raise InvalidEvidence("capacity profile mismatch")
        if not isinstance(policy.get("gates"), dict):
            raise InvalidEvidence("gate thresholds required")
    except InvalidEvidence as exc:
        failures.append(f"qualification_policy: {exc}")
        policy = None
    evidence_entries = manifest.get("evidence", {})
    if not isinstance(evidence_entries, dict):
        evidence_entries = {}
    verified_cases = {}
    for key in REQUIRED_EVIDENCE:
        try:
            evidence = evidence_entries.get(key)
            if not isinstance(evidence, dict) or evidence.get("status") != "passed":
                raise InvalidEvidence("qualified report required")
            content = _read_report(evidence, base_dir)
            expected_format = "junit_xml" if key in TEST_GATES else "qualification_json"
            if evidence.get("format") != expected_format:
                raise InvalidEvidence(f"format must be {expected_format}")
            if key in TEST_GATES:
                verified_cases[key] = _junit_cases(content, evidence.get("required_testcases"))
            elif policy is None:
                raise InvalidEvidence("verified qualification policy required")
            else:
                _qualify(key, content, policy, policy_ref["report_sha256"], profile)
        except InvalidEvidence as exc:
            failures.append(f"{key}: {exc}")
    for item in tasks:
        if not isinstance(item, dict):
            continue
        if item.get("disposition") not in ("migrated", "retained_native", "removed"):
            failures.append(f"{item.get('id')}: migration incomplete")
        if item.get("disposition") == "migrated":
            reference = item.get("evidence")
            try:
                if not isinstance(reference, dict) or reference.get("gate") != "feature_regression":
                    raise InvalidEvidence("verified feature_regression testcases required")
                if not _case_names(reference.get("testcases")) <= verified_cases.get(
                    "feature_regression", set()
                ):
                    raise InvalidEvidence("testcases were not qualified")
            except InvalidEvidence as exc:
                failures.append(f"{item.get('id')}: migration evidence: {exc}")
    if manifest.get("artifact_backend") != "minio":
        failures.append("this release must retain MinIO")
    return failures


def main() -> None:
    """Exit unsuccessfully until the manifest and report bundle qualify."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        failures = violations(json.loads(args.manifest.read_text()), base_dir=args.manifest.parent)
    except (OSError, ValueError) as exc:
        failures = [f"manifest cannot be read: {type(exc).__name__}"]
    for failure in failures:
        print(f"BLOCKED: {failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
