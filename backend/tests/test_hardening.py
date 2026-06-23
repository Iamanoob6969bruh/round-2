"""
Unit tests for the hackathon hardening work:
  • tamper-evident evidence integrity (content hash + hash chain)
  • honest / robust plate OCR correction (BH-series, non-mangling)
  • concurrency-safe SQLite ViolationDatabase (chain, search, migration)

Runnable two ways:
    pytest backend/tests/test_hardening.py
    python  backend/tests/test_hardening.py     (no pytest required)

Deliberately avoids loading YOLO / EasyOCR models so it runs fast and offline.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make `src` importable when run as a plain script.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.evidence import integrity
from src.evidence.generator import EvidencePackage
from src.plate_recognition.flagged_plate_reader import FlaggedVehiclePlateReader
from src.analytics.reporting import ViolationDatabase, AnalyticsEngine


# ──────────────────────────────────────────────────────────────────────────
# 1. Integrity: content hashing + hash chain
# ──────────────────────────────────────────────────────────────────────────
def test_hash_bytes_deterministic_and_sensitive():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 1  # flip a single pixel
    assert integrity.hash_bytes(a) == integrity.hash_bytes(a.copy())
    assert integrity.hash_bytes(a) != integrity.hash_bytes(b)
    assert len(integrity.hash_bytes(a)) == 64  # full sha256 hex


def test_content_hash_detects_metadata_tamper():
    img = np.full((8, 8, 3), 127, dtype=np.uint8)
    meta = {"violation_type": "helmet_violation", "confidence": 0.9}
    h1 = integrity.compute_content_hash(meta, original=img)
    meta2 = dict(meta, confidence=0.1)  # altered field
    h2 = integrity.compute_content_hash(meta2, original=img)
    assert h1 != h2


def test_seal_and_verify_record_roundtrip():
    img = np.full((8, 8, 3), 50, dtype=np.uint8)
    meta = {"violation_type": "triple_riding", "plate": "MH12AB1234"}
    seal = integrity.seal_record(meta, original=img, prev_hash=integrity.GENESIS_HASH)
    ok = integrity.verify_record(seal, meta, original=img)
    assert ok["valid"] is True
    # Tamper the image after sealing -> content check fails
    tampered = img.copy()
    tampered[0, 0, 0] = 200
    bad = integrity.verify_record(seal, meta, original=tampered)
    assert bad["valid"] is False
    assert bad["checks"]["content_hash"] is False


def test_chain_detects_tamper_and_reorder():
    seals = []
    prev = integrity.GENESIS_HASH
    for i in range(4):
        s = integrity.seal_record({"i": i}, prev_hash=prev)
        seals.append(s)
        prev = s["record_hash"]
    assert integrity.verify_chain(seals)["valid"] is True

    # Tamper a middle record's content hash -> chain breaks at that index
    broken = [dict(s) for s in seals]
    broken[2]["content_hash"] = "deadbeef" * 8
    res = integrity.verify_chain(broken)
    assert res["valid"] is False
    assert res["broken_at"] == 2

    # Reorder records -> linkage breaks
    reordered = [seals[0], seals[2], seals[1], seals[3]]
    assert integrity.verify_chain(reordered)["valid"] is False


# ──────────────────────────────────────────────────────────────────────────
# 2. Plate OCR correction: honest + robust
# ──────────────────────────────────────────────────────────────────────────
def _reader():
    return FlaggedVehiclePlateReader()


def test_standard_plate_correction():
    r = _reader()
    # O->0 / I->1 / B->8 confusions in the numeric district + tail
    assert r._correct_plate_ocr("MH12AB1234") == "MH12AB1234"
    assert r._is_valid_plate("MH12AB1234")


def test_bh_series_supported_and_not_mangled():
    r = _reader()
    out = r._correct_plate_ocr("22BH1234AA")
    assert out == "22BH1234AA", out
    assert r._is_valid_plate("22BH1234AA")
    assert r._extract_plate("garbage22BH1234AAxx") == "22BH1234AA"


def test_non_standard_plate_not_destroyed():
    r = _reader()
    # A temporary / non-standard token must pass through unchanged (no forced
    # template). Previously the rigid corrector would rewrite letters<->digits.
    weird = "TEMP12345"
    assert r._correct_plate_ocr(weird) == "TEMP12345"
    # CD (diplomatic-style) token left intact rather than coerced.
    assert r._correct_plate_ocr("CD1234") == "CD1234"


def test_clean_is_non_destructive():
    r = _reader()
    # _clean only normalises; it must NOT swap characters.
    assert r._clean("mh 12 ab 1234") == "MH 12 AB 1234"
    assert r._clean("22-bh-1234-aa") == "22BH1234AA" or r._clean("22-bh-1234-aa") == "22 BH 1234 AA"


# ──────────────────────────────────────────────────────────────────────────
# 3. SQLite ViolationDatabase: chain, search, persistence, migration
# ──────────────────────────────────────────────────────────────────────────
def _pkg(vtype, plate, conf=0.8, sev="high"):
    img = np.full((6, 6, 3), 100, dtype=np.uint8)
    p = EvidencePackage(
        violation_id=f"VIO-{plate}", timestamp="2026-06-20T10:00:00",
        violation_type=vtype, violation_description="d", confidence=conf,
        severity=sev, vehicle_plate=plate, plate_confidence=0.7,
        bbox=(1, 2, 3, 4), image_hash=integrity.hash_bytes(img),
    )
    p.annotated_image = img
    p.content_hash = integrity.compute_content_hash(p.core_metadata(), original=img, annotated=img)
    return p


def test_sqlite_add_search_and_chain():
    with tempfile.TemporaryDirectory() as d:
        db = ViolationDatabase(db_path=str(Path(d) / "violations_db.json"))
        db.add_records([_pkg("helmet_violation", "MH12AB1234"),
                        _pkg("triple_riding", "KA05CD6789")])
        db.add_record(_pkg("helmet_violation", "MH12AB1234", conf=0.9))

        assert len(db.records) == 3
        # search by plate
        assert len(db.search(plate="MH12AB1234")) == 2
        assert len(db.search(violation_type="triple_riding")) == 1
        # analytics still works off the same interface
        stats = AnalyticsEngine(db).summary_stats()
        assert stats["total_violations"] == 3
        # the hash chain verifies end-to-end
        assert db.verify_chain()["valid"] is True
        # a .sqlite file was actually created (not the legacy .json)
        assert (Path(d) / "violations_db.sqlite").exists()


def test_sqlite_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "violations_db.json")
        db1 = ViolationDatabase(db_path=path)
        db1.add_record(_pkg("helmet_violation", "TN09XY4321"))
        db2 = ViolationDatabase(db_path=path)  # reopen
        assert len(db2.records) == 1
        assert db2.records[0]["vehicle_plate"] == "TN09XY4321"
        assert db2.verify_chain()["valid"] is True


def test_legacy_json_migration():
    with tempfile.TemporaryDirectory() as d:
        json_path = Path(d) / "violations_db.json"
        legacy = [{
            "violation_id": "OLD-1", "timestamp": "2026-01-01T00:00:00",
            "violation_type": "helmet_violation", "confidence": 0.5,
            "severity": "high", "vehicle_plate": "DL01AA1111",
        }]
        json_path.write_text(json.dumps(legacy))
        db = ViolationDatabase(db_path=str(json_path))
        assert len(db.records) == 1
        assert db.records[0]["violation_id"] == "OLD-1"
        # original json renamed so it is not re-imported
        assert json_path.with_suffix(".json.imported").exists()
        assert db.verify_chain()["valid"] is True


# ──────────────────────────────────────────────────────────────────────────
def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
