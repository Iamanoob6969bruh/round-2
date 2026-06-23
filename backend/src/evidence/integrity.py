"""
Tamper-Evident Evidence Integrity.

Every violation record is *sealed* into a cryptographic chain so that any later
modification — to the image, the metadata, the plate, the timestamp, or the
ordering of records — is detectable.

Two layers of protection:

  1. CONTENT HASH  — a full SHA-256 over the original frame bytes, the annotated
     evidence bytes, and the canonicalised metadata. Changing a single pixel or
     a single field changes this hash.

  2. HASH CHAIN    — each record stores the `record_hash` of the previous record
     (`prev_hash`) and folds it into its own `record_hash`. This makes the log
     append-only in practice: you cannot alter or delete a record in the middle
     of the chain without recomputing every subsequent record's hash, which the
     verifier detects by recomputing the whole chain.

This is the honest, defensible basis for "court-admissible evidence": we cannot
prove an image is real, but we CAN prove it has not been altered since capture
and that the audit log has not been tampered with.

No external dependencies — pure hashlib + json so it is trivially auditable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np

ALGORITHM = "sha256"
# Sentinel for the first link in a chain (the "genesis" predecessor).
GENESIS_HASH = "0" * 64


# ──────────────────────────────────────────────────────────────────────────
# Low-level hashing primitives
# ──────────────────────────────────────────────────────────────────────────
def _to_bytes(data: Optional[Union[bytes, bytearray, np.ndarray]]) -> bytes:
    """Normalise an image / blob into deterministic bytes for hashing.

    For numpy arrays we hash the raw buffer *plus* the shape and dtype so that
    two arrays with identical bytes but different geometry never collide.
    """
    if data is None:
        return b""
    if isinstance(data, np.ndarray):
        header = f"{data.dtype}|{data.shape}|".encode("utf-8")
        return header + data.tobytes()
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise TypeError(f"Unhashable evidence payload type: {type(data)!r}")


def hash_bytes(data: Optional[Union[bytes, bytearray, np.ndarray]]) -> str:
    """Full hex SHA-256 of an image / blob (empty -> hash of empty string)."""
    return hashlib.sha256(_to_bytes(data)).hexdigest()


def canonical_metadata(metadata: dict) -> str:
    """Deterministic JSON encoding of metadata (stable key order, no whitespace)."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)


def compute_content_hash(
    metadata: dict,
    original: Optional[Union[bytes, np.ndarray]] = None,
    annotated: Optional[Union[bytes, np.ndarray]] = None,
) -> str:
    """SHA-256 binding the metadata to the original + annotated imagery.

    The three components are length-prefixed before concatenation so that the
    boundary between them is unambiguous (prevents extension/ambiguity attacks).
    """
    parts = [
        _to_bytes(original),
        _to_bytes(annotated),
        canonical_metadata(metadata).encode("utf-8"),
    ]
    h = hashlib.sha256()
    for p in parts:
        h.update(str(len(p)).encode("ascii"))
        h.update(b":")
        h.update(p)
    return h.hexdigest()


def compute_record_hash(prev_hash: str, content_hash: str, sealed_at: str) -> str:
    """Fold the predecessor hash into this record to form the chain link."""
    h = hashlib.sha256()
    h.update((prev_hash or GENESIS_HASH).encode("ascii"))
    h.update(b"|")
    h.update(content_hash.encode("ascii"))
    h.update(b"|")
    h.update(sealed_at.encode("ascii"))
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Sealing & verification
# ──────────────────────────────────────────────────────────────────────────
def seal_record(
    metadata: dict,
    original: Optional[Union[bytes, np.ndarray]] = None,
    annotated: Optional[Union[bytes, np.ndarray]] = None,
    prev_hash: str = GENESIS_HASH,
    sealed_at: Optional[str] = None,
) -> dict:
    """Produce the integrity seal for one evidence record.

    Returns a dict suitable for embedding in the stored record:
        {
          "algorithm": "sha256",
          "sealed_at": "<iso8601-utc>",
          "content_hash": "<64 hex>",   # binds image(s) + metadata
          "prev_hash": "<64 hex>",      # link to previous record
          "record_hash": "<64 hex>",    # this record's chain hash
        }
    """
    sealed_at = sealed_at or datetime.now(timezone.utc).isoformat()
    content_hash = compute_content_hash(metadata, original, annotated)
    record_hash = compute_record_hash(prev_hash or GENESIS_HASH, content_hash, sealed_at)
    return {
        "algorithm": ALGORITHM,
        "sealed_at": sealed_at,
        "content_hash": content_hash,
        "prev_hash": prev_hash or GENESIS_HASH,
        "record_hash": record_hash,
    }


def verify_record(
    seal: dict,
    metadata: dict,
    original: Optional[Union[bytes, np.ndarray]] = None,
    annotated: Optional[Union[bytes, np.ndarray]] = None,
) -> dict:
    """Verify a single sealed record.

    If `original`/`annotated` are supplied the content hash is fully re-derived
    from the imagery; otherwise only the metadata-binding and the internal
    record-hash consistency are checked.

    Returns {"valid": bool, "checks": {...}, "reason": str}.
    """
    checks = {}

    # 1. Recompute the chain link from the stored content hash + metadata.
    expected_record = compute_record_hash(
        seal.get("prev_hash", GENESIS_HASH),
        seal.get("content_hash", ""),
        seal.get("sealed_at", ""),
    )
    checks["record_hash"] = (expected_record == seal.get("record_hash"))

    # 2. If imagery / metadata provided, recompute the content hash.
    if original is not None or annotated is not None or metadata is not None:
        expected_content = compute_content_hash(metadata, original, annotated)
        checks["content_hash"] = (expected_content == seal.get("content_hash"))
    else:
        checks["content_hash"] = True  # not checkable without inputs

    valid = all(checks.values())
    reason = "Record intact." if valid else (
        "Content altered since sealing." if not checks["content_hash"]
        else "Chain link inconsistent (record hash mismatch)."
    )
    return {"valid": valid, "checks": checks, "reason": reason}


def verify_chain(seals: list) -> dict:
    """Verify the linkage of an ordered list of seals (oldest -> newest).

    Checks that each record's `prev_hash` equals the previous record's
    `record_hash`, and that every `record_hash` is internally consistent.
    Does NOT re-hash imagery (use verify_record per item for that).

    Returns {"valid": bool, "length": int, "broken_at": Optional[int], "reason": str}.
    """
    prev = GENESIS_HASH
    for i, seal in enumerate(seals):
        # Internal consistency of this link
        expected_record = compute_record_hash(
            seal.get("prev_hash", GENESIS_HASH),
            seal.get("content_hash", ""),
            seal.get("sealed_at", ""),
        )
        if expected_record != seal.get("record_hash"):
            return {"valid": False, "length": len(seals), "broken_at": i,
                    "reason": f"Record {i} hash is inconsistent (tampered or corrupted)."}
        # Linkage to predecessor
        if seal.get("prev_hash", GENESIS_HASH) != prev:
            return {"valid": False, "length": len(seals), "broken_at": i,
                    "reason": f"Chain broken at record {i}: prev_hash does not match preceding record."}
        prev = seal.get("record_hash")

    return {"valid": True, "length": len(seals), "broken_at": None,
            "reason": "Chain intact — append-only log verified end to end."}
