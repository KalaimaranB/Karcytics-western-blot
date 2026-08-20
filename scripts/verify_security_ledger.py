#!/usr/bin/env python3
"""Read-only check that security.json still matches the working tree.

Mirrors the audit steps inside Karcytics-SDK's
PluginSigner.project_sign_plugin (the pyproject.toml manifest-binding
check and the per-file hash audit) without needing karcytics_sdk
importable (which pulls in PyQt6) or the CI project key — just enough
to answer "would CI's project-sign step reject this?" in under a
second, so a stale signature fails at commit/push time or at the start
of CI, not after the full lint+test matrix has already run.

If Karcytics-SDK's hashing algorithm (PluginSigner._hash_file /
manifest_hash) ever changes, update the constants and hash_file() below
to match.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_EXTENSIONS = {".py", ".pyw", ".json", ".yml", ".yaml", ".toml", ".md", ".txt"}


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            text = path.read_text(encoding="utf-8")
            hasher.update(text.replace("\r\n", "\n").encode("utf-8"))
            return hasher.hexdigest()
        except UnicodeDecodeError:
            pass
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def main() -> int:
    security_file = ROOT / "security.json"
    manifest_file = ROOT / "pyproject.toml"

    if not security_file.exists():
        print("No security.json found yet — nothing to verify. Run 'karcytics-sdk sign .' first.")
        return 0

    security_data = json.loads(security_file.read_text(encoding="utf-8"))
    errors: list[str] = []

    manifest_text = manifest_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    actual_manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    expected_manifest_hash = security_data.get("manifest_hash")
    if actual_manifest_hash != expected_manifest_hash:
        errors.append(
            "pyproject.toml has changed since security.json was last signed "
            f"(expected {expected_manifest_hash}, computed {actual_manifest_hash})."
        )

    for rel_path, expected_hash in security_data.get("hashes", {}).items():
        file_path = ROOT / rel_path
        if not file_path.exists():
            errors.append(f"{rel_path} is listed in security.json but no longer exists.")
            continue
        if hash_file(file_path) != expected_hash:
            errors.append(f"{rel_path} has changed since it was signed.")

    if errors:
        print("Security ledger is out of sync with the working tree:")
        for e in errors:
            print(f"  - {e}")
        print("\nRun 'karcytics-sdk sign .' to resync, then commit the result.")
        return 1

    print("Security ledger matches the working tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
