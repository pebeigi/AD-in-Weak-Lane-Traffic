#!/usr/bin/env python3
"""Verify and optionally extract the released trajectory dataset."""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "Final_Lebanon_Data.xz"
EXTRACTED = ROOT / "Final_Lebanon_Data.csv"
ARCHIVE_SHA256 = "0a949b5c9d93f4d4ee95a3b27fdd36e1dadd08506c5cb7461c4a6043f206ddbf"
EXTRACTED_SHA256 = "5807e3122a15b89dcf19e2ec13fa8d8126466267df711361801edfc5cbf9268a"
CHUNK_SIZE = 1 << 20


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(extract: bool = False) -> None:
    archive_digest = file_digest(ARCHIVE)
    if archive_digest != ARCHIVE_SHA256:
        raise SystemExit(
            f"Archive checksum mismatch: expected {ARCHIVE_SHA256}, got {archive_digest}"
        )

    if EXTRACTED.exists():
        extracted_digest = file_digest(EXTRACTED)
        if extracted_digest != EXTRACTED_SHA256:
            raise SystemExit(
                f"Refusing to overwrite {EXTRACTED}: checksum is {extracted_digest}, "
                f"expected {EXTRACTED_SHA256}"
            )
        print(f"Verified {ARCHIVE.relative_to(ROOT)} and {EXTRACTED.name}")
        return

    output = EXTRACTED.with_suffix(EXTRACTED.suffix + ".tmp") if extract else None
    digest = hashlib.sha256()
    try:
        with lzma.open(ARCHIVE, "rb") as source:
            target = output.open("wb") if output else None
            try:
                for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                    digest.update(chunk)
                    if target:
                        target.write(chunk)
            finally:
                if target:
                    target.close()
        if digest.hexdigest() != EXTRACTED_SHA256:
            raise SystemExit(
                "Uncompressed checksum mismatch: "
                f"expected {EXTRACTED_SHA256}, got {digest.hexdigest()}"
            )
        if output:
            os.replace(output, EXTRACTED)
            print(f"Verified and extracted {ARCHIVE.relative_to(ROOT)} to {EXTRACTED.name}")
        else:
            print(f"Verified {ARCHIVE.relative_to(ROOT)} (compressed and uncompressed content)")
    finally:
        if output and output.exists():
            output.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="also create Final_Lebanon_Data.csv")
    args = parser.parse_args()
    verify_archive(args.extract)


if __name__ == "__main__":
    main()
