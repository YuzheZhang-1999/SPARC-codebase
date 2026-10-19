"""Restore the lossless BSEA calibration from the distributed binary parts."""

import argparse
import hashlib
from pathlib import Path

CALIBRATION_SHA256 = "22c5f42bcc78f0faa16dc7fc2cb45eec8946456c2118518f8b40bed312fbbdb4"
PARTS = (
    ("mask.mat.001", "9ace45aeb7ab215759ace29d42c132ca6e5ff842cfffe3db55d4aed9121a3894"),
    ("mask.mat.002", "7ed1851c6045ec2a8574d2d5932f52560ee527255c9639d4a4b0a8b767831045"),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(output):
    source = Path(__file__).resolve().parents[1] / "examples" / "calibration"
    if output.exists():
        if file_sha256(output) == CALIBRATION_SHA256:
            print(f"Calibration already restored and verified: {output}")
            return
        raise FileExistsError(f"Existing output has a different checksum: {output}")
    for filename, expected in PARTS:
        if file_sha256(source / filename) != expected:
            raise ValueError(f"Calibration part is incomplete or corrupted: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".restoring")
    try:
        with temporary.open("xb") as destination:
            for filename, _ in PARTS:
                with (source / filename).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        destination.write(chunk)
        if file_sha256(temporary) != CALIBRATION_SHA256:
            raise ValueError("Restored calibration checksum does not match")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Restored and SHA-256 verified: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "calibration" / "mask.mat",
    )
    restore(parser.parse_args().output)
