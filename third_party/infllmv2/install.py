"""Build the pinned sparse-attention extension with the paper's teacher kernel."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="New directory for the native dependency source",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    patch = root / "paper1.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Native extension patch checksum mismatch")
    destination = args.directory.resolve()
    if destination.exists():
        raise ValueError(
            "Build directory must not exist; existing sources are preserved"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "init", str(destination))
    run("git", "remote", "add", "origin", manifest["upstream"], cwd=destination)
    run("git", "fetch", "--depth", "1", "origin", manifest["revision"], cwd=destination)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=destination)
    run("git", "apply", "--check", str(patch), cwd=destination)
    run("git", "apply", str(patch), cwd=destination)
    cutlass = destination / "csrc/cutlass"
    run("git", "init", str(cutlass))
    run("git", "remote", "add", "origin", manifest["cutlass"], cwd=cutlass)
    run(
        "git",
        "fetch",
        "--depth",
        "1",
        "origin",
        manifest["cutlass_revision"],
        cwd=cutlass,
    )
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=cutlass)
    for name, digest in manifest["files"].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Native source checksum mismatch: {name}")
    if not args.prepare_only:
        run(
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-build-isolation",
            str(destination),
        )


if __name__ == "__main__":
    main()
