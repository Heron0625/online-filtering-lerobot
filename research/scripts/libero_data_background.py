"""Resume pinned LIBERO downloads or watch completed raw tasks for CPU replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import h5py
import requests

REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def download(files: dict, root: Path) -> None:
    """Resume byte ranges, retry transports, and only publish complete files."""
    for path, meta in sorted(files.items()):
        output = root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        size = int(meta["size"])
        if output.exists() and output.stat().st_size == size:
            print(f"complete {path}", flush=True)
            continue
        part = output.with_suffix(".hdf5.part")
        attempt = 0
        while True:
            attempt += 1
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                with h5py.File(part, "r") as f:
                    if "data" not in f:
                        raise ValueError(f"invalid HDF5: {part}")
                part.replace(output)
                print(f"finished {path}: {size} bytes", flush=True)
                break
            if offset > size:
                raise ValueError(f"partial larger than expected: {part}")
            mirror = attempt % 2 == 1
            host = "https://hf-mirror.com" if mirror else "https://huggingface.co"
            session = requests.Session()
            session.trust_env = not mirror
            url = (
                f"{host}/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{REVISION}/{quote(path)}?download=true"
            )
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                print(f"downloading {path} from {offset}/{size} via {host}", flush=True)
                with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("server returned an incorrect range")
                        mode = "ab"
                    else:
                        mode = "wb"
                        offset = 0
                    progress = time.monotonic()
                    with part.open(mode) as f:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                            if time.monotonic() - progress >= 15:
                                print(f"progress {path}: {f.tell()}/{size}", flush=True)
                                progress = time.monotonic()
                if part.stat().st_size > size:
                    raise ValueError("download exceeded expected size")
            except (requests.RequestException, OSError, ValueError) as exc:
                print(f"retry {path}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(60, 5 * attempt))
            finally:
                session.close()
    print("DOWNLOAD COMPLETE: all 40 tasks", flush=True)


def watch(files: dict, raw: Path, output: Path, worker: int, workers: int, python: str) -> None:
    """Replay disjoint tasks; wait for atomic download completion and retry failures."""
    processor = Path(__file__).with_name("replay_libero_xvla_background.py")
    failed_until = {}
    while True:
        done = 0
        for path, meta in sorted(files.items()):
            source = raw / path
            target = output / path
            if target.exists():
                done += 1
                continue
            shard = int.from_bytes(hashlib.sha1(path.encode()).digest()[:8], "big") % workers
            if shard != worker or not source.exists() or source.stat().st_size != int(meta["size"]):
                continue
            if time.monotonic() < failed_until.get(path, 0):
                continue
            print(f"worker {worker} starting {path}", flush=True)
            result = subprocess.run(
                [python, str(processor), "--source", str(source), "--output", str(target)]
            )
            if result.returncode:
                failed_until[path] = time.monotonic() + 300
                print(f"worker {worker} retry later: {path} exit={result.returncode}", flush=True)
        if done == len(files):
            print("REPLAY COMPLETE: all 40 tasks", flush=True)
            return
        print(f"worker {worker}: completed {done}/40; waiting/checking", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "replay"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", default="/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python")
    args = parser.parse_args()
    files = json.loads(args.manifest.read_text())["files"]
    if len(files) != 40:
        raise ValueError(f"expected 40 task files, got {len(files)}")
    if args.mode == "download":
        download(files, args.raw)
    else:
        if args.output is None:
            parser.error("--output is required for replay")
        watch(files, args.raw, args.output, args.worker, args.workers, args.python)
