"""Download the official NASA PSI experimental tables (BASS-II PSI-25, FLEX PSI-69) and verify them.

Uses the PSI portal's own public API (the endpoints its web app calls; no login):
  metadata  GET  https://psi.nasa.gov/geode-py/ws/repo/investigations/<PSI-ID>
  download  GET  https://psi.nasa.gov/geode-py/ws/studies/<PSI-ID>/download?file=<name>&version=<v>&redirect=false
            -> returns a short-lived pre-signed URL on NASA's psi-repo-prod S3 bucket

Each file is accepted only if
  * its byte size equals the size in NASA's metadata, and
  * its SHA-256 equals the checksum registered in config/data_sources.json (official_files).
Files are stored unchanged under data/raw/nasa_psi/. An existing file is never overwritten silently:
if it already matches it is left alone; if it differs the script stops.

Run:  python -m src.fetch_nasa_psi_tables            (download + verify)
      python -m src.fetch_nasa_psi_tables --verify   (verify local files only, no network)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "data_sources.json"
API = "https://psi.nasa.gov/geode-py/ws"
UA = {"User-Agent": "FlameInFreefall-provenance/1.0 (+https://github.com/StudyGeng/Nasa-Space-Apps-Sarawak)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as response:
        return response.read()


def official_files() -> list[dict]:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return [f | {"dataset_key": key} for key, d in sources["datasets"].items() for f in d.get("official_files", [])]


def verify_local(entry: dict) -> tuple[bool, str]:
    path = ROOT / entry["path"]
    if not path.exists():
        return False, f"missing: {entry['path']}"
    data = path.read_bytes()
    if len(data) != entry["size_bytes"]:
        return False, f"size {len(data)} != registered {entry['size_bytes']}"
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        return False, f"sha256 {digest} != registered {entry['sha256']}"
    return True, "ok"


def download(entry: dict) -> str:
    path = ROOT / entry["path"]
    ok, why = verify_local(entry)
    if ok:
        return "already present and verified"
    if path.exists():
        raise SystemExit(f"{entry['path']} exists but does not match the registry ({why}); refusing to overwrite.")
    meta = json.loads(_get(f"{API}/repo/investigations/{entry['investigation_id']}"))
    table = next(t for t in meta["experimentalTable"] if t["file_name"] == entry["file_name"])
    if table["file_size"] != entry["size_bytes"]:
        raise SystemExit(f"NASA metadata now lists {table['file_size']} bytes for {entry['file_name']}; registry has "
                         f"{entry['size_bytes']}. The official file may have changed: review before updating the registry.")
    query = urllib.parse.urlencode({"file": entry["file_name"], "version": table["version"], "redirect": "false"})
    presigned = _get(f"{API}/studies/{entry['investigation_id']}/download?{query}").decode().strip().strip('"')
    if urllib.parse.urlparse(presigned).netloc != "psi-repo-prod.s3.amazonaws.com":
        raise SystemExit(f"Unexpected download host {urllib.parse.urlparse(presigned).netloc}; refusing.")
    data = _get(presigned)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != entry["size_bytes"] or digest != entry["sha256"]:
        raise SystemExit(f"Downloaded {entry['file_name']} does not match the registry (size {len(data)}, sha256 {digest}).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "downloaded and verified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only verify local files against the registry")
    args = parser.parse_args(argv)
    failures = 0
    for entry in official_files():
        if args.verify:
            ok, why = verify_local(entry)
            failures += not ok
            print(f"{entry['dataset_key']:8s} {entry['path']}: {'VERIFIED' if ok else 'FAILED'} ({why})")
        else:
            print(f"{entry['dataset_key']:8s} {entry['path']}: {download(entry)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
