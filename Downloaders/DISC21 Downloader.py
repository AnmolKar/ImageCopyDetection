#!/usr/bin/env python3
"""Parallel downloader for the DISC21 manifest stored in ``links.json``.

The script fetches every URL listed in the manifest, renames each payload to the
friendly ``name`` given in the file, and (optionally) extracts archives into a
``DISC21`` folder located in the current working directory (created on demand).
Use the ``--workers`` flag to control concurrency when downloading many shards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm.auto import tqdm

LOG = logging.getLogger(__name__)

ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tar.xz",
    ".txz",
    ".tar",
    ".zip",
)

# Default output inside the directory where the command is launched, per request.
DEFAULT_OUTPUT = (Path.cwd() / "DISC21").resolve()
DEFAULT_MANIFEST = Path(__file__).with_name("links.json")


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    url: str


@dataclass(frozen=True)
class DownloadConfig:
    output_dir: Path
    chunk_size: int
    timeout: int
    overwrite: bool
    extract: bool
    reextract: bool
    show_progress: bool


def load_manifest(manifest_path: Path) -> List[ManifestEntry]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    entries: List[ManifestEntry] = []
    for idx, item in enumerate(data):
        try:
            name = str(item["name"])
            url = str(item["link"])
        except KeyError as exc:  # pragma: no cover - validates manifest integrity
            raise ValueError(f"Manifest entry {idx} is missing the field {exc}.") from exc
        entries.append(ManifestEntry(name=name, url=url))
    return entries


def is_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return any(suffixes.endswith(sfx) for sfx in ARCHIVE_SUFFIXES)


def make_extract_dir(output_dir: Path, archive_path: Path) -> Path:
    candidate = archive_path.name
    for suffix in ARCHIVE_SUFFIXES:
        if candidate.lower().endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break
    return output_dir / candidate


def download_file(url: str, destination: Path, *, chunk_size: int, timeout: int, show_progress: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    progress = None
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0"))
            if show_progress:
                progress = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {destination.name}",
                    leave=False,
                )

            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))

        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    finally:
        if progress is not None:
            progress.close()


def _guard_path(base_dir: Path, target_path: Path) -> None:
    base = str(base_dir.resolve())
    target = str(target_path.resolve())
    if os.path.commonpath([base, target]) != base:
        raise RuntimeError("Refused to extract outside of destination directory.")


def extract_archive(archive_path: Path, destination: Path, *, reextract: bool, show_progress: bool) -> None:
    if not is_archive(archive_path):
        LOG.info("Skipping extraction for %s (not an archive)", archive_path.name)
        return

    if destination.exists():
        if not reextract:
            LOG.info("Extraction skipped for %s (destination %s exists)", archive_path.name, destination)
            return
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive_path.suffixes).lower()
    if ".tar" in suffixes:
        mode = "r:*"
        with tarfile.open(archive_path, mode) as tar:
            members = tar.getmembers()
            iterable: Iterable[tarfile.TarInfo] = members
            iterator = tqdm(iterable, desc=f"Extracting {archive_path.name}", unit="file", leave=False) if show_progress else iterable
            for member in iterator:
                member_path = destination / member.name
                _guard_path(destination, member_path)
                tar.extract(member, path=destination)
    elif suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            members = zf.infolist()
            iterable = members
            iterator = tqdm(iterable, desc=f"Extracting {archive_path.name}", unit="file", leave=False) if show_progress else iterable
            for member in iterator:
                member_path = destination / member.filename
                _guard_path(destination, member_path)
                zf.extract(member, path=destination)
    else:
        LOG.info("Unsupported archive type for %s", archive_path.name)


def process_entry(entry: ManifestEntry, cfg: DownloadConfig) -> None:
    target_path = cfg.output_dir / entry.name

    if target_path.exists() and not cfg.overwrite:
        LOG.info("Skipping %s (already downloaded)", entry.name)
    else:
        LOG.info("Fetching %s", entry.name)
        download_file(
            entry.url,
            target_path,
            chunk_size=cfg.chunk_size,
            timeout=cfg.timeout,
            show_progress=cfg.show_progress,
        )
        LOG.info("Saved %s", target_path)

    if cfg.extract and target_path.exists() and is_archive(target_path):
        extraction_dir = make_extract_dir(cfg.output_dir, target_path)
        LOG.info("Extracting %s to %s", target_path.name, extraction_dir)
        extract_archive(target_path, extraction_dir, reextract=cfg.reextract, show_progress=cfg.show_progress)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract assets listed in links.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to the JSON manifest (default: links.json next to this script)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory where the files will be stored")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4) * 2), help="Number of concurrent download workers")
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024, help="Download chunk size in bytes")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--no-extract", dest="extract", action="store_false", help="Skip archive extraction")
    parser.add_argument("--reextract", action="store_true", help="Force re-extraction even if the output folder exists")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bars")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    parser.set_defaults(extract=True, progress=True)
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    manifest = load_manifest(args.manifest)
    if not manifest:
        LOG.info("Manifest %s is empty. Nothing to download.", args.manifest)
        return 0

    cfg = DownloadConfig(
        output_dir=args.output_dir.expanduser(),
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        overwrite=args.overwrite,
        extract=args.extract,
        reextract=args.reextract,
        show_progress=args.progress,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(process_entry, entry, cfg): entry for entry in manifest}
        iterator = concurrent.futures.as_completed(future_map)
        if cfg.show_progress:
            iterator = tqdm(iterator, total=len(future_map), desc="Jobs", unit="file")
        for future in iterator:
            entry = future_map[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - critical download failure path
                failures.append((entry.name, exc))
                LOG.exception("Failed to process %s", entry.name)

    if failures:
        failed_labels = ", ".join(name for name, _ in failures)
        LOG.error("%s file(s) failed: %s", len(failures), failed_labels)
        return 1

    LOG.info("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
