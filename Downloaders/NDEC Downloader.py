# %% [markdown]
# # NDEC Dataset Downloader
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import tarfile
import zipfile

from typing import Iterable, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm.auto import tqdm

LOG = logging.getLogger(__name__)


def ensure_gdown() -> None:
    """Import gdown, installing it on the fly if missing."""
    try:
        import gdown  # type: ignore
    except ImportError:  # pragma: no cover - best effort for runtime install
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown  # type: ignore


def stream_download(url: str, destination: Path, chunk_size: int = 1 << 20) -> Path:
    """Download a file with a progress bar using streaming requests."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        )
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    progress.update(len(chunk))
        progress.close()
    return destination


def download_google_file(url: str, destination: Path) -> Path:
    """Download a Google Drive share link using gdown with progress."""
    ensure_gdown()
    import gdown  # type: ignore

    destination.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Downloading {destination.name} via Google Drive...")
    gdown.download(url, str(destination), quiet=False)
    return destination


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract tar/zip archives with a progress bar."""
    suffixes = "".join(archive_path.suffixes)
    destination.mkdir(parents=True, exist_ok=True)

    def _extract_with_progress(members, extractor):
        for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
            extractor(member)

    if ".tar" in suffixes:
        mode = "r:*"
        with tarfile.open(archive_path, mode) as tar:
            members = tar.getmembers()
            _extract_with_progress(members, lambda m: tar.extract(m, path=destination))
    elif suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            members = zf.infolist()
            _extract_with_progress(members, lambda m: zf.extract(m, path=destination))
    else:
        tqdm.write(f"Skipping extraction for {archive_path.name}: unsupported archive type.")


DATASETS: List[Dict[str, str]] = [
    {
        "name": "Training set 1",
        "url": "https://huggingface.co/datasets/WenhaoWang/ASL/resolve/main/negative_pair.tar.aa?download=true",
        "filename": "negative_pair.tar.aa",
        "protocol": "http",
    },
    {
        "name": "Training set 2",
        "url": "https://huggingface.co/datasets/WenhaoWang/ASL/resolve/main/negative_pair.tar.ab?download=true",
        "filename": "negative_pair.tar.ab",
        "protocol": "http",
    },
    {
        "name": "Training set 3",
        "url": "https://huggingface.co/datasets/WenhaoWang/ASL/resolve/main/negative_pair.tar.ac?download=true",
        "filename": "negative_pair.tar.ac",
        "protocol": "http",
    },
    {
        "name": "Query set",
        "url": "https://huggingface.co/datasets/WenhaoWang/ASL/resolve/main/query_images_h5.tar",
        "filename": "query_images_h5.tar",
        "protocol": "http",
    },
]


def select_datasets(
    datasets: Iterable[Dict[str, str]], shard_index: int | None, shard_count: int | None
) -> List[Dict[str, str]]:
    if shard_index is None:
        return list(datasets)

    if shard_count is None or shard_count <= 0:
        raise ValueError("Shard count must be provided and > 0 when sharding downloads")
    if not (0 <= shard_index < shard_count):
        raise ValueError("Shard index must satisfy 0 <= index < shard_count")

    selected = [
        item for idx, item in enumerate(datasets) if idx % shard_count == shard_index
    ]
    LOG.info(
        "Assigned %s dataset(s) to shard %s/%s",
        len(selected),
        shard_index,
        shard_count,
    )
    if not selected:
        LOG.warning(
            "Shard index %s has no assigned datasets (shard_count=%s). Nothing to do.",
            shard_index,
            shard_count,
        )
    return selected


def _download_and_extract(item: Dict[str, str], target_dir: Path) -> None:
    """Download a single dataset item and extract it if it's an archive."""
    target_file = target_dir / item["filename"]
    tqdm.write(f"\n=== {item['name']} ===")

    if target_file.exists():
        tqdm.write(f"{target_file.name} already exists, skipping download.")
    else:
        if item["protocol"] == "http":
            stream_download(item["url"], target_file)
        elif item["protocol"] == "gdrive":
            download_google_file(item["url"], target_file)
        else:
            raise ValueError(f"Unsupported protocol: {item['protocol']}")

    suffixes = "".join(target_file.suffixes)
    is_tar_archive = suffixes.endswith(
        (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")
    )
    is_zip_archive = suffixes.endswith(".zip")
    should_extract = is_tar_archive or is_zip_archive
    if not should_extract:
        tqdm.write(f"Skipping extraction for {target_file.name} (not an archive).")
        return

    extraction_dir = target_dir / item["name"].lower().replace(" ", "_")
    try:
        extract_archive(target_file, extraction_dir)
    except (tarfile.TarError, zipfile.BadZipFile) as err:
        tqdm.write(f"Could not extract {target_file.name}: {err}")


def download_all(
    target_dir: Path,
    datasets: Iterable[Dict[str, str]],
    max_workers: int = 3,
) -> None:
    """Download and extract all datasets, using multithreading per job."""
    target_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloads will be stored in: %s", target_dir)

    datasets_list = list(datasets)
    if not datasets_list:
        LOG.info("No datasets to download.")
        return

    # Guard max_workers
    max_workers = max(1, min(max_workers, len(datasets_list)))

    LOG.info("Using up to %s worker thread(s) for downloads.", max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(_download_and_extract, item, target_dir): item
            for item in datasets_list
        }

        for _ in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Datasets"):
            future = _
            item = future_to_item[future]
            try:
                future.result()
            except Exception as e:
                LOG.error("Error processing %s: %s", item.get("name", "?"), e)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NDEC dataset artifacts.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("/home/jowatson/Deep Learning/NDEC").expanduser(),
        help="Directory where archives and extracted data will be stored.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for parallel (array) jobs.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total number of shards when running in parallel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent download/extract worker threads.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    datasets = select_datasets(DATASETS, args.shard_index, args.shard_count)
    if not datasets:
        LOG.info("No datasets assigned to this shard. Exiting.")
        return 0

    download_all(args.target_dir, datasets, max_workers=args.max_workers)
    LOG.info("Download job finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
