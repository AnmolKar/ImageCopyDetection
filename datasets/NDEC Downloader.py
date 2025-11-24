# %% [markdown]
# # NDEC Dataset Downloader
# Use this notebook to fetch the NDEC training, query, and ground-truth assets into `/home/jowatson/Deep Learning/NDEC`. Run the cells in order; each download and extraction step shows a progress bar so you can monitor long transfers.

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import tarfile
import zipfile

from typing import Iterable, List, Dict

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

# %%
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
    # {
    #     "name": "Query set",
    #     "url": "https://huggingface.co/datasets/WenhaoWang/ASL/resolve/main/query_images_h5.tar",
    #     "filename": "query_images_h5.tar",
    #     "protocol": "http",
    # },
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


def download_all(target_dir: Path, datasets: Iterable[Dict[str, str]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloads will be stored in: %s", target_dir)

    for item in datasets:
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
            continue

        extraction_dir = target_dir / item["name"].lower().replace(" ", "_")
        try:
            extract_archive(target_file, extraction_dir)
        except (tarfile.TarError, zipfile.BadZipFile) as err:
            tqdm.write(f"Could not extract {target_file.name}: {err}")


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

    download_all(args.target_dir, datasets)
    LOG.info("Download job finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


