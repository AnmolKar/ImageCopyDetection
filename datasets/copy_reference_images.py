#!/usr/bin/env python3
"""Copy DISC21 reference images that match reference_id values from the public ground truth CSV."""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from tqdm import tqdm


def read_reference_ids(csv_path: Path) -> List[str]:
    """Return the unique, non-empty reference_id values from the CSV."""
    reference_ids: Set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "reference_id" not in reader.fieldnames:
            raise ValueError("Column 'reference_id' not found in CSV header.")
        for row in tqdm(reader, desc="Reading reference_id", unit="row"):
            value = row["reference_id"].strip()
            if value:
                reference_ids.add(value)
    # Preserve deterministic ordering for later progress reporting
    return sorted(reference_ids)


def build_shard_dirs(first_shard: Path, shard_count: int) -> List[Path]:
    """Return ordered shard directories refs_50k_0 .. refs_50k_{shard_count-1}."""
    base_name = first_shard.name
    if not base_name.endswith("_0"):
        raise ValueError(
            f"Expected first shard directory name to end with '_0', got '{base_name}'."
        )
    prefix = base_name[: -len("_0")]
    parent = first_shard.parent
    shard_dirs = [(parent / f"{prefix}_{idx}") for idx in range(shard_count)]
    missing_dirs = [path for path in shard_dirs if not path.is_dir()]
    if missing_dirs:
        missing = "\n".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Shard directories not found:\n{missing}")
    return shard_dirs


def iter_image_files(shard_dir: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield image files in shard_dir that match allowed extensions."""
    allowed = {ext.lower() for ext in extensions}
    for path in shard_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def index_reference_images(
    shard_dirs: Sequence[Path],
    extensions: Sequence[str],
) -> Dict[str, Path]:
    """Create lookup table of reference_id -> image Path."""
    lookup: Dict[str, Path] = {}
    for shard_dir in shard_dirs:
        iterator = iter_image_files(shard_dir, extensions)
        for image_path in tqdm(
            iterator, desc=f"Indexing {shard_dir.name}", unit="img", leave=False
        ):
            lookup.setdefault(image_path.stem, image_path)
    return lookup


def copy_reference_images(
    reference_ids: Sequence[str],
    lookup: Dict[str, Path],
    destination: Path,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[int, int, List[str]]:
    """Copy matched images into destination and return (copied, skipped, missing_ids)."""
    destination.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    copied = 0
    skipped = 0
    for ref_id in tqdm(reference_ids, desc="Copying matches", unit="img"):
        source = lookup.get(ref_id)
        if source is None:
            missing.append(ref_id)
            continue
        target = destination / (ref_id + source.suffix.lower())
        if target.exists() and not overwrite:
            skipped += 1
            continue
        if dry_run:
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, skipped, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        default="/home/ankarki/DeeplearningProj/NDEC/public_ground_truth_h5.csv",
        type=Path,
        help="Path to the public ground truth CSV file.",
    )
    parser.add_argument(
        "--first-shard",
        default="/home/ankarki/DeeplearningProj/DISC21/refs_50k_0",
        type=Path,
        help="Path to the refs_50k_0 directory (used to derive shard list).",
    )
    parser.add_argument(
        "--shard-count",
        default=20,
        type=int,
        help="Total number of shard directories to scan (refs_50k_0..refs_50k_{n-1}).",
    )
    parser.add_argument(
        "--destination",
        default="/home/ankarki/DeeplearningProj/NDEC/reference_set",
        type=Path,
        help="Directory to copy matched reference images into.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".jpg", ".jpeg"],
        help="Image extensions to consider when indexing the shard directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index and report matches without copying any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the destination if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path.expanduser()
    first_shard: Path = args.first_shard.expanduser()
    destination: Path = args.destination.expanduser()

    reference_ids = read_reference_ids(csv_path)
    shard_dirs = build_shard_dirs(first_shard, args.shard_count)
    lookup = index_reference_images(shard_dirs, args.extensions)
    copied, skipped, missing = copy_reference_images(
        reference_ids,
        lookup,
        destination,
        args.dry_run,
        args.overwrite,
    )

    print()
    print("Results")
    print("=" * 40)
    print(f"Unique reference IDs: {len(reference_ids):,}")
    print(f"Images found: {len(reference_ids) - len(missing):,}")
    print(f"Images copied: {copied:,}" if not args.dry_run else "Images copied: 0 (dry run)")
    print(f"Images skipped (already present): {skipped:,}")
    print(f"Missing images: {len(missing):,}")
    if missing:
        missing_preview = ", ".join(missing[:10])
        print(f"Sample missing IDs: {missing_preview}...")


if __name__ == "__main__":
    main()
