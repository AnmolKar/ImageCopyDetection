import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODE_DIR.parent if CODE_DIR.parent != CODE_DIR else CODE_DIR


def _default_root_from_env(env_var: str, folder_name: str) -> Path:
    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path).expanduser()

    candidates = [Path.cwd(), Path.cwd().parent, WORKSPACE_ROOT]
    for base in candidates:
        candidate = base / folder_name
        if candidate.exists():
            return candidate

    return WORKSPACE_ROOT / folder_name


DEFAULT_DISC21_ROOT = _default_root_from_env("DISC21_PATH", "DISC21")
DEFAULT_IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Disc21DataConfig:
    """Convenience bundle to describe how to construct loaders."""

    root: Optional[Path] = None
    img_size_train: int = 224
    img_size_eval: int = 224
    train_shards: Optional[Sequence[int]] = None
    ref_shards: Optional[Sequence[int]] = None
    batch_size_train: int = 64
    batch_size_eval: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    def resolve_root(self) -> Path:
        return resolve_disc21_root(self.root)


def resolve_disc21_root(root: Optional[Path] = None) -> Path:
    """Resolve the DISC21 root directory, validating its existence."""

    path = Path(root) if root is not None else DEFAULT_DISC21_ROOT
    if not path.exists():
        raise FileNotFoundError(
            f"DISC21 root not found at {path}. Set DISC21_PATH or pass root explicitly."
        )
    return path


def build_transforms(
    img_size_train: int = 224,
    img_size_eval: int = 224,
    augment_train: bool = True,
) -> Tuple[T.Compose, T.Compose]:
    """Return sensible default torchvision transforms for train/eval."""

    train_tfms = [T.Resize((img_size_train, img_size_train))]
    if augment_train:
        train_tfms.append(T.RandomHorizontalFlip())
    train_tfms.append(T.ToTensor())

    eval_tfms = [
        T.Resize((img_size_eval, img_size_eval)),
        T.ToTensor(),
    ]

    return T.Compose(train_tfms), T.Compose(eval_tfms)


class Disc21FolderDataset(Dataset):
    """Load RGB tensors from a DISC21-style folder tree."""

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        extensions: Sequence[str] = DEFAULT_IMG_EXTS,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        exts = {e.lower() for e in extensions}

        if not self.root.exists():
            raise FileNotFoundError(f"Root folder not found: {self.root}")

        self.paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        rel_id = str(path.relative_to(self.root))
        return img, rel_id


def _collect_shard_roots(
    root: Path,
    prefix: str,
    shards: Optional[Sequence[int]] = None,
    expected: int = 20,
) -> Sequence[Path]:
    """Return the resolved shard directories for train/ref splits."""

    shard_iter: Iterable[int]
    shard_iter = shards if shards is not None else range(expected)
    shard_paths = []
    for shard in shard_iter:
        shard_path = root / f"{prefix}_{shard}"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard folder: {shard_path}")
        shard_paths.append(shard_path)
    return shard_paths


def _concat_from_shards(
    shard_roots: Sequence[Path],
    transform: Optional[Callable],
    extensions: Sequence[str] = DEFAULT_IMG_EXTS,
) -> ConcatDataset:
    datasets = [
        Disc21FolderDataset(root=shard_root, transform=transform, extensions=extensions)
        for shard_root in shard_roots
    ]
    return ConcatDataset(datasets)


def get_train_dataset(
    root: Optional[Path] = None,
    transform: Optional[Callable] = None,
    shards: Optional[Sequence[int]] = None,
) -> ConcatDataset:
    """Return the concatenated training dataset (20×50k shards by default)."""

    disc_root = resolve_disc21_root(root)
    shard_roots = _collect_shard_roots(disc_root, "train_50k", shards)
    return _concat_from_shards(shard_roots, transform)


def get_reference_dataset(
    root: Optional[Path] = None,
    transform: Optional[Callable] = None,
    shards: Optional[Sequence[int]] = None,
) -> ConcatDataset:
    """Return the concatenated reference dataset (20×50k shards by default)."""

    disc_root = resolve_disc21_root(root)
    shard_roots = _collect_shard_roots(disc_root, "refs_50k", shards)
    return _concat_from_shards(shard_roots, transform)


def get_query_dataset(
    split: str = "dev",
    root: Optional[Path] = None,
    transform: Optional[Callable] = None,
) -> Disc21FolderDataset:
    """Load dev/test query folders (each single folder instead of shards)."""

    disc_root = resolve_disc21_root(root)
    split = split.lower()
    folder_map = {
        "dev": "dev_queries_50k_0",
        "test": "test_queries_50k_0",
    }
    if split not in folder_map:
        raise ValueError(f"Unsupported split '{split}'. Use 'dev' or 'test'.")
    query_root = disc_root / folder_map[split]
    return Disc21FolderDataset(query_root, transform=transform)


def load_groundtruth(split: str = "dev", root: Optional[Path] = None) -> pd.DataFrame:
    """Load the dev/test ground-truth CSV with stable column names."""

    disc_root = resolve_disc21_root(root)
    split = split.lower()
    if split == "dev":
        csv_path = disc_root / "dev_queries_groundtruth.csv"
    elif split == "test":
        csv_path = disc_root / "test_queries_groundtruth.csv"
    else:
        raise ValueError(f"Unsupported split '{split}'. Use 'dev' or 'test'.")
    return pd.read_csv(csv_path, header=None, names=["query_id", "reference_id"])


def disc21_id_to_path(root: Path, img_id: str) -> Path:
    """Map DISC21 identifiers to actual file paths relative to ``root``."""

    root = Path(root)
    img_id = str(img_id)

    if not img_id.lower().endswith((".jpg", ".jpeg", ".png")):
        img_id = img_id + ".jpg"

    root_name = root.name
    if "dev_queries" in root_name or "test_queries" in root_name:
        return root / img_id

    prefix = img_id[0]
    digits = "".join(ch for ch in img_id if ch.isdigit())
    idx = int(digits)

    shard = idx // 50000
    if prefix == "R":
        return root / f"refs_50k_{shard}" / img_id
    if prefix == "T":
        return root / f"train_50k_{shard}" / img_id
    return root / img_id


class QueryReferencePairDataset(Dataset):
    """Dataset that emits (query_img, ref_img, *metadata) tuples."""

    def __init__(
        self,
        csv_path: Path,
        query_root: Path,
        reference_root: Path,
        query_col: str = "query_id",
        reference_col: str = "reference_id",
        label_col: Optional[str] = None,
        transform_query: Optional[Callable] = None,
        transform_reference: Optional[Callable] = None,
        id_to_path_fn: Optional[Callable[[Path, str], Path]] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.query_root = Path(query_root)
        self.reference_root = Path(reference_root)
        self.query_col = query_col
        self.reference_col = reference_col
        self.label_col = label_col
        self.transform_query = transform_query
        self.transform_reference = transform_reference
        if id_to_path_fn is None:
            raise ValueError("id_to_path_fn must be provided for DISC21")
        self.id_to_path_fn = id_to_path_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        q_id = row[self.query_col]
        r_id = row[self.reference_col]

        q_path = self.id_to_path_fn(self.query_root, q_id)
        r_path = self.id_to_path_fn(self.reference_root, r_id)

        q_img = Image.open(q_path).convert("RGB")
        r_img = Image.open(r_path).convert("RGB")

        if self.transform_query is not None:
            q_img = self.transform_query(q_img)
        if self.transform_reference is not None:
            r_img = self.transform_reference(r_img)

        if self.label_col is not None:
            label = row[self.label_col]
            return q_img, r_img, label, str(q_id), str(r_id)
        return q_img, r_img, str(q_id), str(r_id)


def get_pair_dataset(
    split: str = "dev",
    root: Optional[Path] = None,
    transform_query: Optional[Callable] = None,
    transform_reference: Optional[Callable] = None,
    csv_path: Optional[Path] = None,
) -> QueryReferencePairDataset:
    """Instantiate the query-reference dataset for a given split."""

    disc_root = resolve_disc21_root(root)
    split = split.lower()
    if split == "dev":
        default_query_root = disc_root / "dev_queries_50k_0"
        default_csv = disc_root / "dev_queries_groundtruth.csv"
    elif split == "test":
        default_query_root = disc_root / "test_queries_50k_0"
        default_csv = disc_root / "test_queries_groundtruth.csv"
    else:
        raise ValueError(f"Unsupported split '{split}'. Use 'dev' or 'test'.")

    query_root = default_query_root
    ref_root = disc_root
    csv_path = csv_path or default_csv

    return QueryReferencePairDataset(
        csv_path=csv_path,
        query_root=query_root,
        reference_root=ref_root,
        transform_query=transform_query,
        transform_reference=transform_reference,
        id_to_path_fn=disc21_id_to_path,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """Thin wrapper that standardizes DataLoader defaults for DISC21."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_default_datasets(
    config: Optional[Disc21DataConfig] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Disc21FolderDataset, Disc21FolderDataset]:
    """Utility that mirrors the original script's dataset creation in one call."""

    cfg = config or Disc21DataConfig()
    disc_root = cfg.resolve_root()
    train_tfms, eval_tfms = build_transforms(cfg.img_size_train, cfg.img_size_eval)

    train_ds = get_train_dataset(disc_root, transform=train_tfms, shards=cfg.train_shards)
    ref_ds = get_reference_dataset(disc_root, transform=eval_tfms, shards=cfg.ref_shards)
    dev_queries = get_query_dataset("dev", disc_root, transform=eval_tfms)
    test_queries = get_query_dataset("test", disc_root, transform=eval_tfms)

    return train_ds, ref_ds, dev_queries, test_queries


def build_default_loaders(
    config: Optional[Disc21DataConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Return train/ref/dev-pairs/test-pairs loaders using config defaults."""

    cfg = config or Disc21DataConfig()
    train_tfms, eval_tfms = build_transforms(cfg.img_size_train, cfg.img_size_eval)
    train_ds = get_train_dataset(cfg.root, transform=train_tfms, shards=cfg.train_shards)
    ref_ds = get_reference_dataset(cfg.root, transform=eval_tfms, shards=cfg.ref_shards)
    dev_pairs = get_pair_dataset("dev", cfg.root, eval_tfms, eval_tfms)
    test_pairs = get_pair_dataset("test", cfg.root, eval_tfms, eval_tfms)

    train_loader = create_dataloader(
        train_ds,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    ref_loader = create_dataloader(
        ref_ds,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    dev_loader = create_dataloader(
        dev_pairs,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = create_dataloader(
        test_pairs,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, ref_loader, dev_loader, test_loader


__all__ = [
    "Disc21DataConfig",
    "Disc21FolderDataset",
    "QueryReferencePairDataset",
    "build_default_datasets",
    "build_default_loaders",
    "build_transforms",
    "create_dataloader",
    "disc21_id_to_path",
    "get_pair_dataset",
    "get_query_dataset",
    "get_reference_dataset",
    "get_train_dataset",
    "load_groundtruth",
    "resolve_disc21_root",
]


def _log_dataset_stats() -> None:
    """Helper for manual sanity checks without polluting imports."""

    try:
        train_ds, ref_ds, dev_ds, test_ds = build_default_datasets()
    except Exception as exc:  # pragma: no cover - best-effort helper
        logger.error("Failed to enumerate DISC21 datasets: %s", exc)
        return

    logger.info("Train images: %s", len(train_ds))
    logger.info("Reference images: %s", len(ref_ds))
    logger.info("Dev queries: %s", len(dev_ds))
    logger.info("Test queries: %s", len(test_ds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log_dataset_stats()


