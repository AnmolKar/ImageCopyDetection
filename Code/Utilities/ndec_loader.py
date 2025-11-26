import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


DEFAULT_NDEC_ROOT = _default_root_from_env("NDEC_PATH", "NDEC")
DEFAULT_IMG_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_QUERY_SUBDIR = Path("query_set") / "query_images_h5"
DEFAULT_REFERENCE_SUBDIR = Path("reference_set_true match")
DEFAULT_NEGATIVE_SUBDIR = Path("negative_pair")
DEFAULT_GROUNDTRUTH_NAME = "public_ground_truth_h5.csv"


@dataclass
class NdecDataConfig:
    """Configuration bundle mirroring the DISC21 helpers."""

    root: Optional[Path] = None
    query_subdir: Path = DEFAULT_QUERY_SUBDIR
    reference_subdir: Path = DEFAULT_REFERENCE_SUBDIR
    negative_pair_subdir: Path = DEFAULT_NEGATIVE_SUBDIR
    groundtruth_csv: str = DEFAULT_GROUNDTRUTH_NAME
    img_size_train: int = 224
    img_size_eval: int = 224
    batch_size_pairs: int = 64
    batch_size_eval: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_missing_groundtruth: bool = True

    def resolve_root(self) -> Path:
        return resolve_ndec_root(self.root)

    def resolve_query_dir(self) -> Path:
        return self.resolve_root() / self.query_subdir

    def resolve_reference_dir(self) -> Path:
        return self.resolve_root() / self.reference_subdir

    def resolve_negative_dir(self) -> Path:
        return self.resolve_root() / self.negative_pair_subdir

    def resolve_groundtruth_csv(self) -> Path:
        return self.resolve_root() / self.groundtruth_csv


def resolve_ndec_root(root: Optional[Path] = None) -> Path:
    """Resolve the NDEC root directory, validating that it exists."""

    path = Path(root) if root is not None else DEFAULT_NDEC_ROOT
    if not path.exists():
        raise FileNotFoundError(
            f"NDEC root not found at {path}. Set NDEC_PATH or pass root explicitly."
        )
    return path


def build_transforms(
    img_size_train: int = 224,
    img_size_eval: int = 224,
    augment_train: bool = True,
) -> Tuple[T.Compose, T.Compose]:
    """Return torchvision transforms identical to the DISC defaults."""

    train_tfms = [T.Resize((img_size_train, img_size_train))]
    if augment_train:
        train_tfms.append(T.RandomHorizontalFlip())
    train_tfms.append(T.ToTensor())

    eval_tfms = [
        T.Resize((img_size_eval, img_size_eval)),
        T.ToTensor(),
    ]

    return T.Compose(train_tfms), T.Compose(eval_tfms)


class NdecFolderDataset(Dataset):
    """Load RGB tensors from a flat NDEC folder."""

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


def ndec_id_to_path(root: Path, img_id: str) -> Path:
    """Map the string identifiers used in CSVs to actual files on disk."""

    root = Path(root)
    img_id = str(img_id)

    if not img_id.lower().endswith((".jpg", ".jpeg", ".png")):
        img_id = img_id + ".jpg"

    candidate = root / img_id
    if not candidate.exists():
        raise FileNotFoundError(f"Image '{img_id}' not found under {root}")
    return candidate


class NdecPairDataset(Dataset):
    """Return (query_img, ref_img, query_id, ref_id) tuples from the ground-truth CSV."""

    def __init__(
        self,
        csv_path: Path,
        query_root: Path,
        reference_root: Path,
        query_col: str = "query_id",
        reference_col: str = "reference_id",
        transform_query: Optional[Callable] = None,
        transform_reference: Optional[Callable] = None,
        id_to_path_fn: Callable[[Path, str], Path] = ndec_id_to_path,
        drop_missing: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)
        if drop_missing:
            df = df.dropna(subset=[reference_col])
        self.df = df.reset_index(drop=True)
        self.query_root = Path(query_root)
        self.reference_root = Path(reference_root)
        self.query_col = query_col
        self.reference_col = reference_col
        self.transform_query = transform_query
        self.transform_reference = transform_reference
        self.id_to_path_fn = id_to_path_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        q_id = str(row[self.query_col])
        r_id = str(row[self.reference_col])

        q_path = self.id_to_path_fn(self.query_root, q_id)
        r_path = self.id_to_path_fn(self.reference_root, r_id)

        q_img = Image.open(q_path).convert("RGB")
        r_img = Image.open(r_path).convert("RGB")

        if self.transform_query is not None:
            q_img = self.transform_query(q_img)
        if self.transform_reference is not None:
            r_img = self.transform_reference(r_img)

        return q_img, r_img, q_id, r_id


class NegativePairDataset(Dataset):
    """Iterate over folders that contain explicit mismatched pairs."""

    def __init__(
        self,
        root: Path,
        transform_a: Optional[Callable] = None,
        transform_b: Optional[Callable] = None,
        extensions: Sequence[str] = DEFAULT_IMG_EXTS,
    ) -> None:
        self.root = Path(root)
        exts = {e.lower() for e in extensions}
        self.transform_a = transform_a
        self.transform_b = transform_b

        if not self.root.exists():
            raise FileNotFoundError(f"Negative pair root not found: {self.root}")

        self.pairs: List[Tuple[Path, Path]] = []
        for subdir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            imgs = sorted(
                c for c in subdir.iterdir() if c.is_file() and c.suffix.lower() in exts
            )
            if len(imgs) < 2:
                logger.warning("Skipping negative pair folder %s (expected 2 images)", subdir)
                continue
            self.pairs.append((imgs[0], imgs[1]))

        if not self.pairs:
            raise RuntimeError(f"No negative pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a_path, b_path = self.pairs[idx]
        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        if self.transform_a is not None:
            a_img = self.transform_a(a_img)
        if self.transform_b is not None:
            b_img = self.transform_b(b_img)

        return a_img, b_img, str(a_path.name), str(b_path.name)


def load_groundtruth(
    root: Optional[Path] = None,
    csv_name: str = DEFAULT_GROUNDTRUTH_NAME,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Load the public ground-truth CSV, optionally dropping missing references."""

    ndec_root = resolve_ndec_root(root)
    csv_path = ndec_root / csv_name
    df = pd.read_csv(csv_path)
    if drop_missing:
        df = df.dropna(subset=["reference_id"]).reset_index(drop=True)
    return df


def get_query_dataset(
    root: Optional[Path] = None,
    transform: Optional[Callable] = None,
    subdir: Optional[Path] = None,
) -> NdecFolderDataset:
    """Return the query dataset (single folder)."""

    ndec_root = resolve_ndec_root(root)
    query_dir = ndec_root / (subdir or DEFAULT_QUERY_SUBDIR)
    return NdecFolderDataset(query_dir, transform=transform)


def get_reference_dataset(
    root: Optional[Path] = None,
    transform: Optional[Callable] = None,
    subdir: Optional[Path] = None,
) -> NdecFolderDataset:
    """Return the reference dataset (single folder)."""

    ndec_root = resolve_ndec_root(root)
    reference_dir = ndec_root / (subdir or DEFAULT_REFERENCE_SUBDIR)
    return NdecFolderDataset(reference_dir, transform=transform)


def get_negative_pair_dataset(
    root: Optional[Path] = None,
    transform_a: Optional[Callable] = None,
    transform_b: Optional[Callable] = None,
    subdir: Optional[Path] = None,
) -> NegativePairDataset:
    """Return the negative pair dataset rooted at ``negative_pair``."""

    ndec_root = resolve_ndec_root(root)
    negative_dir = ndec_root / (subdir or DEFAULT_NEGATIVE_SUBDIR)
    return NegativePairDataset(negative_dir, transform_a, transform_b)


def get_pair_dataset(
    root: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    transform_query: Optional[Callable] = None,
    transform_reference: Optional[Callable] = None,
    drop_missing: bool = True,
) -> NdecPairDataset:
    """Instantiate the query/reference pair dataset described by the CSV."""

    ndec_root = resolve_ndec_root(root)
    csv_file = Path(csv_path) if csv_path is not None else ndec_root / DEFAULT_GROUNDTRUTH_NAME
    query_root = ndec_root / DEFAULT_QUERY_SUBDIR
    reference_root = ndec_root / DEFAULT_REFERENCE_SUBDIR

    return NdecPairDataset(
        csv_path=csv_file,
        query_root=query_root,
        reference_root=reference_root,
        transform_query=transform_query,
        transform_reference=transform_reference,
        drop_missing=drop_missing,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """Wrapper that mirrors the DISC21 helper for consistency."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_default_datasets(
    config: Optional[NdecDataConfig] = None,
) -> Tuple[NdecFolderDataset, NdecFolderDataset, NdecPairDataset, NegativePairDataset]:
    """Return (query_ds, reference_ds, positive_pairs, negative_pairs)."""

    cfg = config or NdecDataConfig()
    query_tfms, eval_tfms = build_transforms(cfg.img_size_train, cfg.img_size_eval)

    query_ds = get_query_dataset(cfg.root, transform=query_tfms, subdir=cfg.query_subdir)
    reference_ds = get_reference_dataset(
        cfg.root, transform=eval_tfms, subdir=cfg.reference_subdir
    )
    pair_ds = get_pair_dataset(
        cfg.root,
        csv_path=cfg.resolve_groundtruth_csv(),
        transform_query=eval_tfms,
        transform_reference=eval_tfms,
        drop_missing=cfg.drop_missing_groundtruth,
    )
    negative_ds = get_negative_pair_dataset(
        cfg.root,
        transform_a=query_tfms,
        transform_b=eval_tfms,
        subdir=cfg.negative_pair_subdir,
    )

    return query_ds, reference_ds, pair_ds, negative_ds


def build_default_loaders(
    config: Optional[NdecDataConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Return dataloaders for (queries, references, positive pairs, negative pairs)."""

    cfg = config or NdecDataConfig()
    query_tfms, eval_tfms = build_transforms(cfg.img_size_train, cfg.img_size_eval)

    query_ds = get_query_dataset(cfg.root, transform=query_tfms, subdir=cfg.query_subdir)
    reference_ds = get_reference_dataset(
        cfg.root, transform=eval_tfms, subdir=cfg.reference_subdir
    )
    pair_ds = get_pair_dataset(
        cfg.root,
        csv_path=cfg.resolve_groundtruth_csv(),
        transform_query=eval_tfms,
        transform_reference=eval_tfms,
        drop_missing=cfg.drop_missing_groundtruth,
    )
    negative_ds = get_negative_pair_dataset(
        cfg.root,
        transform_a=query_tfms,
        transform_b=query_tfms,
        subdir=cfg.negative_pair_subdir,
    )

    query_loader = create_dataloader(
        query_ds,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    reference_loader = create_dataloader(
        reference_ds,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    pair_loader = create_dataloader(
        pair_ds,
        batch_size=cfg.batch_size_pairs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    negative_loader = create_dataloader(
        negative_ds,
        batch_size=cfg.batch_size_pairs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return query_loader, reference_loader, pair_loader, negative_loader


__all__ = [
    "NegativePairDataset",
    "NdecDataConfig",
    "NdecFolderDataset",
    "NdecPairDataset",
    "build_default_datasets",
    "build_default_loaders",
    "build_transforms",
    "create_dataloader",
    "get_negative_pair_dataset",
    "get_pair_dataset",
    "get_query_dataset",
    "get_reference_dataset",
    "load_groundtruth",
    "ndec_id_to_path",
    "resolve_ndec_root",
]


def _log_dataset_stats() -> None:
    """Helper for manual sanity checks."""

    try:
        query_ds, reference_ds, pair_ds, negative_ds = build_default_datasets()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to enumerate NDEC datasets: %s", exc)
        return

    logger.info("Queries: %s", len(query_ds))
    logger.info("References: %s", len(reference_ds))
    logger.info("Pairs (positive): %s", len(pair_ds))
    logger.info("Pairs (negative): %s", len(negative_ds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log_dataset_stats()
