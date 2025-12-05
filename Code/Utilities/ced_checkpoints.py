"""
Checkpoint Management for CEDetector
====================================

Utilities for saving and loading model checkpoints with training state.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class TrainingState:
    """
    Training state for resumption across epochs.

    Tracks progress for both DISC21 and NDEC training phases,
    including loss history, accuracy history, and early stopping counters.
    """
    epoch_disc21: int = 0
    epoch_ndec: int = 0
    best_disc21_loss: float = float("inf")
    best_ndec_loss: float = float("inf")
    disc21_bad_epochs: int = 0
    ndec_bad_epochs: int = 0
    disc21_losses: List[float] = field(default_factory=list)
    ndec_losses: List[float] = field(default_factory=list)
    disc21_accuracies: List[float] = field(default_factory=list)  # BCE accuracy per epoch

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingState':
        """Create from dictionary."""
        return cls(**d)


class CheckpointManager:
    """
    Manage model checkpoints with training state persistence.

    Features:
    - Save/load model and optimizer state
    - Persist training progress for resumption
    - Rank-aware saving (only rank 0 saves)
    - Automatic directory creation
    """

    def __init__(self, checkpoint_path: Path, rank: int):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            rank: Process rank (only rank 0 saves checkpoints)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.rank = rank

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_state: TrainingState,
    ):
        """
        Save checkpoint (only on rank 0).

        Args:
            model: Model to save (should be unwrapped from DDP)
            optimizer: Optimizer to save
            training_state: Training state to persist
        """
        if self.rank != 0:
            return

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": training_state.to_dict(),
        }

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.checkpoint_path)
        print(f"[Rank 0] Saved checkpoint to {self.checkpoint_path}")

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[TrainingState]:
        """
        Load checkpoint and return training state.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            TrainingState if checkpoint exists, None otherwise
        """
        if not self.exists():
            if self.rank == 0:
                print(f"[Checkpoint] No existing checkpoint at {self.checkpoint_path}, starting fresh.")
            return None

        map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=map_location)

        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        training_state = TrainingState.from_dict(ckpt.get("state", {}))

        if self.rank == 0:
            print(f"[Checkpoint] Loaded checkpoint from {self.checkpoint_path}")
            print(
                f"[Checkpoint] Resuming from DISC21 epoch {training_state.epoch_disc21} "
                f"and NDEC epoch {training_state.epoch_ndec}"
            )

        return training_state

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()


# Legacy functions for backwards compatibility
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    training_state: Dict,
):
    """Legacy wrapper - prefer using CheckpointManager class."""
    if rank != 0:
        return
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": training_state,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[Rank 0] Saved checkpoint to {path}")


def load_checkpoint_if_available(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    rank: int,
    training_state: Dict,
) -> Dict:
    """Legacy wrapper - prefer using CheckpointManager class."""
    if not os.path.exists(path):
        if rank == 0:
            print(f"[Checkpoint] No existing checkpoint at {path}, starting fresh.")
        return training_state

    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    loaded_state = ckpt.get("state", {})
    training_state.update(loaded_state)

    if rank == 0:
        print(f"[Checkpoint] Loaded checkpoint from {path}")
        print(
            f"[Checkpoint] Resuming from DISC21 epoch {training_state['epoch_disc21']} "
            f"and NDEC epoch {training_state['epoch_ndec']}"
        )
    return training_state


__all__ = [
    'TrainingState',
    'CheckpointManager',
    'save_checkpoint',  # Legacy
    'load_checkpoint_if_available',  # Legacy
]
