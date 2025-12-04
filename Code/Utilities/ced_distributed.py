"""
Distributed Training Utilities for CEDetector
=============================================

DDP (DistributedDataParallel) management utilities for multi-GPU training.
"""

import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPManager:
    """
    Manage distributed training state and operations.

    Handles:
    - Initialization from environment variables (set by torchrun)
    - Device management
    - Model wrapping
    - Synchronization primitives
    - Main process detection
    """

    def __init__(self):
        """Initialize distributed training from environment variables."""
        self.rank, self.world_size, self.local_rank = self._setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

    @staticmethod
    def _setup_distributed() -> Tuple[int, int, int]:
        """
        Initialize torch.distributed using environment variables set by torchrun.

        Returns:
            rank: global rank
            world_size: total number of processes
            local_rank: index of GPU on this node to use in this process
        """
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Fallback: single-process, single-GPU (no true DDP)
            rank, world_size, local_rank = 0, 1, 0
            # torch.distributed uses env:// rendezvous, so provide sane defaults
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return rank, world_size, local_rank

    def cleanup(self):
        """Cleanup distributed process group."""
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def wrap_model(self, model: torch.nn.Module, find_unused_parameters: bool = False) -> DDP:
        """
        Wrap model in DistributedDataParallel.

        Args:
            model: Model to wrap
            find_unused_parameters: Whether to find unused parameters (slower)

        Returns:
            DDP-wrapped model
        """
        return DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def barrier(self):
        """Synchronize all processes."""
        dist.barrier()

    def all_reduce_scalar(self, value: float) -> float:
        """
        Average scalar value across all processes.

        Args:
            value: Local scalar value

        Returns:
            Global average across all processes
        """
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.world_size

    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Sum tensor across all processes (in-place).

        Args:
            tensor: Tensor to reduce

        Returns:
            Reduced tensor
        """
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def broadcast_scalar(self, value: float, src: int = 0) -> float:
        """
        Broadcast scalar from source rank to all ranks.

        Args:
            value: Value to broadcast (only meaningful on src rank)
            src: Source rank

        Returns:
            Broadcasted value
        """
        tensor = torch.tensor(value, device=self.device)
        dist.broadcast(tensor, src=src)
        return tensor.item()


# Legacy functions for backwards compatibility
def setup_distributed() -> Tuple[int, int, int]:
    """Legacy wrapper - prefer using DDPManager class."""
    manager = DDPManager()
    return manager.rank, manager.world_size, manager.local_rank


def cleanup_distributed():
    """Legacy wrapper - prefer using DDPManager.cleanup()."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    'DDPManager',
    'setup_distributed',  # Legacy
    'cleanup_distributed',  # Legacy
]
