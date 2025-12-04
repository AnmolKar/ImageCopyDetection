"""
CEDetector Utilities Package
============================

Modular utilities for CEDetector training and evaluation.
"""

# Model architecture
from .ced_models import (
    DinoV3Backbone,
    CEDFeatureAggregator,
    TransformerBlock,
    CopyEditClassifier,
    CEDModel,
)

# Loss functions
from .ced_losses import (
    nt_xent_loss,
    kozachenko_leonenko_loss,
    similarity_kl_loss,
    multi_similarity_loss,
    asl_loss,
    bce_loss,
)

# Distributed training
from .ced_distributed import (
    DDPManager,
    setup_distributed,
    cleanup_distributed,
)

# Checkpoint management
from .ced_checkpoints import (
    TrainingState,
    CheckpointManager,
    save_checkpoint,
    load_checkpoint_if_available,
)

# Augmentations
from .ced_augmentations import (
    CEDAugmentationPipeline,
    build_ced_transforms,
    check_dependencies,
    AUGLY_AVAILABLE,
    ALBUMENTATIONS_AVAILABLE,
)

# Evaluation
from .ced_evaluation import (
    compute_descriptors_for_loader,
    evaluate_retrieval,
    cosine_similarity,
    compute_muap_and_rp90,
    benchmark_inference,
    ced_two_stage_eval,
    build_ref_index_map,
    normalize_id,
    make_six_patches,
    make_six_patches_batch,
)

# Data loaders
from .disc21_loader import (
    Disc21DataConfig,
    Disc21FolderDataset,
    QueryReferencePairDataset,
    build_transforms,
    get_train_dataset,
    get_reference_dataset,
    get_query_dataset,
    get_pair_dataset,
    create_dataloader,
    load_groundtruth,
    disc21_id_to_path,
    build_default_datasets,
    build_default_loaders,
)

from .ndec_loader import (
    NdecDataConfig,
    build_default_loaders as build_ndec_loaders,
    load_groundtruth as load_ndec_groundtruth,
)

__all__ = [
    # Models
    'DinoV3Backbone',
    'CEDFeatureAggregator',
    'TransformerBlock',
    'CopyEditClassifier',
    'CEDModel',
    # Losses
    'nt_xent_loss',
    'kozachenko_leonenko_loss',
    'similarity_kl_loss',
    'multi_similarity_loss',
    'asl_loss',
    'bce_loss',
    # Distributed
    'DDPManager',
    'setup_distributed',
    'cleanup_distributed',
    # Checkpoints
    'TrainingState',
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint_if_available',
    # Augmentations
    'CEDAugmentationPipeline',
    'build_ced_transforms',
    'check_dependencies',
    'AUGLY_AVAILABLE',
    'ALBUMENTATIONS_AVAILABLE',
    # Evaluation
    'compute_descriptors_for_loader',
    'evaluate_retrieval',
    'cosine_similarity',
    'compute_muap_and_rp90',
    'benchmark_inference',
    'ced_two_stage_eval',
    'build_ref_index_map',
    'normalize_id',
    'make_six_patches',
    'make_six_patches_batch',
    # Data loaders
    'Disc21DataConfig',
    'Disc21FolderDataset',
    'QueryReferencePairDataset',
    'build_transforms',
    'get_train_dataset',
    'get_reference_dataset',
    'get_query_dataset',
    'get_pair_dataset',
    'create_dataloader',
    'load_groundtruth',
    'disc21_id_to_path',
    'build_default_datasets',
    'build_default_loaders',
    'NdecDataConfig',
    'build_ndec_loaders',
    'load_ndec_groundtruth',
]
