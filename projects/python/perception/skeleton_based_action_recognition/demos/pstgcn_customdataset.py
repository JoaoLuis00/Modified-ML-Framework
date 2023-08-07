import torch

from opendr.perception.skeleton_based_action_recognition.progressive_spatio_temporal_gcn_learner import (
    ProgressiveSpatioTemporalGCNLearner,
    _MODEL_NAMES,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path


def main():
    tmp_path = Path(__file__).parent / "tmp"

    # Define learner
    learner = ProgressiveSpatioTemporalGCNLearner(
        temp_path="./parent_dir",
        batch_size=64,
        epochs=65,
        checkpoint_after_iter=10,
        val_batch_size=128,
        dataset_name="custom",
        experiment_name="pstgcn_custom",
        blocksize=20,
        numblocks=1,
        numlayers=1,
        topology=[],
        layer_threshold=1e-4,
        block_threshold=1e-4,
        graph_type='custom',
        num_class=4,
        num_point=46,
        
    )

    # Define datasets path
    data_path = tmp_path / "data"

    train_ds_path = data_path / "custom"
    val_ds_path = data_path / "custom"

    train_ds = ExternalDataset(path=str(train_ds_path), dataset_type="custom")

    val_ds = ExternalDataset(path=str(val_ds_path), dataset_type="custom")

    learner.network_builder(
        dataset=train_ds,
        val_dataset=val_ds,
        train_data_filename="train_joints.npy",
        train_labels_filename="train_labels.pkl",
        val_data_filename="val_joints.npy",
        val_labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
    )

    # results = learner.eval(val_ds)
    # print("Evaluation results: ", results)

    learner.optimize(do_constant_folding=True)

    save_path = Path(__file__).parent / "models"

    learner.save(path=str(save_path), model_name="pstgcn_optimized")


if __name__ == "__main__":
    main()
