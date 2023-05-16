import torch

from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    SpatioTemporalGCNLearner,
    _MODEL_NAMES,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path


def main():
    tmp_path = Path(__file__).parent / "tmp"

    # Define learner
    learner = SpatioTemporalGCNLearner(
        # device=args.device,
        temp_path=str(tmp_path),
        # batch_size=args.batch_size,
        # backbone=args.backbone,
        num_workers=8,
        num_person=1,
        num_frames=150,
        num_subframes=50,
        num_point=46,
        experiment_name="stgcn_custom",
        dataset_name="custom",
        num_class=5,
        graph_type="custom",
        device="cpu",
        checkpoint_after_iter=10,
        val_batch_size=3,
        batch_size=4,
        lr=0.01,
        epochs=250,
    )

    # Define datasets path
    data_path = tmp_path / "data"

    train_ds_path = data_path / "custom"
    val_ds_path = data_path / "custom"

    train_ds = ExternalDataset(path=str(train_ds_path), dataset_type="custom")

    val_ds = ExternalDataset(path=str(val_ds_path), dataset_type="custom")

    learner.fit(
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


if __name__ == "__main__":
    main()
