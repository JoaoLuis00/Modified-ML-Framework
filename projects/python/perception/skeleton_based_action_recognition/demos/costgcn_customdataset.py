import torch

from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    CoSTGCNLearner, _MODEL_NAMES
)
from opendr.engine.datasets import ExternalDataset
from pathlib import Path


def main():
    tmp_path = Path(__file__).parent / "tmp"

    # Define learner
    learner = CoSTGCNLearner(
        device='cpu',
        temp_path=str(tmp_path),
        batch_size=1,
        backbone='costgcn',
        num_workers=0,
        graph_type='custom',
        num_point=46,
        num_person=1,
        num_classes=5,
        sequence_len=150,
        iters=250,
        lr=0.1,
        weight_decay=0.0001
    )

    # Define datasets
    data_path = tmp_path / "data"
    
    train_ds_path = data_path / "custom"
    val_ds_path = data_path / "custom"


    train_ds = learner._prepare_dataset(
        ExternalDataset(path=str(train_ds_path), dataset_type="custom"),
        data_filename="train_joints.npy",
        labels_filename="train_labels.pkl",
        skeleton_data_type="joint",
        phase="train",
        verbose=False,
    )

    val_ds = learner._prepare_dataset(
        ExternalDataset(path=str(val_ds_path), dataset_type="custom"),
        data_filename="val_joints.npy",
        labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
        phase="val",
        verbose=False,
    )

    learner.fit(dataset=train_ds, val_dataset=val_ds)


    # results = learner.eval(val_ds)
    # print("Evaluation results: ", results)



if __name__ == "__main__":
    main()