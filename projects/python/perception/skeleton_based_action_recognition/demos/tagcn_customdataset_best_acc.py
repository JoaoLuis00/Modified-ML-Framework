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
        num_frames=300,
        num_point=46,
        experiment_name="tagcn_custom_00subframes",
        dataset_name="custom",
        num_class=5,
        graph_type="custom",
        device="cpu",
        checkpoint_after_iter=10,
        val_batch_size=32, #7
        batch_size=32, #10
        epochs=1,
        in_channels=3,
        num_person=1,
        lr=0.1,
        method_name='tagcn',
        num_subframes=100,
    )

    # Define datasets path
    data_path = tmp_path / "data"
    learner.best_acc
    train_ds_path = data_path / "custom"
    val_ds_path = data_path / "custom"

    train_ds = ExternalDataset(path=str(train_ds_path), dataset_type="custom")

    val_ds = ExternalDataset(path=str(val_ds_path), dataset_type="custom")
    epoch = 0
    while True:
        
        if learner.best_acc < 0.92:
            learner.fit(
                dataset=train_ds,
                val_dataset=val_ds,
                train_data_filename="train_joints.npy",
                train_labels_filename="train_labels.pkl",
                val_data_filename="val_joints.npy",
                val_labels_filename="val_labels.pkl",
                skeleton_data_type="joint",
                verbose=False,
                silent=True
            )
            epoch += 1
        else:
            break
        
    results = learner.eval(val_ds,wrong_file=True, save_score=True)
    # print("Evaluation results: ", results)
    with open(f'{Path(__file__).parent}/statistics/{learner.experiment_name}.txt', 'w') as f:
        f.write(str(results))
        
    learner.optimize(do_constant_folding=True)
    
    save_path = Path(__file__).parent/'models'
    
    learner.save(path=str(save_path),model_name=f'{learner.experiment_name}_optimized')

if __name__ == "__main__":
    main()