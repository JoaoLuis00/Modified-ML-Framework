import torch
import os
from opendr.perception.skeleton_based_action_recognition.progressive_spatio_temporal_gcn_learner import (
    ProgressiveSpatioTemporalGCNLearner,
    _MODEL_NAMES,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path

KEYPOINTS = 24

experiment_name = "pstgcn_custom"

def main():
    tmp_path = Path(__file__).parent / "tmp"

    # Define learner
    learner = ProgressiveSpatioTemporalGCNLearner(
        batch_size=15,
        epochs=50,
        checkpoint_after_iter=10,
        val_batch_size=5,
        dataset_name="custom",
        experiment_name=experiment_name,
        blocksize=20,
        numblocks=1,
        numlayers=1,
        topology=[],
        layer_threshold=1e-4,
        block_threshold=1e-4,
        graph_type='custom',
        num_class=5,
        num_point=KEYPOINTS,
        in_channels=3,
        tmp_path = Path(__file__).parent/'models'/learner.experiment_name/'model'
    )

    folder_path = Path(__file__).parent/'models'/learner.experiment_name

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    # Define datasets path
    data_path = Path(__file__).parent / "data" / "pkl_files"
    train_ds_path = data_path
    val_ds_path = data_path

    train_ds = ExternalDataset(path=str(train_ds_path), dataset_type="custom")

    val_ds = ExternalDataset(path=str(val_ds_path), dataset_type="custom")

    ret = learner.network_builder(
        dataset=train_ds,
        val_dataset=val_ds,
        train_data_filename="train_joints.npy",
        train_labels_filename="train_labels.pkl",
        val_data_filename="val_joints.npy",
        val_labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
    )

    results = learner.eval(val_ds,result_file=os.path.join(folder_path, 'results.txt') )
    # print("Evaluation results: ", results)
    with open(os.path.join(folder_path, f'{learner.experiment_name}.txt'), 'w') as f:
        f.write(str(ret))
        f.write(str(results))

    learner.optimize(do_constant_folding=True)
    
    save_model_path = folder_path/'model'
    
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
    
    learner.save(path=str(save_model_path),model_name=f'{learner.experiment_name}')

if __name__ == "__main__":
    main()
