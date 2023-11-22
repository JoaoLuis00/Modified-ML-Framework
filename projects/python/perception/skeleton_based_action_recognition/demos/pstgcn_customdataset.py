import torch
import os
from opendr.perception.skeleton_based_action_recognition.progressive_spatio_temporal_gcn_learner import (
    ProgressiveSpatioTemporalGCNLearner,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path

KEYPOINTS = 24

epochs = 30
lr = 0.1
subframes = 100

datatype = 'final_v2'
#datatype = 'modified_val_data/augmented_data_noise'

experiment_name = f"pstgcn_{epochs}epochs_{lr}lr_dropafterepoch5060_batch15"
#experiment_name = f"test"
tmp_path = Path(__file__).parent / "models" / str(datatype) / str(experiment_name) / "model"

def main():

    # Define learner
    learner = ProgressiveSpatioTemporalGCNLearner(
        batch_size=15,
        device='cuda',
        epochs=epochs,
        checkpoint_after_iter=10,
        val_batch_size=64,
        dataset_name="custom",
        experiment_name=experiment_name,
        blocksize=20,
        numblocks=9,
        numlayers=1,
        topology=[],
        layer_threshold=1e-4,
        block_threshold=1e-4,
        graph_type='custom',
        num_class=4,
        num_point=KEYPOINTS,
        in_channels=3,
        temp_path=str(tmp_path),
        num_workers=8,
        num_person=1,
        lr=lr,
        drop_after_epoch=[50,60]
    )
    
    folder_path = Path(__file__).parent/'models'/str(datatype)/str(learner.experiment_name)

    if not os.path.isdir(Path(__file__).parent/'models'/str(datatype)):
        os.mkdir(Path(__file__).parent/'models'/str(datatype))
    
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    # Define datasets path
    data_path = Path(__file__).parent / "data" / str(datatype)
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
        logging_path=str(folder_path)
    )

    results = learner.eval(val_ds,result_file=os.path.join(folder_path, 'results.txt'),wrong_file=os.path.join(folder_path,'wrong.txt') )
    # print("Evaluation results: ", results)
    with open(os.path.join(folder_path, f'{learner.experiment_name}.txt'), 'w') as f:
        f.write(str(ret))
        f.write(str(results))

    #learner.optimize(do_constant_folding=True)
    
    save_model_path = folder_path/'model'
    
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
    
    learner.save(path=str(save_model_path),model_name=f'{learner.experiment_name}')

if __name__ == "__main__":
    main()
