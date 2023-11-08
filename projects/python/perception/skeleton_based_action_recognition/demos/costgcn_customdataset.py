import torch

from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    CoSTGCNLearner, _MODEL_NAMES
)
from opendr.engine.datasets import ExternalDataset
from pathlib import Path
import os

KEYPOINTS = 24

epochs = 50
lr = 0.1
subframes = 100

datatype = 'final_v2'
#datatype = 'modified_val_data/augmented_data_noise'

experiment_name = f"costgcn_{epochs}epochs_{lr}lr_dropafterepoch5060_batch30"
#experiment_name = f"test"
tmp_path = Path(__file__).parent / "models" / str(datatype) / str(experiment_name) / "model"

def main():

    # Define learner
    learner = CoSTGCNLearner(
        device='cuda',
        temp_path=str(tmp_path),
        batch_size=2,
        backbone='costgcn',
        num_workers=8,
        graph_type='custom',
        num_point=KEYPOINTS,
        num_classes=4,
        sequence_len=300,
        iters=epochs,
        num_person=1,
        
    )

    folder_path = Path(__file__).parent/'models'/str(datatype)/str(experiment_name)

    if not os.path.isdir(Path(__file__).parent/'models'/str(datatype)):
        os.mkdir(Path(__file__).parent/'models'/str(datatype))
    
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # Define datasets path
    data_path = Path(__file__).parent / "data" / str(datatype)
    train_ds_path = data_path
    val_ds_path = data_path


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

    ret = learner.fit(dataset=train_ds, val_dataset=val_ds)


    results = learner.eval(val_ds)
    # print("Evaluation results: ", results)
    with open(os.path.join(folder_path, f'{experiment_name}.txt'), 'w') as f:
        f.write(str(ret))
        f.write(str(results))

    #learner.optimize(do_constant_folding=True)
    
    save_model_path = folder_path/'model'
    
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
    
    learner.save(path=str(save_model_path),model_name=f'{experiment_name}')
    
#test commment to git source control

if __name__ == "__main__":
    main()
