import torch
import os,sys
from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    SpatioTemporalGCNLearner,
    _MODEL_NAMES,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path

KEYPOINTS = 46

epochs = 70
lr = 0.1
subframes = 75

datatype = 'final_atualizado_fullsize'
#datatype = 'modified_val_data/augmented_data_noise'

experiment_name = f"tagcn_{epochs}epochs_{lr}lr_{subframes}subframes_dropafterepoch5060_batch15"
#experiment_name = f"test"
tmp_path = Path(__file__).parent / "models" / str(datatype) / str(experiment_name) / "model"

def main():

    # Define learner
    learner = SpatioTemporalGCNLearner(
        # device=args.device,
        # batch_size=args.batch_size,
        # backbone=args.backbone,
        num_workers=8,
        num_frames=250,
        num_point=KEYPOINTS,
        dataset_name="custom",
        num_class=6,
        graph_type="custom",
        device="cuda",
        checkpoint_after_iter=10,
        val_batch_size=64, 
        batch_size=30, 
        epochs=epochs,
        in_channels=3,
        num_person=1,
        lr=lr,
        method_name='tagcn',
        num_subframes=subframes,
        experiment_name=experiment_name,
        temp_path = str(tmp_path),
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

    ret = learner.fit(
        dataset=train_ds,
        val_dataset=val_ds,
        train_data_filename="train_joints.npy",
        train_labels_filename="train_labels.pkl",
        val_data_filename="val_joints.npy",
        val_labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
        logging_path=str(folder_path)
    )
    
    # ret = learner.fit(
    #     dataset=train_ds,
    #     val_dataset=val_ds,
    #     train_data_filename="augmented_train_data_noise.npy",
    #     train_labels_filename="augmented_train_labels_noise.pkl",
    #     val_data_filename="val_joints.npy",
    #     val_labels_filename="val_labels.pkl",
    #     skeleton_data_type="joint",
    #     logging_path=str(folder_path)
    # )
    
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
