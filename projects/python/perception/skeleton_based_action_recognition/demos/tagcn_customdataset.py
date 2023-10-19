import torch
import os,sys
from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    SpatioTemporalGCNLearner,
    _MODEL_NAMES,
)

from opendr.engine.datasets import ExternalDataset
from pathlib import Path

KEYPOINTS = 24

epochs = 70
lr = 0.1
subframes = 75

datatype = 'with_multi_requests'

experiment_name = f"tagcn_{epochs}epochs_{lr}lr_{subframes}subframes_dropafterepoch304060_batch128"
#experiment_name = f"test"
tmp_path = Path(__file__).parent / "models" / str(datatype) / str(experiment_name) / "model"

# TENTAR COM OUTROS VALORES DE DROP E MUDAR PARA O DATA_TYPE COM OS REQUESTS SO PARA A FRENTE
# AUMENTAR UM POUCO MAIS AS EPOCHS E EXPERIMENTAR COM OS MERDAS DOS DROPS
# should try to mess with lr=0.15 subframes=100 drop=[50,70,90]/[50,70,90,110] ou qualquer coisa assim e ir vendo e avaliando a loss

def main():

    # Define learner
    learner = SpatioTemporalGCNLearner(
        # device=args.device,
        # batch_size=args.batch_size,
        # backbone=args.backbone,
        num_workers=8,
        num_frames=300,
        num_point=KEYPOINTS,
        dataset_name="custom",
        num_class=6,
        graph_type="custom",
        device="cpu",
        checkpoint_after_iter=10,
        val_batch_size=64, 
        batch_size=128, 
        epochs=epochs,
        in_channels=3,
        num_person=1,
        lr=lr,
        method_name='tagcn',
        num_subframes=subframes,
        experiment_name=experiment_name,
        temp_path = str(tmp_path),
        drop_after_epoch=[30,40,60]
    )

    folder_path = Path(__file__).parent/'models'/str(datatype)/str(learner.experiment_name)

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
    
    results = learner.eval(val_ds,result_file=os.path.join(folder_path, 'results.txt'),wrong_file=os.path.join(folder_path,'wrong.txt') )
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
