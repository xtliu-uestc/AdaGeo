# AdaGeo
![](https://img.shields.io/badge/python-3.12.11-green)
![](https://img.shields.io/badge/pytorch-2.8.0-green)
![](https://img.shields.io/badge/cudatoolkit-12.6-green)
![](https://img.shields.io/badge/cudnn-9.10.2-green)

This folder provides a reference implementation of **AdaGeo**, as described in the paper: "Adapting After Deployment: Towards Robust Source-Free IP Geolocation via Test-Time Adaptation".


## Basic Usage

### Requirements

The code was tested with `python 3.12.11, `pytorch 2.8.0+cu126`,  `cudatoolkit 12.6`, and `cudnn 9.10.2`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name AdaGeo python=3.12.11

# activate environment
conda activate AdaGeo

# install pytorch & cudatoolkit & cuDNN
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia

# install other requirements
conda install numpy pandas
pip install scikit-learn
```

### Run the code

```shell
# Open the "AdaGeo" folder
cd AdaGeo

# data preprocess (executing IP clustering). 
python preprocess.py --dataset "New_York"
python preprocess.py --dataset "Los_Angeles"
python preprocess.py --dataset "Shanghai"

# Source city pretraining
python main.py --stage pretrain --dataset New_York --dim_in 30 --seed 2022 --lr 5e-3
python main.py --stage pretrain --dataset Los_Angeles --dim_in 30 --seed 2022 --lr 3e-3
python main.py --stage pretrain --dataset Shanghai --dim_in 51 --seed 2022 --lr 1e-3

#Cross-city transfer examples (run contrastive then pseudo on the target city; source pretrain must exist):

#New_York → Los_Angeles
python main.py --stage contrastive --dataset Los_Angeles --dim_in 30 \
    --pretrained_model asset/model/New_York_pretrain_final.pt \
    --seed 2022 --lr 1e-7 --lambda_1 2 --eta_1 0.5 --eta_2 0.5 --saved_epoch 30 --intere 3
python main.py --stage pseudo --dataset Los_Angeles --dim_in 30 \
    --constrastive_model asset/model/Los_Angeles_contrastive_final.pt \
    --pseudolabel asset/model/Los_Angeles_pseudolabels.pt \
    --seed 2022 --lr 5e-6 --quantile 0.03

#Los_Angeles → New_York
python main.py --stage contrastive --dataset New_York --dim_in 30 \
    --pretrained_model asset/model/Los_Angeles_pretrain_final.pt \
    --seed 2022 --lr 1e-7 --lambda_1 2 --eta_1 0.1 --eta_2 0.3 --saved_epoch 15 --intere 3
python main.py --stage pseudo --dataset New_York --dim_in 30 \
    --constrastive_model asset/model/New_York_contrastive_final.pt \
    --pseudolabel asset/model/New_York_pseudolabels.pt \
    --seed 2022 --lr 5e-6 --quantile 0.07

#New_York → Shanghai
python main.py --stage contrastive --dataset Shanghai --dim_in 51 \
    --pretrained_model asset/model/New_York_pretrain_final.pt \
    --seed 2022 --lr 1e-4 --lambda_1 2 --eta_1 0.1 --eta_2 0.3 --saved_epoch 15 --intere 3
python main.py --stage pseudo --dataset Shanghai --dim_in 51 \
    --constrastive_model asset/model/Shanghai_contrastive_final.pt \
    --pseudolabel asset/model/Shanghai_pseudolabels.pt \
    --seed 2022 --lr 5e-6 --quantile 0.01

#Shanghai → New_York
python main.py --stage contrastive --dataset New_York --dim_in 30 \
    --pretrained_model asset/model/Shanghai_pretrain_final.pt \
    --seed 2022 --lr 1e-6 --lambda_1 5 --eta_1 0.3 --eta_2 0.1 --saved_epoch 15 --intere 3
python main.py --stage pseudo --dataset New_York --dim_in 30 \
    --constrastive_model asset/model/New_York_contrastive_final.pt \
    --pseudolabel asset/model/New_York_pseudolabels.pt \
    --seed 2022 --lr 5e-6 --quantile 0.85

#Los_Angeles → Shanghai
python main.py --stage contrastive --dataset Shanghai --dim_in 51 \
    --pretrained_model asset/model/Los_Angeles_pretrain_final.pt \
    --seed 2022 --lr 1e-4 --lambda_1 2 --eta_1 0.1 --eta_2 0.3 --saved_epoch 15 --intere 3
python main.py --stage pseudo --dataset Shanghai --dim_in 51 \
    --constrastive_model asset/model/Shanghai_contrastive_final.pt \
    --pseudolabel asset/model/Shanghai_pseudolabels.pt \
    --seed 2022 --lr 5e-6 --quantile 0.05

#Shanghai → Los_Angeles
python main.py --stage contrastive --dataset Los_Angeles --dim_in 30 \
    --pretrained_model asset/model/Shanghai_pretrain_final.pt \
    --seed 2022 --lr 1e-7 --lambda_1 2 --eta_1 0.1 --eta_2 0.3 --saved_epoch 30 --intere 3
python main.py --stage pseudo --dataset Los_Angeles --dim_in 30 \
    --constrastive_model asset/model/Los_Angeles_contrastive_final.pt \
    --pseudolabel asset/model/Los_Angeles_pseudolabels.pt \
    --seed 2022 --lr 5e-4 --quantile 0.7
```



## The description of hyperparameters used in main.py

| Hyperparameter       | Description                                                                 |
| :------------------- | --------------------------------------------------------------------------- |
| seed                 | Random seed for reproducibility                                             |
| stage                | Training stage: "pretrain" (supervised pretraining), "contrastive" (self-supervised contrastive learning), or "pseudo" (semi-supervised fine-tuning with pseudo-labels) |
| dataset              | Dataset to use: "Shanghai", "New_York", or "Los_Angeles"                     |
| pretrained_model     | Path to the pre-trained model checkpoint (loaded in contrastive and pseudo stages for transfer learning) |
| constrastive_model   | Path to the contrastive-stage model checkpoint (loaded in pseudo stage)     |
| pseudolabel          | Path to the saved pseudo-label checkpoints (used in pseudo stage for high-confidence sample filtering) |
| dim_in               | Input feature dimension (51 for Shanghai, 30 for New_York and Los_Angeles)  |
| dim_med              | Hidden dimension in the prediction MLP (used when c_mlp=False)              |
| dim_z                | Dimension of attention keys/queries/values and intermediate representations |
| c_mlp                | Whether to use collaborative attention-based MLP (True) or simple MLP (False) for coordinate prediction |
| beta1                | Adam optimizer beta1 (momentum term)                                        |
| beta2                | Adam optimizer beta2 (RMSprop term)                                         |
| lr                   | Initial learning rate                                                       |
| harved_epoch         | Number of consecutive epochs without improvement before halving the learning rate |
| early_stop_epoch     | Number of consecutive epochs without improvement before triggering early stopping |
| eta_1                | Augmentation strength (noise magnitude) for the first view in contrastive stage |
| eta_2                | Augmentation strength (noise magnitude) for the second view in contrastive stage (usually stronger) |
| lambda_1             | Weight of the geographic boundary constraint loss in the contrastive stage  |
| mm                   | Initial momentum coefficient for target network moving average update in contrastive stage |
| intere               | Interval (in epochs) for saving pseudo-label checkpoints during contrastive training |
| saved_epoch          | Total number of training epochs in the contrastive stage                     |
| quantile             | Quantile threshold for selecting stable pseudo-labels (lower values keep only the most stable samples) in pseudo stage |


## Folder Structure

```tex
└── AdaGeo
	├── datasets # Contains three large-scale real-world street-level IP geolocation datasets.
	│	|── New_York # Street-level IP geolocation dataset collected from New York City including 91,808 IP addresses.
	│	|── Los_Angeles # Street-level IP geolocation dataset collected from Los Angeles including 92,804 IP addresses.
	│	|── Shanghai # Street-level IP geolocation dataset collected from Shanghai including 126,258 IP addresses.
	├── lib # Contains model implementation files
	│	|── bgrl.py # Bootstrap Graph Representation Learning (BGRL) wrapper implementation
	│	|── layers.py # The code of the attention mechanism.
	│	|── model.py # The core source code of proposed RIPGeo
	│	|── predictors.py # MLP predictor for BGRL contrastive learning
	│	|── scheduler.py # Cosine decay learning rate scheduler with warmup
	│	|── sublayers.py # The support file for layer.py
	│	|── utils.py # Auxiliary functions
	├── asset # Contains saved checkpoints and logs when running the model
	│	|── log # Contains logs when running the model 
	│	|── model # Contains the saved checkpoints
	├── main.py # # Main entry point for three-stage training pipeline (pretrain → contrastive → pseudo)
	├── preprocess.py # Preprocess dataset and execute IP clustering the for model running
	├── pretrain.py # Stage 1: Supervised pretraining with landmark data
	├── train.py # Stage 2: Contrastive learning with BGRL framework and data augmentation
	├── pseudo_train.py # Stage 3: Pseudo-label refinement with stability-based filtering
	└── README.md
```

## Dataset Information

The datasets used in this project is identical to the one used in the CIPGeo project. You can download it from: https://github.com/xtliu-uestc/CIPGeo/tree/main/CIPGeo_code/datasets.

The "datasets" folder contains three subfolders corresponding to three large-scale real-world street-level IP geolocation    datasets collected from New York City, Los Angeles and Shanghai. There are three files in each subfolder:

- data.csv    *# features (including attribute knowledge and network measurements) and labels (longitude and latitude) for street-level IP geolocation* 
- ip.csv    *# IP addresses*
- last_traceroute.csv    *# last four routers and corresponding delays for efficient IP host clustering*

The detailed **columns and description** of data.csv in New York dataset are as follows:

#### New York  

| Column Name                     | Data Description                                             |
| ------------------------------- | ------------------------------------------------------------ |
| ip                              | The IPv4 address                                             |
| as_mult_info                    | The ID of the autonomous system where IP locates             |
| country                         | The country where the IP locates                             |
| prov_cn_name                    | The state/province where the IP locates                      |
| city                            | The city where the IP locates                                |
| isp                             | The Internet Service Provider of the IP                      |
| vp900/901/..._ping_delay_time   | The ping delay from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._trace             | The traceroute list from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._tr_steps          | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._last_router_delay | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| vp900/901/..._total_delay       | The total delay from probing hosts "vp900/901/..." to the IP host |
| longitude                       | The longitude of the IP (as label)                           |
| latitude                        | The latitude of the IP host (as label)                       |

PS: The detailed columns and description of data.csv in other two datasets are similar to New York dataset's.

