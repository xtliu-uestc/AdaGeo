import random
import torch
import argparse
from lib.utils import *
import numpy as np
import pretrain
import train
import pseudo_train
import os




parser = argparse.ArgumentParser(description="IP")
# parameters of initializing
parser.add_argument('--model_name', type=str, default='TrustGeo')
parser.add_argument('--seed', type=int, default=2022, help='manual seed')

parser.add_argument("--stage", type=str, default="contrastive", help="pretrain/contrastive/pseudo")
parser.add_argument('--dataset', type=str, default='Los_Angeles', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')
# parameters of save
parser.add_argument("--model_save_path", type=str, default="asset/model/")
parser.add_argument("--log_path", type=str, default=f"asset/log")
parser.add_argument("--pretrained_model", type=str, default= f"asset/model/New_York_pretrain_final.pt")
parser.add_argument("--constrastive_model", type=str, default=f"asset/model/Los_Angeles_contrastive_final.pt")
parser.add_argument("--pseudo_model", type=str, default=f"asset/model/Los_Angeles_pseudo_final.pt")
parser.add_argument("--pseudolabel", type=str, default=f"asset/model/Los_Angeles_pseudolabels.pt")

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")
parser.add_argument('--dim_med', type=int, default=32)
parser.add_argument('--dim_z', type=int, default=32)
parser.add_argument('--c_mlp', type=bool, default=True)

# parameters of pretrain
parser.add_argument('--beta1', type=float, default=0.9, help='optimizer beta1')
parser.add_argument('--beta2', type=float, default=0.999,help='optimizer beta2')
parser.add_argument('--lr', type=float, default=2e-3)#
parser.add_argument('--harved_epoch', type=int, default=5, help='number of epochs to wait before learing rate decay')
parser.add_argument('--early_stop_epoch', type=int, default=50, help='number of epochs to wait before early stop')

# parameters of train
parser.add_argument('--eta_1', type=float, default=0.1)
parser.add_argument('--eta_2', type=float, default=0.3)
parser.add_argument('--proj_dim', type=int, default=30)
parser.add_argument("--intere", type=int, default=3,help='pseudo-label saving interval')
parser.add_argument('--mm', type=float, default=0.996, help='Momentum for target network update')
parser.add_argument('--saved_epoch', type=int, default=15)
parser.add_argument('--lambda_1', type=float, default=1)

# parameters of pseudo_train
parser.add_argument(
    "--quantile", type=float, default=0.1, help="stable score quantile"
)


opt = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(opt.seed)
print(f"Seed fixed to {opt.seed}")


'''load data'''
train_data = np.load("./datasets/{}/Clustering_s1234_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1234_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")

if __name__ == '__main__':
    set_seed(opt.seed)
    if opt.stage == "pretrain":
        pretrain.initiate(opt, train_data, test_data)
    elif opt.stage == "contrastive":
        train.initiate(opt,train_data, test_data)
    elif opt.stage == "pseudo":
        pseudo_train.initiate(opt,train_data, test_data)
