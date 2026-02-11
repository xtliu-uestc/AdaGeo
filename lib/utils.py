from __future__ import print_function
import numpy as np
import torch
import warnings
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

warnings.filterwarnings(action='once')

import torch
import numpy as np
import torch
import torch.nn as nn




class DataPerturb:
    def __init__(self, eta=0.1, seed=None):
        self.eta = eta
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.call_count = 0  

    def perturb(self, data):
        
        if self.seed is not None:
            self.rng.manual_seed(self.seed + self.call_count)
            self.call_count += 1
            
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # original
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay = data
        lm_X_tensor = Tensor(lm_X)
        lm_Y_tensor = Tensor(lm_Y)
        tg_X_tensor = Tensor(tg_X)
        tg_Y_tensor = Tensor(tg_Y)
        lm_delay_tensor = Tensor(lm_delay)
        tg_delay_tensor = Tensor(tg_delay)

        # add Gaussian data perturb
        new_lm_X = lm_X_tensor.clone()
        new_lm_Y = lm_Y_tensor.clone()
        new_tg_X = tg_X_tensor.clone()
        new_tg_Y = tg_Y_tensor.clone()
        new_lm_delay = lm_delay_tensor.clone()
        new_tg_delay = tg_delay_tensor.clone()

        
        noise_lm_X = torch.randn(new_lm_X[:, -16:].shape, generator=self.rng, device=self.device)
        new_lm_X[:, -16:] += self.eta * noise_lm_X * torch.abs(new_lm_X[:, -16:])
        
        noise_tg_X = torch.randn(new_tg_X[:, -16:].shape, generator=self.rng, device=self.device)
        new_tg_X[:, -16:] += self.eta * noise_tg_X * torch.abs(new_tg_X[:, -16:])
        
        noise_lm_delay = torch.randn(new_lm_delay.shape, generator=self.rng, device=self.device)
        new_lm_delay += self.eta * noise_lm_delay * torch.abs(new_lm_delay)
        
        noise_tg_delay = torch.randn(new_tg_delay.shape, generator=self.rng, device=self.device)
        new_tg_delay += self.eta * noise_tg_delay * torch.abs(new_tg_delay)

        return new_lm_X, new_lm_Y, new_tg_X, new_tg_Y, new_lm_delay, new_tg_delay
    
    def reset(self):
        self.call_count = 0
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

def freeze_for_pseudo(model):
    for name, param in model.named_parameters():
        if 'att_attribute' in name or 'w_1' in name or 'w_2' in name:
            param.requires_grad =True
        elif 'pred' in name or 'gamma' in name or 'alpha' in name or 'beta' in name:
            param.requires_grad = False
        else:
            param.requires_grad =False
    return model

def filter_high_confidence_data(opt, train_data):
    """
    Filter high-confidence target data based on pseudo-label stability (mean absolute difference between adjacent checkpoints).

    Args:
        opt: Argument parser object containing configurations (e.g., pseudo-label path, quantile threshold).
        train_data: List of dictionaries containing cluster data with keys 'lm_X', 'lm_Y', 'tg_X', 'tg_Y',
                    'lm_delay', 'tg_delay', 'center', 'router', 'exist'.

    Returns:
        List of dictionaries containing filtered high-confidence target data, where 'tg_Y' is the mean pseudo-label.
    """
    # Load pseudo-label checkpoints
    checkpoint = torch.load(opt.pseudolabel)  
    
    # Validate checkpoint format
    if not isinstance(checkpoint, torch.Tensor) or checkpoint.ndim != 3 or checkpoint.shape[2] != 2:
        raise ValueError(f"Invalid checkpoint format: Expected (num_checkpoints, total_targets, 2), got {checkpoint.shape}")

    num_checkpoints, total_targets, _ = checkpoint.shape

    # Calculate the mean absolute difference of pseudo-labels between adjacent checkpoints
    res = []
    for i in range(num_checkpoints - 1):
        res.append(torch.abs(checkpoint[i + 1] - checkpoint[i]).unsqueeze(0))
    res = torch.cat(res)  
    res = torch.mean(res, dim=0) 
    res = torch.mean(res, dim=1)  

   # Calculate quantile threshold
    if not hasattr(opt, 'quantile') or opt.quantile is None:
        raise ValueError("Must provide opt.quantile to calculate quantile threshold")
    threshold = torch.quantile(res, opt.quantile)

    # Filter high-confidence targets 
    high_conf_mask = (res <threshold).cpu().numpy()  

    # Calculate mean of pseudo-labels
    mean_labels = checkpoint.mean(dim=0).cpu().numpy()  # (total_targets, 2)
            
   # Verify target count matches
    expected_targets = sum(len(cluster["tg_Y"]) for cluster in train_data)
    if expected_targets != total_targets:
        raise ValueError(
            f"Mismatch: Checkpoint has {total_targets} targets, training data has {expected_targets} targets.")

    # Filter clusters containing only high-confidence targets
    filtered_train_data = []
    offset = 0
    for cluster in train_data:
        num_tg = len(cluster["tg_Y"])
        if num_tg == 0:
            continue

        cluster_mask = high_conf_mask[offset:offset + num_tg]  
        offset += num_tg

        if cluster_mask.sum() == 0:
            continue 

        
        filtered_cluster = {
            "lm_X": cluster["lm_X"],  
            "lm_Y": cluster["lm_Y"],  
            "tg_X": cluster["tg_X"][cluster_mask],  
            # Use pseudo-label mean
            "tg_Y": mean_labels[offset - num_tg:offset][cluster_mask],  
            "lm_delay": cluster["lm_delay"],  
            "tg_delay": cluster["tg_delay"][cluster_mask],  
            "center": cluster["center"],  
            "router": cluster["router"],  
            "exist": cluster["exist"] , 
            "y_max": cluster["y_max"],  
            "y_min": cluster["y_min"]  
           
        }
        filtered_train_data.append(filtered_cluster)

    if not filtered_train_data:
        print("Warning: No clusters containing high-confidence targets found.")

    # # Save indices and average labels of high-confidence targets 
    low_conf_indices = torch.nonzero(res < threshold).squeeze()  
    low_conf_labels = mean_labels[low_conf_indices]  
    torch.save(low_conf_indices, opt.selected_indice if hasattr(opt, 'selected_indice') else f"{opt.model_save_path}{opt.dataset}_high_conf_indices.pt")
    torch.save(low_conf_labels, opt.selected_label if hasattr(opt, 'selected_label') else f"{opt.model_save_path}{opt.dataset}_high_conf_labels.pt")

    return filtered_train_data


class MaxMinLogRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)


class MaxMinRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        # data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)


class MaxMinLogScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data[data != 0] = -np.log(data[data != 0] + 1)
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data[data != 0] = (data[data != 0] - min) / (max - min + 1e-12)
        return data

    def inverse_transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data = data * (max - min) + min
        return np.exp(data)


class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


def graph_normal(graphs, normal=2):
    if normal == 2:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)

    elif normal == 1:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = [1, 1], [0, 0]

    return graphs


def get_data_generator(opt, data_train, data_test, normal=2):
    # load data
    data_train = data_train[np.array([graph["exist"] for graph in data_train])]
    data_test = data_test[np.array([graph["exist"] for graph in data_test])]

    data_train, data_test = graph_normal(data_train, normal=normal), graph_normal(data_test, normal=normal)

    random.seed(opt.seed)
    random.shuffle(data_train)
    random.seed(opt.seed)
    random.shuffle(data_test)

    return data_train, data_test


def square_sum(gamma1, gamma):
    out = ((gamma1 - gamma) ** 2).sum(dim=1, keepdim=True)
    return out


# fusion two NIG
def fuse_nig(gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
    # Eq. 16
    gamma = (gamma1 * v1 + gamma2 * v2) / (v1 + v2 + 1e-12)
    v = v1 + v2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (v1 * square_sum(gamma1, gamma) + v2 * square_sum(gamma2, gamma))
    return gamma, v, alpha, beta


def dis_loss(y, y_pred, max, min):
    y[:, 0] = y[:, 0] * (max[0] - min[0])
    y[:, 1] = y[:, 1] * (max[1] - min[1])
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0])
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1])
    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100) + 1e-8).sum(dim=1))
    return distance


def NIG_NLL(gamma, v, alpha, beta, mse):
    om = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v + 1e-12) \
          - alpha * torch.log(om) \
          + (alpha + 0.5) * torch.log(v * mse + om) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return torch.mean(nll)


def NIG_Reg(v, alpha, mse):
    reg = mse * (2 * v + alpha)
    return torch.mean(reg)


def NIG_loss(gamma, v, alpha, beta, mse, coeffi=0.01):
    # our loss function
    om = 2 * beta * (1 + v)
    loss = \
        (0.5 * torch.log(np.pi / v + 1e-12) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(
            v * mse + om) + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)).sum() / len(gamma)
    lossr = coeffi * (mse * (2 * v + alpha)).sum() / len(gamma)
    loss = loss + lossr
    return loss + lossr


def get_adjancy(func, delay, hop, nodes):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    hops = []
    delays = []
    x1 = []
    x2 = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            delays.append(delay[i, j])
            hops.append(hop[i, j])
            x1.append(nodes[i].cpu().detach().numpy())
            x2.append(nodes[j].cpu().detach().numpy())
    dis = func(Tensor(delays), Tensor(hops), Tensor(x1), Tensor(x2))
    A = torch.zeros_like(delay)
    index = 0
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            A[i, j] = dis[index]
            index += 1
    return A


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            # nn.init.constant_(m.bias, val=0)


# save checkpoint of model
def save_cpt(model, optim, epoch, save_path):
    """
    save checkpoint, for inference/re-training
    :return:
    """
    model.eval()
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
        save_path
    )


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))


def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))


def draw_cdf(ds_sort):
    last, index = min(ds_sort), 0
    x = []
    y = []
    while index < len(ds_sort):
        x.append([last, ds_sort[index]])
        y.append([index / len(ds_sort), index / len(ds_sort)])

        if index < len(ds_sort):
            last = ds_sort[index]
        index += 1
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(x).reshape(-1, 1).squeeze(),
             np.array(y).reshape(-1, 1).squeeze(),
             c='k',
             lw=2,
             ls='-')
    plt.xlabel('Geolocation Error(km)')
    plt.ylabel('Cumulative Probability')
    plt.grid()
    plt.show()
