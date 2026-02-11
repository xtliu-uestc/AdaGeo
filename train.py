from lib.model import *
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import copy
from torch.nn.functional import cosine_similarity
from lib.utils import *
from lib.predictors import MLP_Predictor
from lib.bgrl import BGRL
from lib.scheduler import CosineDecayScheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



def initiate(opt, train_data, test_data):
    set_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===== Check Pretrained Model Dimension =====
    pretrain_dim = None
    if opt.pretrained_model and os.path.exists(opt.pretrained_model):
        pretrained_state = torch.load(opt.pretrained_model, map_location=device)
        for k, v in pretrained_state.items():
            if 'att_attribute.q_w.weight' in k:
                pretrain_dim = v.shape[1]
                break
        if pretrain_dim and pretrain_dim != opt.dim_in:
            if pretrain_dim < opt.dim_in:
                print(f"Detected expansion transfer: {pretrain_dim}dim -> {opt.dim_in}dim")
            else:
                print(f"Detected contraction transfer: {pretrain_dim}dim -> {opt.dim_in}dim")
        else:
            pretrain_dim = None

    set_seed(opt.seed)

    # ===== Create Model =====
    encoder = AdaGeo(
        dim_in=opt.dim_in,
        dim_z=opt.dim_z,
        dim_med=opt.dim_med,
        dim_out=2,
        proj_dim=opt.proj_dim,
        collaborative_mlp=opt.c_mlp,
        pretrain_dim=pretrain_dim
    )
    
    # ===== Load Pretrained Weights =====
    if opt.pretrained_model and os.path.exists(opt.pretrained_model):
        print(f"\nLoading pretrained model: {opt.pretrained_model}")
        pretrained_dict = torch.load(opt.pretrained_model, map_location=device)
        model_dict = encoder.state_dict()
        
        compatible_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
                print(f"Loaded: {k}")
            else:
                print(f"Skipped: {k}")
        
        model_dict.update(compatible_dict)
        encoder.load_state_dict(model_dict)
        print(f"\nLoading complete: {len(compatible_dict)}/{len(model_dict)} parameters")
    
   
    # ===== Create BGRL Model =====
    set_seed(opt.seed)
    feature_dim = encoder.get_feature_dim()
    predictor = MLP_Predictor(feature_dim, feature_dim, hidden_size=512)
    model = BGRL(encoder, predictor).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.trainable_parameters(), opt.lr, betas=(opt.beta1, opt.beta2))

    # Encapsulate settings
    settings = {
        "model": model,
        "lr": opt.lr,
        "optimizer": optimizer,
        "device": device
    }
    
    return train_model(train_data, test_data, settings, opt)
def train_model(train_data, test_data, settings, opt):
    model = settings["model"]
    lr = settings["lr"]
    optimizer = settings["optimizer"]
    device = settings["device"]
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # Data perturbers
    transform_1 = DataPerturb(eta=opt.eta_1, seed=opt.seed)
    transform_2 = DataPerturb(eta=opt.eta_2, seed=opt.seed + 1)
    
    # Momentum scheduler
    mm_scheduler = CosineDecayScheduler(1 - opt.mm, 0.01, opt.saved_epoch)
    
    
    log_path = opt.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_file = f"asset/log/{opt.dataset}_contrastive.txt"
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{opt.dataset} Contrastive (Enhanced Prediction Constraints)\n")
        f.write(f"lr={opt.lr}, eta_1={opt.eta_1}, eta_2={opt.eta_2}\n")
        f.write(f"lambda_1={opt.lambda_1}\n")
        f.write(f"{'='*60}\n")
    
    model_save_path = opt.model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    
    
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)
    
    def train_epoch(step):
        model.train()
        total_contrastive = 0
        total_pred_consist = 0
        total_center = 0
        total_boundary = 0
        total_delay = 0
        num_batches = len(train_data)
        
        mm = 1 - mm_scheduler.get(step)
        
        for i in range(num_batches):
            lm_X = train_data[i]["lm_X"]
            lm_Y = train_data[i]["lm_Y"]
            tg_X = train_data[i]["tg_X"]
            tg_Y = train_data[i]["tg_Y"]
            lm_delay = train_data[i]["lm_delay"]
            tg_delay = train_data[i]["tg_delay"]
            y_max = train_data[i]["y_max"]
            y_min = train_data[i]["y_min"]
            
            # Generate augmented views
            x1_data = transform_1.perturb((lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay))
            x2_data = transform_2.perturb((lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay))
            
            optimizer.zero_grad()
            
            x0 = (Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), 
                  Tensor(lm_delay), Tensor(tg_delay))
            x1 = x1_data
            x2 = x2_data
            
            
            y_pred0, online_q0 = model.online_encoder(x0)
            online_q1, target_y2, y_pred1 = model(x1, x2)
            online_q2, target_y1, y_pred2 = model(x2, x1)
            
            
            
            # 1. Contrastive loss
            contrastive_loss = 2 - cosine_similarity(online_q1, target_y2, dim=-1).mean() \
                                 - cosine_similarity(online_q2, target_y1, dim=-1).mean()

           
            # 2. Boundary constraint (prediction should be within landmark bounds)
            lm_Y_tensor = Tensor(lm_Y)
            lm_min = lm_Y_tensor.min(dim=0)[0]
            lm_max = lm_Y_tensor.max(dim=0)[0]
            boundary_loss = F.relu(lm_min - y_pred0).mean() + F.relu(y_pred0 - lm_max).mean()
            
            
            # Total loss
            loss = contrastive_loss +opt.lambda_1* boundary_loss 
                  
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_network(mm)
            
            total_contrastive += contrastive_loss.item()
            total_boundary += boundary_loss.item()
        
        return {
            'contrastive': total_contrastive / num_batches,
            'boundary': total_boundary / num_batches,
        }
    
    def evaluate():
        model.eval()
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                lm_X = test_data[i]["lm_X"]
                lm_Y = test_data[i]["lm_Y"]
                tg_X = test_data[i]["tg_X"]
                tg_Y = test_data[i]["tg_Y"]
                lm_delay = test_data[i]["lm_delay"]
                tg_delay = test_data[i]["tg_delay"]
                y_max = test_data[i]["y_max"]
                y_min = test_data[i]["y_min"]
                
                pred_coords, _ = model.online_encoder([
                    Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), 
                    Tensor(tg_Y), Tensor(lm_delay), Tensor(tg_delay)
                ])
                
                distance = dis_loss(Tensor(tg_Y), pred_coords, y_max, y_min)
                dislist.extend(distance.cpu().numpy())
                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()
        
        total_mse = total_mse / test_num
        total_mae = total_mae / test_num
        median = sorted(dislist)[len(dislist) // 2]
        
        return total_mse, total_mae, median
    
    def save_checkpoints():
        model.eval()
        results = []
        total_mae, total_num = 0, 0
        
        with torch.no_grad():
            for i in range(len(train_data)):
                lm_X = train_data[i]["lm_X"]
                lm_Y = train_data[i]["lm_Y"]
                tg_X = train_data[i]["tg_X"]
                tg_Y = train_data[i]["tg_Y"]
                lm_delay = train_data[i]["lm_delay"]
                tg_delay = train_data[i]["tg_delay"]
                y_max = train_data[i]["y_max"]
                y_min = train_data[i]["y_min"]
                
                pred_coords, _ = model.online_encoder([
                    Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),
                    Tensor(tg_Y), Tensor(lm_delay), Tensor(tg_delay)
                ])
                
                distance = dis_loss(Tensor(tg_Y), pred_coords, y_max, y_min)
                total_mae += distance.sum()
                total_num += len(tg_Y)
                results.append(pred_coords)
        
        results = torch.cat(results, dim=0)
        print(f"Pseudo-label MAE: {total_mae/total_num:.4f}")
        return results
    
    # ===== Main Training Loop =====
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0
    checkpoint = []
    best_model = None
    final_epoch = 0
    
    print("Saving initial pseudo-labels...")
    checkpoint.append(save_checkpoints().unsqueeze(0))
    
    print(f"\n{'='*70}")
    print("Starting training with enhanced prediction constraints")
    print(f"{'='*70}")
    
    for epoch in range(opt.saved_epoch):
        print(f"\nEpoch {epoch}")
        
        transform_1.reset()
        transform_2.reset()
        
        losses_dict = train_epoch(epoch)
        
        print(f"contrastive={losses_dict['contrastive']:.4f} | "
              f"boundary={losses_dict['boundary']:.6f} | "
)
        
        test_mse, test_mae, test_median = evaluate()
        print(f"Test: mse={test_mse:.4f} | mae={test_mae:.4f} | median={test_median:.4f}")
        
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: mae={test_mae:.4f}, median={test_median:.4f}\n")
        
        batch_metric = test_mae.cpu().numpy()
        if batch_metric < np.min(losses) - 1e-6:
            no_better_epoch = 0
            early_stop_epoch = 0
            print(f"*** Better MAE: {batch_metric:.4f} ***")
            best_model = copy.deepcopy(model.online_encoder.state_dict())
            final_epoch = epoch
        else:
            no_better_epoch += 1
            early_stop_epoch += 1
        
        losses.append(batch_metric)
        
        if epoch % opt.intere == 0 and epoch > 0:
            checkpoint.append(save_checkpoints().unsqueeze(0))
        
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            lr /= 2
            print(f"LR changed to {lr}")
            optimizer = torch.optim.Adam(model.trainable_parameters(), lr, betas=(opt.beta1, opt.beta2))
            no_better_epoch = 0
        
        if early_stop_epoch >= opt.early_stop_epoch:
            print(f"Early stopping at epoch {epoch}")
            break
    
    
    if best_model is not None:
        save_path = f"{model_save_path}{opt.dataset}_contrastive_final.pt"
        torch.save(best_model, save_path)
        print(f"\nSaved model at {save_path} (epoch {final_epoch})")
    
    checkpoint.append(save_checkpoints().unsqueeze(0))
    checkpoint = torch.cat(checkpoint, dim=0).cpu()
    torch.save(checkpoint, f"{model_save_path}{opt.dataset}_pseudolabels.pt")
    
    print(f"\nBest MAE: {np.min(losses[1:]):.4f} at epoch {final_epoch}")