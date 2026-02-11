from lib.model import *
import numpy as np
import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.utils import *
import os



def initiate(opt, train_data, test_data):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Initialize model
    model = AdaGeo(dim_in=opt.dim_in, dim_z=opt.dim_z, dim_med=opt.dim_med, dim_out=2,collaborative_mlp=opt.c_mlp)
    model.apply(init_network_weights)
    if cuda:
        model.cuda()


    # Initialize optimizer
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,
                                 betas=(opt.beta1, opt.beta2))


    # Encapsulate settings
    settings = {

        "model": model,
        "lr": lr,
        "optimizer": optimizer,
        "device":device

    }

    return train_model(train_data, test_data,settings, opt)


def train_model(train_data, test_data,settings, opt ):
    model = settings["model"]
    lr=settings["lr"]
    optimizer = settings["optimizer"]
    device= settings["device"]
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    log_path =opt.log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    log_file = f"asset/log/{opt.dataset}_pretrain.txt"
    f = open(log_file, 'a')
    f.write(f"\n*********{opt.dataset} Pretrain Stage*********\n")
    f.write("dim_in=" + str(opt.dim_in) + ", ")
    f.write("dim_z=" + str(opt.dim_z) + ", ")
    f.write("dim_med=" + str(opt.dim_med) + ", ")
    f.write("lr=" + str(opt.lr) + ", ")
    f.write("beta1=" + str(opt.beta1) + ", ")
    f.write("beta2=" + str(opt.beta2) + ", ")
    f.write("harved_epoch=" + str(opt.harved_epoch) + ", ")
    f.write("early_stop_epoch=" + str(opt.early_stop_epoch) + ", ")
    f.write("saved_epoch=" + str(opt.saved_epoch) + ", ")
    f.write("c_mlp=" + str(opt.c_mlp) + ", ")
    f.close()

    model_save_path = opt.model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    def train(model, optimizer,opt):
        model.train()
        total_loss, total_mae, train_num, total_data_perturb_loss = 0, 0, 0, 0
        for i in range(len(train_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = train_data[i]["lm_X"], \
                train_data[i]["lm_Y"], \
                train_data[i]["tg_X"], \
                train_data[i]["tg_Y"], \
                train_data[i]["lm_delay"], \
                train_data[i]["tg_delay"], \
                train_data[i]["y_max"], \
                train_data[i]["y_min"]
            optimizer.zero_grad()

            pred_coords,graph_feature =model([Tensor(lm_X), Tensor(lm_Y),
                              Tensor(tg_X),Tensor(tg_Y), Tensor(lm_delay),
                             Tensor(tg_delay)])

            distance = dis_loss(Tensor(tg_Y), pred_coords, y_max, y_min)

            mse_loss = distance * distance  
            mse_loss = mse_loss.sum()

            mse_loss.backward()
            optimizer.step()

            total_loss += mse_loss.item()
            total_mae += distance.sum()
            train_num += len(tg_Y)

        total_loss = total_loss / train_num
        total_mae = total_mae / train_num

        print("-" * 50)
        print("train: mse: {:.4f} mae: {:.4f}".format(total_loss, total_mae))
        f = open(log_file, 'a')
        f.write(f"Train - mse: {total_loss:.4f} mae: {total_mae:.4f}\n")
        f.close()
        return total_loss,total_mae

    def evaluate(model):
        model.eval()
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []
        distance_all = []

        with torch.no_grad():
            for i in range(len(test_data)):
                lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                    test_data[i][
                        "tg_X"], test_data[i]["tg_Y"], \
                    test_data[i][
                        "lm_delay"], test_data[i]["tg_delay"], \
                    test_data[i]["y_max"], test_data[i]["y_min"]

                pred_coords,graph_feature = model([Tensor(lm_X), Tensor(lm_Y),
                                Tensor(tg_X), Tensor(tg_Y), Tensor(lm_delay),
                                Tensor(tg_delay)])

                distance = dis_loss(Tensor(tg_Y), pred_coords, y_max, y_min)

                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])
                    distance_all.append(distance.cpu().detach().numpy()[i])

                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()


            total_mse = total_mse / test_num
            total_mae = total_mae / test_num

            print("test:  mse: {:.4f}  mae: {:.4f}".format(total_mse,total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])
            f = open(log_file, 'a')
            f.write(
                f"Test - mse: {total_mse:.4f} mae: {total_mae:.4f} median: {dislist_sorted[int(len(dislist_sorted) / 2)]:.4f}\n")
            f.close()

        return total_mse ,total_mae


    # Main training loop

    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0
    best_model = None
    final_epoch = 0
    tolerance = 1e-6  

    for epoch in range(2000):
        print("epoch {}.    ".format(epoch))
        f = open(log_file, 'a')
        f.write(f"epoch : {epoch} \n")
        f.close()

        # Training 
        train(model, optimizer, opt)
         # Validation phase
        test_mse ,test_mae = evaluate(model)

        batch_metric = test_mae.cpu().numpy()
        if batch_metric < np.min(losses) - tolerance:
            no_better_epoch = 0
            early_stop_epoch = 0
            print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_metric))
            best_model = copy.deepcopy(model.state_dict())  # Save best model state
            final_epoch = epoch
            f = open(log_file, 'a')
            f.write(f"Better MAE in epoch {epoch}: {batch_metric:.4f}\n")
            f.close()
            print("-" * 50)
        else:
            no_better_epoch = no_better_epoch + 1
            early_stop_epoch = early_stop_epoch + 1

        losses.append(batch_metric)


        # Dynamic LR decay
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            lr /= 2
            print("learning rate changes to {}!\n".format(lr))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
            no_better_epoch = 0
            f = open(log_file, 'a')
            f.write(f"Learning rate changed to {lr}\n")
            f.close()

        if early_stop_epoch == opt.early_stop_epoch:
            print(f"Early stopping triggered at epoch {epoch}!")
            f = open(log_file, 'a')
            f.write(f"Early stopping triggered at epoch {epoch}\n")
            f.close()
            break

        # Save the final best model
    if best_model is not None:
        save_path = f"{model_save_path}{opt.dataset}_pretrain_final.pt"
        torch.save(best_model, save_path)
        print(f"Saved final model at {save_path} (epoch {final_epoch})!")

        f = open(log_file, 'a')
        f.write(f"Saved final model at {save_path} (epoch {final_epoch})\n")
        f.close()
