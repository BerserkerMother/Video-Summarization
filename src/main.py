
from dataset import TSDataset, PreTrainDataset
import time
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split

from utils import AverageMeter
from models import MAE
from models import EncoderOnly
from evaluatation import f1_score

MODE = ['freeze', 'non-freeze']
PHASE = ['pre-train', 'fine-tune']


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    pr_dataset = PreTrainDataset(args.data)
    ft_dataset = TSDataset()
    # data_loader
    # ###
    pr_t_data_loader = DataLoader(
        pr_dataset, shuffle=True, batch_size=args.batch_size)
    # ###
    train_set, val_set = random_split(ft_dataset, [40, 10])
    ft_train_loader = DataLoader(
        train_set, shuffle=True, batch_size=args.batch_size)
    ft_test_loader = DataLoader(
        val_set, shuffle=True, batch_size=args.batch_size)

    model = MAE(args.heads, args.d_enc, args.d_dec,
                args.enc_layers, args.dec_layers, max_len=10000)
    optim1 = Adam(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay)

    pre_trained_model = pre_train(model, optim1, pr_t_data_loader,
                                  args.pr_t_max_epoch, device)
    check_point_state = {
        "model_state": pre_trained_model.state_dict()
    }
    torch.save(check_point_state, './pr_checkpoint.pth')

    # ######
    ft_model = EncoderOnly(pre_trained_model, args.d_enc, mode='non-freeze')
    optim2 = Adam(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay)

    result = fine_tune(ft_model, optim2, ft_train_loader,
                       ft_test_loader, args.ft_max_epoch)


def pre_train(model, optim, pr_t_data_loader, pr_t_max_epoch, device):
    print('Starting pre-training...')
    model.train()
    pr_time_start = time.time()
    model = model.to(device)
    for e in range(pr_t_max_epoch):
        e_start = time.time()
        avg_loss = AverageMeter()
        for i, (features, _) in enumerate(pr_t_data_loader):
            features = features.to(device)
            # rand masking
            vid_len = features.size(1)
            mask_ratio = 0.3 * vid_len
            # ## select rand indics
            vis_idx = torch.from_numpy(np.random.choice(
                vid_len, mask_ratio, replace=False)).to(device)
            vis_idx_to_zero = torch.arange(vid_len, device=device)
            # ## set random indices to -1
            vis_idx_to_zero[vis_idx] = -1
            # ## remove out visible indices
            mask_idx = vis_idx_to_zero[vis_idx_to_zero != -1]

            pred = model(features, mask_idx, vis_idx, phase='pre-train')
            loss = model.pre_train_criterian(
                pred, features[:, mask_idx, :], mode='Sim')

            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss.update(loss.item(), 1)

        e_end = time.time()
        print(
            f"Pre-train Epoch {e} : [Loss {avg_loss.avg():.4f}, Epoch time {e_end-e_start:.4f}]")

    pr_time_end = time.time()
    print(f"\nPre-train total time: {pr_time_end-pr_time_start:.4f}\n")

    return model


def fine_tune(model, optim, ft_train_loader, ft_test_loader, ft_max_epoch, splits, device):
    print('Start fine-tuning...')
    ft_time_start = time.time()
    model = model.to(device)
    for e in range(ft_max_epoch):
        e_start = time.time()
        train_loss = train_step(model, optim, ft_train_loader, device)
        e_end = time.time()
        val_loss, f_score = val_step(model, ft_test_loader, device)

        print(
            f"fine-tune Epoch {e} : [Train loss {train_loss:.4f}, \
                Val loss {val_loss}, f-score {f_score} Epoch time {e_end-e_start:.4f}]")

    ft_time_end = time.time()
    print(f"\nFine-tune total time: {ft_time_end-ft_time_start:.4f}\n")


def train_step(model, optim, ft_train_loader, device):
    model.train()
    loss_avg = AverageMeter()
    for i, (feature, target, name) in enumerate(ft_train_loader):
        feature = feature.to(device)

        vid_len = feature.size(1)
        vis_idx = torch.arange(vid_len, device=device)
        mask_idx = torch.tensor([0], device=device)

        pred = model(feature, mask_idx, vis_idx, mode='fine-tune')
        loss = model.fine_tune_criterian(pred, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.update(loss.item())

    return loss_avg.avg()


@torch.no_grad()
def val_step(model, ft_test_loader, device):
    model.eval()
    score_dict = {}
    loss_avg = AverageMeter()
    for i, (feature, target, vid_name) in enumerate(ft_test_loader):
        feature = feature.to(device)
        target = target.to(device)

        vid_len = feature.size(1)
        vis_idx = torch.arange(vid_len, device=device)
        mask_idx = torch.tensor([0], device=device)

        pred = model(feature, mask_idx, vis_idx, mode='fine-tune')
        loss = model.fine_tune_criterian(pred, target)

        loss_avg.update(loss.item())
        score_dict[vid_name[0]] = pred.squeeze(0).detach().cpu().numpy()
        loss_avg.update(loss.item(), 1)
    f_score = f1_score(score_dict)

    return loss_avg.avg(), f_score
