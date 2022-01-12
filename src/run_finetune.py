import yaml
import time
import argparse

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split

from utils import AverageMeter
from models import EncoderOnly, MAE
from evaluatation import f1_score
from dataset import TSDataset


MODE = ['freeze', 'non-freeze']
PHASE = ['pre-train', 'fine-tune']


def main(args, splits):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_trained_model = MAE(args.heads, args.d_enc, args.d_dec,
                            args.enc_layers, args.dec_layers, max_len=10000, device=device)
    pre_trained_model.load_state_dict(
        torch.load('../pretrain.pth')['model_state'])

    print('Start fine-tuning...')
    avg_fscore = AverageMeter()
    for split_idx, split in enumerate(splits):
        print(f"\nSplit {split_idx+1}")
        model = EncoderOnly(pre_trained_model, args.d_enc,
                            args.heads, args.enc_layers, mode='non-freeze')
        optim = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

        train_split = split['train_keys']
        test_split = split['test_keys']

        train_split_set = TSDataset(args.data, args.dataset, train_split)
        val_split_set = TSDataset(args.data, args.dataset, test_split)

        train_loader = DataLoader(
            dataset=train_split_set,
            shuffle=True,
            batch_size=args.batch_size,
        )

        val_loader = DataLoader(
            dataset=val_split_set,
            shuffle=True,
            batch_size=args.batch_size,
        )

        ft_time_start = time.time()
        model = model.to(device)
        fs_list = []
        for e in range(args.ft_max_epoch):
            e_start = time.time()
            train_loss = train_step(model, optim, train_loader, device)
            e_end = time.time()
            val_loss, f_score = val_step(model, val_loader, device, args)
            fs_list.append(f_score)

            if e % 10 == 0:
                print(
                    f"fine-tune Epoch {e} : [Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Epoch time {e_end-e_start:.4f}]")
                print(50*'-')

        ft_time_end = time.time()
        avg_fscore.update(max(fs_list), 1)
        print(
            f"\nFine-tune on split total time: {(ft_time_end-ft_time_start)/60:.4f}\n")

    print(avg_fscore.avg())


def train_step(model, optim, ft_train_loader, device):
    model.train()
    loss_avg = AverageMeter()
    for i, (feature, target, name) in enumerate(ft_train_loader):
        feature = feature.to(device)

        vid_len = feature.size(1)
        vis_idx = torch.arange(vid_len, device=device)
        mask_idx = torch.tensor([0], device=device)

        pred = model(feature, mask_idx, vis_idx)
        loss = model.fine_tune_criterian(pred, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.update(loss.item(), 1)

    return loss_avg.avg()


@torch.no_grad()
def val_step(model, ft_test_loader, device, args):
    model.eval()
    score_dict = {}
    loss_avg = AverageMeter()
    for i, (feature, target, vid_name) in enumerate(ft_test_loader):
        feature = feature.to(device)
        target = target.to(device)

        vid_len = feature.size(1)
        vis_idx = torch.arange(vid_len, device=device)
        mask_idx = torch.tensor([0], device=device)

        pred = model(feature, mask_idx, vis_idx)
        loss = model.fine_tune_criterian(pred, target)

        loss_avg.update(loss.item(), 1)
        score_dict[vid_name[0]] = pred.squeeze(0).detach().cpu().numpy()

    f_score = f1_score(score_dict, args)

    return loss_avg.avg(), f_score


args = argparse.ArgumentParser('Finetune encoder')
args.add_argument('--heads', default=4, type=int)
args.add_argument('--d_enc', default=512, type=int)
args.add_argument('--d_dec', default=128, type=int)
args.add_argument('--enc_layers', default=3, type=int)
args.add_argument('--dec_layers', default=1, type=int)

args.add_argument('--lr', default=1e5, type=float)
args.add_argument('--weight_decay', default=0.01, type=float)

args.add_argument('--data', type=str)
args.add_argument('--dataset', type=str)
args.add_argument('--batch_size', default=1, type=int)
args.add_argument('--ft_max_epoch', default=200, type=int)

arguments = args.parse_args()

if __name__ == '__main__':
    split_path = "splits/tvsum.yaml"
    with open(split_path, 'r') as f:
        try:
            splits = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    main(arguments, splits)
