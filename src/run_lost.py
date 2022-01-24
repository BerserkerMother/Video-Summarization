import yaml
import time
import argparse

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from models import TLOST

from utils import AverageMeter
from evaluation.compute_metrics import eval_metrics
from dataset import TSDataset


def main(args, splits):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Start Training...')
    avg_fscore = AverageMeter()
    avg_ktau = AverageMeter()
    avg_spr = AverageMeter()
    for split_idx, split in enumerate(splits):
        print(f"\nSplit {split_idx+1}")
        model = TLOST(args.heads, args.d_model, args.num_sumtokens, args.layers,
                      args.mask_size, max_len=10000, device=device)
        optim = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

        num_parameters = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
        print('model has %dM parameters' % (num_parameters // 1000000))

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
        kt_list = []
        sp_list = []
        for e in range(args.max_epoch):
            e_start = time.time()
            train_loss = train_step(model, optim, train_loader, device)
            e_end = time.time()
            val_loss, f_score, ktau, spr = val_step(
                model, val_loader, device, args)
            fs_list.append(f_score)
            kt_list.append(ktau)
            sp_list.append(spr)

            if e % 10 == 0:
                print(
                    f"Epoch {e} : [Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Epoch time {e_end-e_start:.4f}]")
                print(50*'-')

        ft_time_end = time.time()
        avg_fscore.update(max(fs_list), 1)
        avg_ktau.update(max(kt_list), 1)
        avg_spr.update(max(sp_list), 1)
        print(
            f"\nTotal time spent: {(ft_time_end-ft_time_start)/60:.4f}mins\n")

    print(f"Total fscore: {avg_fscore.avg()}")
    print(f"Kendall_tau: {avg_ktau.avg()}")
    print(f"Spearsman_r: {avg_spr.avg()}")


def train_step(model, optim, ft_train_loader, device):
    model.train()
    loss_avg = AverageMeter()
    for i, (feature, target, name) in enumerate(ft_train_loader):
        feature = feature.to(device)
        target = target.to(device)

        pred = model(feature).squeeze(dim=-1)
        loss = model.criterian(pred, target)

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

        pred = model(feature).squeeze(dim=-1)
        loss = model.criterian(pred, target)

        loss_avg.update(loss.item(), 1)
        score_dict[vid_name[0]] = pred.squeeze(0).detach().cpu().numpy()

    f_score, ktau, spr = eval_metrics(score_dict, args)

    return loss_avg.avg(), f_score, ktau, spr


args = argparse.ArgumentParser('LOST')
args.add_argument('--heads', default=4, type=int)
args.add_argument('--d_model', default=512, type=int)
args.add_argument('--num_sumtokens', default=128, type=int)
args.add_argument('--layers', default=3, type=int)
args.add_argument('--mask_size', default=1, type=int)

args.add_argument('--lr', default=1e5, type=float)
args.add_argument('--weight_decay', default=0.01, type=float)

args.add_argument('--data', type=str)
args.add_argument('--dataset', type=str)
args.add_argument('--batch_size', default=1, type=int)
args.add_argument('--max_epoch', default=200, type=int)

arguments = args.parse_args()

if __name__ == '__main__':
    split_path = "splits/tvsum.yaml"
    with open(split_path, 'r') as f:
        try:
            splits = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    main(arguments, splits)
