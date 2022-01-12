
from dataset import PreTrainDataset
import time
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, random_split
import argparse

from utils import AverageMeter
from models import MAE

MODE = ['freeze', 'non-freeze']
PHASE = ['pre-train', 'fine-tune']


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    print("Preparing dataset...")
    pr_dataset = PreTrainDataset(args.data)
    # data_loader
    pr_t_data_loader = DataLoader(
        pr_dataset, shuffle=True, batch_size=args.batch_size)

    print("Preparing mdoel...")
    model = MAE(args.heads, args.d_enc, args.d_dec,
                args.enc_layers, args.dec_layers, max_len=10000, device=device)
    optim1 = Adam(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay)

    pre_trained_model = pre_train(model, optim1, pr_t_data_loader,
                                  args.pr_t_max_epoch, device)


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
            mask_ratio = int(0.3 * vid_len)
            # ## select rand indics
            vis_idx = torch.from_numpy(np.random.choice(
                vid_len, mask_ratio, replace=False)).to(device)
            vis_idx_to_zero = torch.arange(vid_len, device=device)
            # ## set random indices to -1
            vis_idx_to_zero[vis_idx] = -1
            # ## remove out visible indices
            mask_idx = vis_idx_to_zero[vis_idx_to_zero != -1]

            pred = model(features, mask_idx, vis_idx)
            loss = model.pre_train_criterian(
                pred, features[:, mask_idx, :], mode='mse')

            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss.update(loss.item(), 1)

        e_end = time.time()
        if e % 1 == 0:
            print(
                f"Pre-train Epoch {e}/{pr_t_max_epoch} : [Loss {avg_loss.avg():.4f}, Epoch time {e_end-e_start:.4f}]")

            check_point_state = {
                "model_state": model.state_dict()
            }
            torch.save(check_point_state, '../pretrain.pth')

    pr_time_end = time.time()
    print(
        f"\nPre-train total time: {(pr_time_end-pr_time_start)/60:.4f}mins\n")

    return model


args = argparse.ArgumentParser('MAE Pre-training')
args.add_argument('--heads', default=4, type=int)
args.add_argument('--d_enc', default=512, type=int)
args.add_argument('--d_dec', default=128, type=int)
args.add_argument('--enc_layers', default=3, type=int)
args.add_argument('--dec_layers', default=1, type=int)

args.add_argument('--lr', default=1e5, type=float)
args.add_argument('--weight_decay', default=0.01, type=float)

args.add_argument('--data', type=str)
args.add_argument('--batch_size', default=1, type=int)
args.add_argument('--pr_t_max_epoch', default=200, type=int)

arguments = args.parse_args()

if __name__ == '__main__':
    main(arguments)
