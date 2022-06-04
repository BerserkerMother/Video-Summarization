import time
import argparse
import wandb
import logging
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda import amp
from model import SimNet

from utils import set_seed, AverageMeter, load_json, load_yaml, mse_with_mask_loss
from evaluation.compute_metrics import eval_metrics
from data import TSDataset, collate_fn_train, collate_fn_test


def main(args, splits):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Start Training...')
    avg_fscore = AverageMeter()
    avg_ktau = AverageMeter()
    avg_spr = AverageMeter()
    for split_idx, split in enumerate(splits):
        logging.info(f"\nSplit {split_idx + 1}")

        # define model
        model = SimNet(num_heads=args.num_heads, d_model=args.d_model,
                       num_layers=args.num_layers, sparsity=0.,
                       use_cls=False, dropout=args.dropout,
                       num_classes=1, use_pos=True).cuda()
        optim = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

        scaler = amp.GradScaler()
        # loads model.pth
        if os.path.exists("pretrain.pth"):
            if args.use_model:
                state_dict = torch.load("pretrain.pth")
                model.load_state_dict(state_dict, strict=True)
                logging.info("ATTENTION: loaded pretrained model")

        num_parameters = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
        logging.info('model has %d parameters' % num_parameters)
        wandb.config.num_el = num_parameters

        train_split = split['train_keys']
        test_split = split['test_keys']

        train_split_set = TSDataset(args.data, args.ex_dataset, args.datasets,
                                    train_split)
        val_split_set = TSDataset(args.data, args.ex_dataset, args.datasets,
                                  test_split, split="val")

        train_loader = DataLoader(
            dataset=train_split_set,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_train,
            batch_size=args.batch_size
        )

        val_loader = DataLoader(
            dataset=val_split_set,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_test,
            batch_size=1
        )

        ft_time_start = time.time()
        model = model.to(device)

        fs_list, kt_list, sp_list = [], [], []
        for e in range(args.max_epoch):
            e_start = time.time()
            train_loss = train_step(model, optim, train_loader, scaler, device)
            e_end = time.time()
            val_loss, f_score, ktau, spr = val_step(
                model, val_loader, device)
            fs_list.append(f_score)
            kt_list.append(ktau)
            sp_list.append(spr)

            logging.info(
                f"Epoch {e} : [Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Epoch time {e_end - e_start:.4f}]")
            logging.info(50 * '-')
            # save model's state dict
            torch.save(model.state_dict(), "model_mae.pth")

        ft_time_end = time.time()
        avg_fscore.update(max(fs_list), 1)
        avg_ktau.update(max(kt_list), 1)
        avg_spr.update(max(sp_list), 1)
        logging.info(
            f"\nTotal time spent: {(ft_time_end - ft_time_start) / 60:.4f}mins\n")

        wandb.finish()

    logging.info(f"Total fscore: {avg_fscore.avg()}")
    logging.info(f"Kendall_tau: {avg_ktau.avg()}")
    logging.info(f"Spearsman_r: {avg_spr.avg()}")


def train_step(model, optim, ft_train_loader, scaler, device):
    model.train()
    loss_avg = AverageMeter()
    for i, (feature,  target) in enumerate(ft_train_loader):
        feature = feature.to(device)
        target = target.to(device)
        # make padding mask, 1000 is padding value
        mask = (feature[:, :, 0] == 1000)

        with amp.autocast():
            pred, _ = model(feature, mask)
            loss = mse_with_mask_loss(pred, target, mask)

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_avg.update(loss.item(), 1)

    return loss_avg.avg()


@torch.no_grad()
def val_step(model, ft_test_loader, device):
    model.eval()
    score_dict, user_dict = {}, {}
    loss_avg = AverageMeter()
    for i, (feature, target, user) in enumerate(ft_test_loader):
        feature = feature.to(device)
        target = target.to(device)

        pred, _ = model(feature).view(1, -1)
        loss = F.mse_loss(pred, target)

        loss_avg.update(loss.item(), 1)
        score_dict[user.name] = pred.squeeze(0).detach().cpu().numpy()
        user_dict[user.name] = user
    f_score, ktau, spr = eval_metrics(score_dict, user_dict)

    return loss_avg.avg(), f_score, ktau, spr


@torch.no_grad()
def save_attention_weights(model, ft_train_loader, device):
    model.eval()
    attention_weights_all = {}
    for i, (feature, target, user) in enumerate(ft_train_loader):
        feature = feature.to(device)

        pred, attn_weights = model(feature, True)
        attention_weights_all[user.name] = attn_weights

    torch.save(attention_weights_all, "weights.pth")


arg_parser = argparse.ArgumentParser('lol')
arg_parser.add_argument('--num_heads', default=4, type=int,
                        help="number of self attention heads")
arg_parser.add_argument('--d_model', default=512, type=int)
arg_parser.add_argument('--num_layers', default=3, type=int,
                        help="number of transformer layers")
arg_parser.add_argument('--dropout', default=0.3, type=float,
                        help="model dropout")

arg_parser.add_argument('--lr', default=1e5, type=float, help="optimizer lr")
arg_parser.add_argument('--weight_decay', default=0.01, type=float,
                        help="optimizer l2 norm weight")

arg_parser.add_argument('--data', type=str)
arg_parser.add_argument('--ex_dataset', type=str, default="tvsum",
                        help="experimenting data")
arg_parser.add_argument('--datasets', type=str, help="datasets to load")
arg_parser.add_argument('--batch_size', default=4, type=int,
                        help="mini batch size")
arg_parser.add_argument('--max_epoch', default=200, type=int,
                        help="number of training epochs")
arg_parser.add_argument("--name", default="", type=str,
                        help="wandb experiment name")
arg_parser.add_argument("--use_model", action="store_true",
                        help="if true it loads model.pth")
arg_parser.add_argument("--save", action="store_true",
                        help="if true it saved model after each epoch")

arg_parser.add_argument('--dsnet_split', action='store_true')

arguments = arg_parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    logging.info(arguments.dsnet_split)
    if arguments.dsnet_split:
        split_path = "src/splits_dsnet/tvsum.yaml"
        splits = load_yaml(split_path)
    else:
        split_path = "src/splits_summarizer/tvsum_splits.json"
        splits = load_json(split_path)

    # print(splits)
    main(arguments, splits)
