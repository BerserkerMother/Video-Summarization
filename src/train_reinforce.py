import time
import argparse
import wandb
import logging
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import distributions

from models import TLOSTReinforce
from utils import set_seed, AverageMeter, load_json, load_yaml
from evaluation.compute_metrics import eval_metrics
from dataset import TSDataset, collate_fn


def main(args, splits):
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Start Training...')
    avg_fscore = AverageMeter()
    avg_ktau = AverageMeter()
    avg_spr = AverageMeter()
    for split_idx, split in enumerate(splits):
        set_seed(34123312)
        wandb.init(project='Video-Summarization', entity='berserkermother',
                   name=args.__str__(), config=args, reinit=True)
        wandb.config.seed = 34123312
        logging.info(f"\nSplit {split_idx + 1}")
        model = TLOSTReinforce(args.heads, args.d_model, args.num_sumtokens,
                               args.layers, args.mask_size, args.dropout,
                               max_len=10000, device=device)
        optim = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
        # loads model.pth
        if args.use_model:
            if os.path.exists(args.use_model):
                state_dict = torch.load(args.use_model)
                model.load_state_dict(state_dict, strict=False)

        num_parameters = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
        logging.info('model has %dM parameters' % (num_parameters // 1000000))
        wandb.config.num_el = 34123312

        train_split = split['train_keys']
        test_split = split['test_keys']

        train_split_set = TSDataset(args.data, args.dataset, train_split)
        val_split_set = TSDataset(args.data, args.dataset, test_split)

        train_loader = DataLoader(
            dataset=train_split_set,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            batch_size=args.batch_size
        )

        val_loader = DataLoader(
            dataset=val_split_set,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            batch_size=args.batch_size
        )

        ft_time_start = time.time()
        model = model.to(device)
        fs_list, kt_list, sp_list = [], [], []
        for e in range(args.max_epoch):
            e_start = time.time()
            train_reward = train_step(model, optim, train_loader, device)
            e_end = time.time()
            f_score, ktau, spr = val_step(
                model, val_loader, device, args)
            val_reward = f_score + ktau + spr
            fs_list.append(f_score)
            kt_list.append(ktau)
            sp_list.append(spr)
            wandb.log(
                {
                    'split%d' % split_idx:
                        {
                            'reward': {
                                'train': train_reward,
                                'val': val_reward
                            },
                            'F Score': f_score,
                            'Kendal': ktau,
                            'SpearMan': spr
                        }

                }
            )

            logging.info(
                f"Epoch {e} : [Train reward {train_reward:.4f},"
                f" Val reward {val_reward:.4f},"
                f" Epoch time {e_end - e_start:.4f}]")
            logging.info("F score: %2.4f, Kendal: %2.4f, Spearman: %2.4f"
                         % (f_score, ktau, spr))
            # save model's state dict
            torch.save(model.state_dict(), "model_rl.pth")

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


def train_step(model, optim, ft_train_loader, device):
    model.train()
    reward_avg = AverageMeter()
    for i, (feature, target, user) in enumerate(ft_train_loader):
        score_dict, user_dict = {}, {}
        feature = feature.to(device)

        means, stds = model(feature)
        # create normal distribution
        action_distro = distributions.Normal(means, stds)
        pred = action_distro.sample()

        score_dict[user.name] = pred.squeeze(0).detach().cpu().numpy()
        user_dict[user.name] = user
        f_score, ktau, spr = eval_metrics(score_dict, user_dict, arg_parser)
        reward = f_score + ktau + spr
        reward_avg.update(reward, 1)

        # calculate expectation
        expect = -(reward * action_distro.log_prob(pred)).sum()

        # optimizing
        optim.zero_grad()
        expect.backward()
        optim.step()

    return reward_avg.avg()


@torch.no_grad()
def val_step(model, ft_test_loader, device, args):
    model.eval()
    score_dict, user_dict = {}, {}
    for i, (feature, target, user) in enumerate(ft_test_loader):
        feature = feature.to(device)

        means, stds = model(feature)
        # create normal distribution
        action_distro = distributions.Normal(means, stds)
        pred = action_distro.sample()

        score_dict[user.name] = pred.squeeze(0).detach().cpu().numpy()
        user_dict[user.name] = user
    f_score, ktau, spr = eval_metrics(score_dict, user_dict, args)

    return f_score, ktau, spr


arg_parser = argparse.ArgumentParser('LOST Reinforce')
arg_parser.add_argument('--heads', default=4, type=int)
arg_parser.add_argument('--d_model', default=512, type=int)
arg_parser.add_argument('--num_sumtokens', default=128, type=int)
arg_parser.add_argument('--layers', default=3, type=int)
arg_parser.add_argument('--mask_size', default=1, type=int)
arg_parser.add_argument('--dropout', default=0.3, type=float)

arg_parser.add_argument('--lr', default=1e5, type=float)
arg_parser.add_argument('--weight_decay', default=0.01, type=float)

arg_parser.add_argument('--data', type=str)
arg_parser.add_argument('--dataset', type=str)
arg_parser.add_argument('--batch_size', default=1, type=int)
arg_parser.add_argument('--max_epoch', default=200, type=int)
arg_parser.add_argument("--use_model", default="", type=str,
                        help="given a model path, it will load it")
arg_parser.add_argument("--save", action="store_true",
                        help="if true it saved model after each epoch")

arg_parser.add_argument('--dsnet_split', action='store_true')

arguments = arg_parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    if arguments.dsnet_split:
        split_path = "src/splits_dsnet/tvsum.yaml"
        splits = load_yaml(split_path)
    else:
        split_path = "src/splits_summarizer/tvsum_splits.json"
        splits = load_json(split_path)

    # print(splits)
    main(arguments, splits)
