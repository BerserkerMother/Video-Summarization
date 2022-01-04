import logging
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.cuda import amp
import wandb

from model import MyNetwork
from data import TSDataset, collate_fn
from utils import set_seed, AverageMeter
from evaluatation import f1_score


def main(args):
    for seed_num, seed in enumerate(args.seeds):
        logging.info('Experiment %d | seed number %d' % (seed_num + 1, seed))
        # set seed for experimenting
        set_seed(seed)
        # wandb logger
        wandb.init(project='Video-Summarization', entity='berserkermother',
                   name=arguments.name, config=args, reinit=True)
        # dataset
        dataset = TSDataset(args.data)
        # split dataset
        train_set, val_set = random_split(dataset, [40, 10])
        # make data loaders
        train_loader = DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=args.batch_size,
        )
        val_loader = DataLoader(
            dataset=val_set,
            shuffle=True,
            batch_size=args.batch_size,
        )
        model = MyNetwork(d_model=args.d_model, num_heads=args.num_heads,
                          num_layer=args.num_layers,
                          attention_dim=args.attention_dim,
                          dropout=args.dropout, in_features=args.in_features,
                          num_class=1, use_pos=args.use_pos).cuda()
        num_el = sum(module.numel() for module in model.parameters()
                     if module.requires_grad) // 1000000
        wandb.config.num_el = num_el
        logging.info('number of model parameter %dM' % num_el)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scaler = amp.GradScaler()

        val_loss, f_score_val = val(model, val_loader, args)
        logging.info('random weight network: Val Loss: %f\n\n' % val_loss)
        logging.info('Starting Training')
        for epoch in range(args.epochs):
            train_loss, f_score_train = train(model, optimizer, scaler,
                                              train_loader, args)
            val_loss, f_score_val = val(model, val_loader, args)
            # logging
            logging.info('Epoch %2d\nTrain Loss: %3.4f, Val Loss: %3.4f,'
                         ' Train F Score: %f, Val F Score: %f' % (
                             epoch + 1, train_loss, val_loss, f_score_train, f_score_val))
            wandb.log(
                {
                    'seed%d' % seed:
                        {
                            'loss': {
                                'train': train_loss,
                                'val': val_loss
                            },
                            'F Score': {
                                'train': f_score_train,
                                'val': f_score_val
                            }
                        }
                }
            )
        wandb.finish()


def train(model, optimizer, scaler, loader, args):
    score_dict = {}
    train_loss = AverageMeter()
    for i, data in enumerate(loader):
        features, targets, vid_name = data
        features = features.cuda()
        targets = targets.cuda()

        batch_size = features.size()[0]
        with amp.autocast():
            # forward pass
            logits = model(features)
            logits = torch.sigmoid(logits).view(batch_size, -1)
            loss = F.mse_loss(logits, targets, reduction='sum') \
                   + 0.1 * torch.linalg.norm(logits)

        # optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logging
        score_dict[vid_name[0]] = logits.squeeze(0).detach().cpu().numpy()
        train_loss.update(loss.item(), 1)
        logging.info('step %d, loss: %f' % (i + 1, loss.item()))
    f_score = f1_score(score_dict, args)
    return train_loss.avg(), f_score


def val(model, loader, args):
    score_dict = {}
    test_loss = AverageMeter()
    for data in loader:
        features, targets, vid_name = data
        features = features.cuda()
        targets = targets.cuda()

        batch_size = features.size()[0]
        output = model(features)
        output = torch.sigmoid(output).view(batch_size, -1)
        loss = F.mse_loss(output, targets, reduction='sum') \
               + 0.1 * torch.linalg.norm(output)

        score_dict[vid_name[0]] = output.squeeze(0).detach().cpu().numpy()
        test_loss.update(loss.item(), 1)
    f_score = f1_score(score_dict, args)
    return test_loss.avg(), f_score


# arguments
arg_parser = argparse.ArgumentParser(description=
                                     'Video Summarization with Deep Learning'
                                     )
# data
arg_parser.add_argument('--data', required=True, type=str,
                        help='path to data folder')
arg_parser.add_argument('--dataset', default='tvsum', type=str,
                        choices=['summe', 'tvsum', 'ovp', 'youtube'],
                        help='dataset to run experiments on')
arg_parser.add_argument('--batch_size', default=1, help='batch size')
# model
arg_parser.add_argument('--d_model', type=int, default=256,
                        help='hidden size dimension')
arg_parser.add_argument('--attention_dim', type=int, default=256,
                        help='attention dimension for multi head attention')
arg_parser.add_argument('--use_pos', type=bool, default=True,
                        help='weather to use positional encoding or not')
arg_parser.add_argument('--num_layers', type=int, default=6,
                        help='number of encoder layers')
arg_parser.add_argument('--num_heads', type=int, default=4,
                        help='number of attention heads')
arg_parser.add_argument('--dropout', type=float, default=.2,
                        help='dropout probability')
arg_parser.add_argument('--in_features', type=int, default=1024,
                        help='frame features dimension')
# optimizer
arg_parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate value')
arg_parser.add_argument('--momentum', type=float, default=.9,
                        help='optimizer momentum')
arg_parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='optimizer weight decay')
# others
arg_parser.add_argument('--seeds', nargs='+', type=int,
                        default=[234, 451, 5554, 1231, 31],
                        help='seeds to experiment on')
arg_parser.add_argument('--eval', type=str, default='avg', choices=['avg', 'max'],
                        help='f score evaluation protocol')
arg_parser.add_argument('--name', type=str, required=True,
                        help='wandb experiment name')
arg_parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
arg_parser.add_argument('--save', type=str, default='',
                        help='path to save directory')
arguments = arg_parser.parse_args()
logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    main(arguments)
