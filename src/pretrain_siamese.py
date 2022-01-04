import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from model import PretrainModel
from data import TSDataset, collate_fn
from utils import AverageMeter


def main(args):
    # dataset
    dataset = TSDataset(args.data)
    # make data loaders
    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    model = PretrainModel(d_model=args.d_model, num_heads=args.num_heads,
                          num_layer=args.num_layers, sparsity=0.7,
                          attention_dim=args.attention_dim,
                          dropout=args.dropout, in_features=args.in_features,
                          num_class=128, use_pos=args.use_pos).cuda()
    num_el = sum(module.numel() for module in model.parameters()
                 if module.requires_grad) // 1000000
    logging.info('number of model parameter %dM' % num_el)
    optimizer = torch.optim.Adam(model.encoder_main.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scaler = amp.GradScaler()

    logging.info('Starting Pretraining')
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, scaler, train_loader, epoch)
        logging.info("Total Loss %f" % train_loss)
    if args.save:
        torch.save(model.encoder_main.parameters(), "pretrain.pth")


def train(model, optimizer, scaler, loader, e):
    train_loss = AverageMeter()
    for i, features in enumerate(loader):
        features = features.cuda()
        # make padding mask
        mask = (features[:, :, 0] == -1)

        with amp.autocast():
            # forward pass
            loss = model(features, mask)

        # optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logging
        train_loss.update(loss.item(), 1)
        logging.info('Epoch %3d ,Step %d, loss: %f' % (e, i + 1, loss.item()))
    return train_loss.avg()


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
arg_parser.add_argument('--batch_size', default=2, help='batch size')
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
