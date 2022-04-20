import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from model import PretrainModel
from data import PreTrainDataset, collate_fn_pretrain
from utils import AverageMeter
from schedular import CosineSchedularLinearWarmup


def main(args):
    # data
    dataset = PreTrainDataset(args.data)
    # make data loaders
    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collate_fn_pretrain,
        num_workers=4,
        drop_last=True
    )
    logging.info("number of videos: %d" % len(dataset))

    model = PretrainModel(num_heads=args.num_heads,
                          num_layers=args.num_layers, sparsity=args.sparsity,
                          dropout=args.dropout, num_classes=1,
                          use_pos=args.use_pos).cuda()
    num_el = sum(module.numel() for module in model.parameters()
                 if module.requires_grad) // 1000000
    logging.info('number of model parameter %dM' % num_el)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    schedular = CosineSchedularLinearWarmup(optimizer, 75 // args.batch_size,
                                            10, args.epochs, args.lr)
    scaler = amp.GradScaler()

    logging.info('Starting Pretraining')
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, schedular, scaler, train_loader, epoch)
        logging.info("Total Loss %f" % train_loss)
        print('_' * 50)
        torch.save(model.encoder.state_dict(), "pretrain.pth")


def train(model, optimizer, schedular, scaler, loader, e):
    train_loss = AverageMeter()
    temp_loss = 0
    for i, (features, vid_rep) in enumerate(loader):
        features = features.cuda()
        vid_rep = vid_rep.cuda()
        # make padding mask, 1000 is padding value
        mask = (features[:, :, 0] == 1000)

        with amp.autocast():
            # forward pass
            loss = model(features, vid_rep, mask)

        # optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr = schedular.step()

        # logging
        temp_loss += loss.item()
        if ((i + 1) % 2) == 0:
            train_loss.update(temp_loss, 1)
            logging.info('Epoch %3d ,Step %d, loss: %f, lr: %f' %
                         (e, i + 1, temp_loss, lr))
            temp_loss = 0
    return train_loss.avg()


# arguments
arg_parser = argparse.ArgumentParser(
    description='Video Summarization with Deep Learning')
# data
arg_parser.add_argument('--data', required=True, type=str,
                        help='path to data folder')
arg_parser.add_argument('--datasets', default='tvsum+summe+ovp+youtube',
                        type=str, help='datasets to contain')
arg_parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
# model
arg_parser.add_argument('--d_model', type=int, default=512,
                        help='hidden size dimension')
arg_parser.add_argument('--use_pos', type=bool, default=True,
                        help='weather to use positional encoding or not')
arg_parser.add_argument('--num_layers', type=int, default=3,
                        help='number of encoder layers')
arg_parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads')
arg_parser.add_argument('--dropout', type=float, default=.2,
                        help='dropout probability')
arg_parser.add_argument('--sparsity', type=float, default=0.5,
                        help="control sparsity of model")
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
