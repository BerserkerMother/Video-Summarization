import torch
from torch.utils.data import random_split, DataLoader
from torch.cuda import amp

import logging
import argparse
import wandb

from model import MyNetwork
from data import TSDataset, collate_fn
from utils import set_seed, AverageMeter, mse_with_mask_loss
from evaluatation import f1_score


def main(args):
    # set seed for experimenting
    set_seed(args.seed)
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

    model = MyNetwork(in_features=args.in_features, num_class=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler()

    train_loss = train(model, optimizer, scaler, train_loader)
    val_loss = val(model, val_loader)
    logging.info('random weight network: Train Loss: %f, Val Loss: %f\n\n' % (train_loss, val_loss))

    logging.info('Starting Training')
    epochs = 25
    for epoch in range(epochs):
        train_loss = train(model, optimizer, scaler, train_loader)
        val_loss = val(model, val_loader)
        # logging
        logging.info('Epoch %2d\nTrain Loss: %3.4f, Val Loss: %3.4f' % (epoch + 1, train_loss, val_loss))
        wandb.log(
            {
                'loss': {
                    'train': train_loss,
                    'val': val_loss
                }
            }
        )


def train(model, optimizer, scaler, loader):
    train_loss = AverageMeter()
    for i, data in enumerate(loader):
        features, targets, _ = data
        features = features.cuda()
        targets = targets.cuda()
        mask = (targets == 0.0)
        with amp.autocast():
            # forward pass
            logits = model(features, mask)
            loss = mse_with_mask_loss(logits, targets, mask)
        # optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # logging
        train_loss.update(loss.item(), 1)

        logging.info('step %d, loss: %f' % (i + 1, loss.item()))

    return train_loss.avg()


def val(model, loader):
    score_dict = {}
    test_loss = AverageMeter()
    for data in loader:
        features, targets, vid_name = data
        features = features.cuda()
        targets = targets.cuda()
        mask = (targets == 0.0)

        output = model(features, mask)
        loss = mse_with_mask_loss(output, targets, mask)
        test_loss.update(loss.item(), 1)

        score_dict[vid_name[0]] = output.squeeze(0).detach().cpu().numpy()

    f_score = f1_score(score_dict, 'TVSum')
    wandb.log(
        {
            'F Score': f_score
        }
    )

    return test_loss.avg()


# arguments
arg_parser = argparse.ArgumentParser(description='Video Summarization with Deep Learning')
# data
arg_parser.add_argument('--data', required=True, type=str, help='path to data folder')
arg_parser.add_argument('--batch_size', default=1, help='batch size')
# model
arg_parser.add_argument('--in_features', type=int, default=1024, help='frame features dimension')
# optimizer
arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate value')
# others
arg_parser.add_argument('--seed', type=int, default=234, help='experiment seed')
arg_parser.add_argument('--name', type=str, required=True, help='wandb experiment name')
arguments = arg_parser.parse_args()
# wandb logger
wandb.init(project='Video-Summarization', entity='berserkermother', name=arguments.name, config=arguments)
# formatting logger
logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.DEBUG
)
main(arguments)
