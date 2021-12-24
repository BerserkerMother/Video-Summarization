import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.cuda import amp
import yaml
import time

import random
import json

from model import MyNetwork, Seq2SeqTransformer
from data import TSDataset, collate_fn
from utils import set_seed, AverageMeter, mse_with_mask_loss
from evaluatation import f1_score
from argparse import ArgumentParser


def main():
    # set seed for experimenting
    seed = 9423
    set_seed(seed)

    parser = ArgumentParser("Video Sum Experiments")
    parser.add_argument('-c',
                        '--configs',
                        dest='configs',
                        default='./configs.yaml')
    config_file = parser.parse_args()

    with open(config_file.configs, 'r') as c:
        try:
            args = yaml.safe_load(c)
            model_params = args['model']
            exp_params = args['exp']
            fit_params = args['fit']
        except yaml.YAMLError as y:
            print(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global root
    root = "../datasets/eccv16_dataset_tvsum_google_pool5.h5"

    # split path
    split_path = "splits/tvsum.yaml"
    with open(split_path, 'r') as f:
        try:
            splits = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    avg_fscore = AverageMeter()
    max_epoch = fit_params['max_epoch']
    print('Starting Training')
    for split_idx, split in enumerate(splits):
        print(f"\nSplit {split_idx}")
        # model = MyNetwork(in_features=1024, num_class=1).cuda()
        model = Seq2SeqTransformer(in_features=1024,
                                   num_class=1,
                                   max_seq_len=10000,
                                   device=device,
                                   **model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=exp_params['lr'])
        scaler = amp.GradScaler()

        train_split = split['train_keys']
        test_split = split['test_keys']

        train_split_set = TSDataset(root, train_split)
        val_split_set = TSDataset(root, test_split)

        train_loader = DataLoader(
            dataset=train_split_set,
            shuffle=True,
            batch_size=exp_params['batch_size'],
            # collate_fn=collate_fn
        )

        val_loader = DataLoader(
            dataset=val_split_set,
            shuffle=True,
            batch_size=exp_params['batch_size'],
            # collate_fn=collate_fn
        )

        _, _, split_fscore = fit(
            max_epoch, model, optimizer, scaler, train_loader, val_loader)
        avg_fscore.update(split_fscore, 1)

    fscore = avg_fscore.avg()
    print(f"Average Fscore: {fscore:.4f}")


def fit(max_epoch, model, optimizer, scaler, train_loader, val_loader):
    # train_loss = train(model, optimizer, scaler, train_loader)
    # val_loss = val(model, val_loader)
    # print('random weight network: Train Loss: %f, Val Loss: %f\n\n' % (train_loss, val_loss))
    for epoch in range(max_epoch):
        start = time.time()
        train_loss = train(model, optimizer, scaler, train_loader)
        end = time.time()
        val_loss, split_fscore = val(model, val_loader)

        if epoch % 10 == 0:
            print('Epoch%2d [Train Loss: %3.4f, Val Loss: %3.4f, Epoch time: %3.4f]' %
                  (epoch + 1, train_loss, val_loss, end-start))
            print('_' * 50)

    return train_loss, val_loss, split_fscore


def train(model, optimizer, scaler, loader):
    train_loss = AverageMeter()
    for i, data in enumerate(loader):
        features, targets, _ = data
        features = features.cuda()
        targets = targets.cuda()
        mask = (targets == 0.0)

        # forward pass

        logits = model(features, mask)
        loss = mse_with_mask_loss(logits, targets, mask)

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        train_loss.update(loss.item(), 1)

        # print('step %d, loss: %f' % (i + 1, loss.item()))

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

    split_fscore = f1_score(score_dict, 'TVSum', root)

    return test_loss.avg(), split_fscore


main()
