import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.cuda import amp

import random
import json

from model import MyNetwork
from data import TSDataset, collate_fn
from utils import set_seed, AverageMeter, mse_with_mask_loss
from evaluatation import f1_score


def main():
    # set seed for experimenting
    seed = 9423
    set_seed(seed)

    # dataset
    dataset = TSDataset('../data/eccv16_dataset_tvsum_google_pool5.h5')
    # split dataset
    train_set, val_set = random_split(dataset, [40, 10])
    # make data loaders
    batch_size = 1

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=batch_size,
        #collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_set,
        shuffle=True,
        batch_size=batch_size,
        #collate_fn=collate_fn
    )

    model = MyNetwork(in_features=1024, num_class=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = amp.GradScaler()

    train_loss = train(model, optimizer, scaler, train_loader)
    val_loss = val(model, val_loader)
    print('random weight network: Train Loss: %f, Val Loss: %f\n\n' % (train_loss, val_loss))

    print('Starting Training')
    epochs = 25
    for epoch in range(epochs):
        train_loss = train(model, optimizer, scaler, train_loader)
        val_loss = val(model, val_loader)

        print('Epoch %2d\nTrain Loss: %3.4f, Val Loss: %3.4f' % (epoch + 1, train_loss, val_loss))
        print('_' * 50)


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

        print('step %d, loss: %f' % (i + 1, loss.item()))

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

    f1_score(score_dict, 'TVSum')

    return test_loss.avg()



main()
