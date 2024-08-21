"""Launch script to learn to classify images: this is just a trial to see how to prepare the training set and train it in pytorch"""
import numpy as np
import gymnasium as gym
from gym_CTgraph import CTgraph_env
from gym_CTgraph.CTgraph_plot import CTgraph_plot
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
from learning.networks import ConvNet3l, linearL
from learning.logger import Logger
from learning.createDataset import CustomDatasetFromImages

import argparse
import json
import random
import matplotlib.pyplot as plt
import timeit

import torch
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset
import torch.utils.data as data_utils
from torch.autograd import Variable

def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        logger.nextstep()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print ('Step [{}], Loss: {:.4f}, Acc: {:.2f}'
               .format(logger.step+1, loss.item(), accuracy.item()))


            # 1. Log scalar values (scalar summary)
            info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, logger.step+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), logger.step+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), logger.step+1)

            # 3. Log training images (image summary)
            info = { 'images': data.view(-1, 12, 12)[0:32].cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, images, logger.step+1)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    batch_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            print('correct:', batch_correct)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

  # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test', type=int, default=0, metavar='N',
                            help='0 for training (default) 1 for testing')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # fetch the parameters from the json file
    configuration = CTgraph_conf("graph.json")
    conf_data = configuration.getParameters()

    MimagesObj = CTgraph_images(conf_data)

    # when fetching the images twice, noise will be added differently, so effectively the train and test set will be different
    imgDataset = CustomDatasetFromImages(MimagesObj)
    testDataset = CustomDatasetFromImages(MimagesObj)

    #plotting
    CTgraph_plot.plotImages(MimagesObj, False)
    CTgraph_plot.plotImages(MimagesObj, False)
    CTgraph_plot.plotImages(MimagesObj, True)


    imgDataset_loader = torch.utils.data.DataLoader(imgDataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    imgDataset_loader = torch.utils.data.DataLoader(imgDataset,
        batch_size=100, shuffle=True, **kwargs)


    testDataset_loader = torch.utils.data.DataLoader(testDataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    import datetime
    stamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
    logger = Logger('./logs/' + stamp, )
    model = ConvNet3l().to(device).double()
    #model = linearL().to(device).double()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.test == 0:
        for season in range(0,1):
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, imgDataset_loader, optimizer, epoch, logger)
                # Save the model checkpoint
                torch.save(model.state_dict(), 'params.ckpt')
                test(args, model, device, testDataset_loader)
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, imgDataset_loader2, optimizer, epoch, logger)
                # Save the model checkpoint
                torch.save(model.state_dict(), 'params.ckpt')
                test(args, model, device, testDataset_loader2)

    else:
        # Save and load only the model parameters (recommended).
        model.load_state_dict(torch.load('params.ckpt'))
        test(args, model, device, testDataset_loader)

if __name__ == '__main__':
    main()
