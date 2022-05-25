from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from utils import load_data
from utils import make_model
from utils import load_hyperparemeter
from utils import load_model
from utils import save_model
from functions import scheduler_step


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='BPLC+NOSO FMNIST/CIFAR10')

parser.add_argument('--task', type=str, default='FMNIST', help='which task to run (FMNIST or CIFAR10)')
parser.add_argument('--network', type=str, default='FMNISTnet', help='which network to run (FMNISTnet or CIFARnet)')
parser.add_argument('--mode', type=str, default='train', help='whether to train or eval')

# Hyperparameters 
parser.add_argument('--thresh', type=float, default=0.15, help='Dual spiking threshold [mV]')
parser.add_argument('--tau_m', type=float, default=200, help='Time constant of membrane potential kernel [ms]')
parser.add_argument('--tau_s', type=float, default=50, help='Time constant of synaptic current kernel [ms]')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--num_steps', type=int, default=100, help='Number of time steps')
parser.add_argument('--scaling_factor', type=float, default=0.3, help='Constant input scaling factor')
parser.add_argument('--weight_decay', type=float, default=5E-3, help='Weight decay (L2 regularization) coefficient')
parser.add_argument('--learning_rate', type=float, default=5E-3, help='Initial learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_decay_interval', type=int, default=3, help='Learning rate decay interval')


args = parser.parse_args()

def main():
    names = args.task + '_' + args.network
    train_loader, test_loader = load_data(args.task, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.mode == 'train':
        model = make_model(args.network, args.task, args.thresh, args.tau_m, args.tau_s, args.num_steps, args.scaling_factor).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        acc_hist = list([])
        train_loss_hist = list([])
        test_loss_hist = list([])
        spike_train_hist = list([])
        spike_test_hist = list([])
        
        for epoch in range(args.num_epochs):
            # lr scheduler step
            if epoch != 0 :
                scheduler_step(optimizer, epoch, args.lr_decay_interval, args.lr_decay_rate)
            
            start_time = time.time()
            train_loss, spike_map_train = train(model, train_loader, criterion, optimizer)
            test_loss, spike_map_test, acc = test(model, test_loader, criterion)
            
            spike_train_hist.append(spike_map_train)
            spike_test_hist.append(spike_map_test)
            acc_hist.append(acc)
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)
            time_elapsed = time.time() - start_time
            
            print('\nEpoch: {}/{}.. '.format(epoch+1, args.num_epochs).ljust(14),
                      'Train Loss: {:.3f}.. '.format(train_loss).ljust(20),
                      'Test Loss: {:.3f}.. '.format(test_loss).ljust(19),
                      'Test Accuracy: {:.3f}'.format(acc))        
            print('Time elapsed: {:.6f} \n'.format(time_elapsed))
            
            
            # save model pth file
            save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, spike_train_hist, spike_test_hist)
            
            
    elif args.mode == 'eval':
        names = names + '_saved'
        thresh, tau_m, tau_s, num_steps, scaling_factor = load_hyperparemeter(names)
        model = make_model(args.network, args.task, thresh, tau_m, tau_s, num_steps, scaling_factor).to(device)
        model = load_model(names, model)
        
        test_loss, spike_map_test, acc = test(model, test_loader, criterion)
        print('Test Accuracy: {:.3f}'.format(acc))
        print('spike count = {}'.format(spike_map_test))
        
        
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        spike_map_train = list([])
        model.zero_grad()
        optimizer.zero_grad()
        
        # Adding gaussian noise
        if args.task == 'FMNIST':
            images += torch.tensor(np.random.normal(0, 5E-2, images.size()))
            images = torch.clamp(images, 0.0, 1.0)
        
        outputs = model(images.to(device), args.batch_size, args.mode)
        
        # outputs[0]: output spike latency
        # outputs[1]: mebrane potential of output neurons when spiking
        # outputs[2]: spike map over SNN
        
        loss = F.cross_entropy(-outputs[0], labels.to(device))
        
        # Training progress
        if args.task == 'FMNIST':
            train_set_len = 6000
        elif args.task == 'CIFAR10':
            train_set_len = 5000
        if i % (train_set_len / args.batch_size) == 0 : 
            print('=== {} / {} updating...'.format(i, len(train_loader)))
            print('spike count = {}'.format(outputs[2]))
        
        train_loss += loss.item() / len(train_loader)
        loss.backward()
        optimizer.step()
        spike_map_train.append(outputs[2])
        
    
    del loss, outputs
    return train_loss, spike_map_train


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            spike_map_test = list([])
            outputs = model(inputs.to(device), args.batch_size, args.mode)
            
            # outputs[0]: output spike latency
            # outputs[1]: mebrane potential of output neurons when spiking
            # outputs[2]: spike map over SNN
            
            loss = criterion(-outputs[0].cpu(), targets)
            test_loss += loss.item() / len(test_loader)
            
            # If multi output neurons has same output, compare the membrane potential of output neurons when spiking
            _, predicted = (torch.abs(outputs[1].cpu()) * (outputs[0].cpu() == (outputs[0].cpu().min(1))[0][:, None])).max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            acc = 100. * float(correct) / float(total)
            
            spike_map_test.append(outputs[2])
            
            
    return test_loss, spike_map_test, acc

if __name__=='__main__':
    main()
