from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import torch

from networks import FMNISTnet
from networks import CIFARnet



def load_data(task, batch_size):
    data_path = './Dataset/'
    
    # Load Fashion-MNIST dataset
    if task == 'FMNIST':
        data_path = data_path
        
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform_train)    
        test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    # Load CIFAR10 dataset
    elif task == 'CIFAR10':        
        data_path = data_path + "CIFAR10/"
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)    
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    return train_loader, test_loader



def make_model(network, task, thresh, tau_m, tau_s, num_steps, scaling_factor):
    if network == 'FMNISTnet':
        return FMNISTnet(task, thresh, tau_m, tau_s, num_steps, scaling_factor)
    elif network == 'CIFARnet':
        return CIFARnet(task, thresh, tau_m, tau_s, num_steps, scaling_factor)
    else:
        print('=== Enter the correct network name.')



def load_model(names, model):
    PATH =  './Pretrained_params/' 
    checkpoint = torch.load(PATH + names + ".pth")
    model.load_state_dict(checkpoint['net'])
    # epoch = checkpoint['epoch']
    # acc = checkpoint['acc']
    # acc_hist = checkpoint['acc_hist']
    # loss_train_hist = checkpoint['loss_train_hist']
    # loss_test_hist = checkpoint['loss_test_hist']
    # spike_train_hist = checkpoint['spike_train_hist']
    # spike_test_hist = checkpoint['spike_test_hist']
    
    return model



def load_hyperparemeter(names):
    # return thresh, tau_m, tau_s, num_steps, scaling_factor
    if names == 'FMNIST_FMNISTnet_saved':
        return 0.15, 200, 50, 100, 0.3

    elif names == 'CIFAR10_CIFARnet_saved':
        return 0.15, 165, 50, 100, 0.3

    else:
        print('=== Enter the correct name.')



def save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, spike_train_hist, spike_test_hist):
    state = {
        'net': model.state_dict(),
        'opt': optimizer.state_dict(),
        'acc': acc,
        'acc_hist': acc_hist,
        'epoch': epoch,
        'loss_train_hist': train_loss_hist,
        'loss_test_hist': test_loss_hist,
        'spike_train_hist': spike_train_hist,
        'spike_test_hist': spike_test_hist 
    }
    
    torch.save(state, './' + "{}.pth".format(names))
    
    # best_acc = 0  # best test accuracy
    best_acc = max(acc_hist)
    
    if acc == best_acc:
        torch.save(state, './' + '{}_best.pth'.format(names))



def scheduler_step(opt, epoch, decay_interval, gamma) : 
    lr = []
    if epoch % decay_interval == 0 :
        for g in opt.param_groups:
            g['lr'] = g['lr']*gamma
            lr.append(g['lr'])
    
        print("=== learning rate decayed to {}.\n".format(lr))

