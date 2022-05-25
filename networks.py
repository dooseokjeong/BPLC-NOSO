import torch
import torch.nn as nn

from functions import noso
from functions import encoding_neuron
from functions import dodt
from functions import MinPool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


# FMNIST network
class FMNISTnet(nn.Module):
    def __init__(self, task, thresh, tau_m, tau_s, num_steps, scaling_factor):
        super(FMNISTnet, self).__init__()
        self.task = task
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.num_steps = num_steps
        self.scaling_factor = scaling_factor
        
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2, bias=False).float()
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=False).float()
        self.fc1 = nn.Linear(3136, 600, bias=False).float()
        self.fc2 = nn.Linear(600, 10, bias=False).float()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight.data)


    def forward(self, input, batch_size, mode):
        input_u = in_spike = torch.zeros(batch_size, 1, 28, 28, dtype=dtype, device=device)
        
        hc1_vm = hc1_vs = torch.zeros(batch_size, 1, 28, 28, dtype=dtype, device=device)
        hc1_um = hc1_us = hc1_spike = torch.zeros(batch_size, 32, 28, 28, dtype=dtype, device=device)
        hc1_sav = torch.ones(batch_size, 32, 28, 28, dtype=dtype, device=device)
        
        pool1_t_elapse = torch.zeros_like(hc1_spike, dtype=dtype, device=device)
        pool1_latency = self.num_steps * torch.ones_like(hc1_spike, dtype=dtype, device=device)
        
        hc2_vm = hc2_vs = torch.zeros(batch_size, 32, 14, 14, dtype=dtype, device=device)
        hc2_um = hc2_us = hc2_spike = torch.zeros(batch_size, 64, 14, 14, dtype=dtype, device=device)
        hc2_sav = torch.ones(batch_size, 64, 14, 14, dtype=dtype, device=device)
        
        pool2_t_elapse = torch.zeros_like(hc2_spike, dtype=dtype, device=device)
        pool2_latency = self.num_steps * torch.ones_like(hc2_spike, dtype=dtype, device=device)
        
        hf1_vm = hf1_vs = torch.zeros(batch_size, 3136, dtype=dtype, device=device)
        hf1_um = hf1_us = hf1_spike = torch.zeros(batch_size, 600, dtype=dtype, device=device)
        hf1_sav = torch.ones(batch_size, 600, dtype=dtype, device=device)
        
        hf2_vm = hf2_vs = torch.zeros(batch_size, 600, dtype=dtype, device=device)
        hf2_um = hf2_us = hf2_u = hf2_spike = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        hf2_sav = torch.ones(batch_size, 10, dtype=dtype, device=device)        
        
        out_t = self.num_steps * torch.ones(batch_size, 10, dtype=dtype, device=device)
        out_u = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        t_elapse = torch.zeros(out_t.shape, dtype=dtype, device=device)
        sum_sp = torch.zeros(5) 
        
        
        for step in range(self.num_steps): # simulation time steps       
            early_out_check = 1.0   # early stopping check in eval mode
            
            # Input encoding : LIF neuron spike encoding
            input_u, in_spike = encoding_neuron(1.0, self.tau_m, input*self.scaling_factor, input_u, in_spike)
            
            
            # Calculation of first convolutional layer
            hc1_sav, hc1_vm, hc1_vs, hc1_um, hc1_us, hc1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv1, 
                                                                      in_spike, 
                                                                      hc1_sav, 
                                                                      hc1_vm, 
                                                                      hc1_vs, 
                                                                      hc1_spike, 
                                                                      outneuron=False)
            
            
            # Calculate spike latency of first convolutional layer
            with torch.no_grad():
                pool1_t_elapse += ((pool1_t_elapse + (hc1_um != 0)) != 0).float()
                pool1_latency += hc1_spike * (pool1_t_elapse - self.num_steps +1)
            
    
            # Calculation of second convolutional layer
            hc2_sav, hc2_vm, hc2_vs, hc2_um, hc2_us, hc2_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv2, 
                                                                      MinPool.apply(hc1_spike, pool1_latency, 2, 2),
                                                                      hc2_sav, 
                                                                      hc2_vm, 
                                                                      hc2_vs, 
                                                                      hc2_spike, 
                                                                      outneuron=False)
            
            
            # Calculate spike latency of second convolutional layer
            with torch.no_grad():
                pool2_t_elapse += ((pool2_t_elapse + (hc2_um != 0)) != 0).float()
                pool2_latency += hc2_spike * (pool2_t_elapse - self.num_steps +1)
        
            
            # Calculation of fisrt linear layer
            hf1_sav, hf1_vm, hf1_vs, hf1_um, hf1_us, hf1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.fc1, 
                                                                      MinPool.apply(hc2_spike, pool2_latency, 2, 2).view(batch_size, -1),
                                                                      hf1_sav, 
                                                                      hf1_vm, 
                                                                      hf1_vs, 
                                                                      hf1_spike, 
                                                                      outneuron=False)
            
            # Calculation of second linear layer
            hf2_sav, hf2_vm, hf2_vs, hf2_um, hf2_us, hf2_u, hf2_spike = noso(self.thresh, 
                                                                             self.tau_m, 
                                                                             self.tau_s, 
                                                                             self.fc2, 
                                                                             hf1_spike, 
                                                                             hf2_sav, 
                                                                             hf2_vm, 
                                                                             hf2_vs, 
                                                                             hf2_spike, 
                                                                             outneuron=True)
            
            
            # Least latency output decoding
            t_elapse += ((t_elapse + (hf2_u != 0)) != 0).float()
            out_t += dodt.apply(hf2_spike, t_elapse, self.num_steps)
            out_u += hf2_spike * hf2_u
            
            
            # Spike count
            sum_sp[0] += in_spike.sum().item()
            sum_sp[1] += hc1_spike.sum().item()
            sum_sp[2] += hc2_spike.sum().item()
            sum_sp[3] += hf1_spike.sum().item()
            sum_sp[4] += hf2_spike.sum().item()
            
            
            # Early stopping in eval mode
            if mode == 'eval': 
                for batch in range(batch_size): 
                    early_out_check *= (out_t[batch].min() < self.num_steps).float()
                    if early_out_check == 0.0: 
                        break
                if early_out_check == 1.0:
                    return out_t, out_u, sum_sp
            
            
            # If all output neurons spiking, end the network operation
            if out_t.max() < self.num_steps:
                return out_t, out_u, sum_sp
            
        return out_t, out_u, sum_sp



# CIFAR10 network
class CIFARnet(nn.Module):
    def __init__(self, task, thresh, tau_m, tau_s, num_steps, scaling_factor):
        super(CIFARnet, self).__init__()
        self.task = task
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.num_steps = num_steps
        self.scaling_factor = scaling_factor
        
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=False).float()
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=2, bias=False).float()
        self.conv3 = nn.Conv2d(128, 256, 5, stride=1, padding=2, bias=False).float()
        self.conv4 = nn.Conv2d(256, 512, 5, stride=1, padding=2, bias=False).float()
        self.conv5 = nn.Conv2d(512, 256, 5, stride=1, padding=2, bias=False).float()
        self.fc1 = nn.Linear(16384, 1024, bias=False).float()
        self.fc2 = nn.Linear(1024, 512, bias=False).float()
        self.fc3 = nn.Linear(512, 10, bias=False).float()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight.data)


    def forward(self, input, batch_size, mode):
        input_u = in_spike = torch.zeros(batch_size, 3, 32, 32, dtype=dtype, device=device)
        
        hc1_vm = hc1_vs = torch.zeros(batch_size, 3, 32, 32, dtype=dtype, device=device)
        hc1_um = hc1_us = hc1_spike = torch.zeros(batch_size, 64, 32, 32, dtype=dtype, device=device)
        hc1_sav = torch.ones(batch_size, 64, 32, 32, dtype=dtype, device=device)
        
        hc2_vm = hc2_vs = torch.zeros(batch_size, 64, 32, 32, dtype=dtype, device=device)
        hc2_um = hc2_us = hc2_spike = torch.zeros(batch_size, 128, 32, 32, dtype=dtype, device=device)
        hc2_sav = torch.ones(batch_size, 128, 32, 32, dtype=dtype, device=device)
        
        pool1_t_elapse = torch.zeros(hc2_spike.shape, dtype=dtype, device=device)
        pool1_latency = self.num_steps * torch.ones(hc2_spike.shape, dtype=dtype, device=device)
        
        hc3_vm = hc3_vs = torch.zeros(batch_size, 128, 16, 16, dtype=dtype, device=device)
        hc3_um = hc3_us = hc3_spike = torch.zeros(batch_size, 256, 16, 16, dtype=dtype, device=device)
        hc3_sav = torch.ones(batch_size, 256, 16, 16, dtype=dtype, device=device)
        
        pool2_t_elapse = torch.zeros(hc3_spike.shape, dtype=dtype, device=device)
        pool2_latency = self.num_steps * torch.ones(hc3_spike.shape, dtype=dtype, device=device)
        
        hc4_vm = hc4_vs = torch.zeros(batch_size, 256, 8, 8, dtype=dtype, device=device)
        hc4_um = hc4_us = hc4_spike = torch.zeros(batch_size, 512, 8, 8, dtype=dtype, device=device)
        hc4_sav = torch.ones(batch_size, 512, 8, 8, dtype=dtype, device=device)
        
        hc5_vm = hc5_vs = torch.zeros(batch_size, 512, 8, 8, dtype=dtype, device=device)
        hc5_um = hc5_us = hc5_spike = torch.zeros(batch_size, 256, 8, 8, dtype=dtype, device=device)
        hc5_sav = torch.ones(batch_size, 256, 8, 8, dtype=dtype, device=device)
        
        hf1_vm = hf1_vs = torch.zeros(batch_size,16384, dtype=dtype, device=device)
        hf1_um = hf1_us = hf1_spike = torch.zeros(batch_size, 1024, dtype=dtype, device=device)
        hf1_sav = torch.ones(batch_size, 1024, dtype=dtype, device=device)
        
        hf2_vm = hf2_vs = torch.zeros(batch_size, 1024, dtype=dtype, device=device)
        hf2_um = hf2_us = hf2_spike = torch.zeros(batch_size, 512, dtype=dtype, device=device)
        hf2_sav = torch.ones(batch_size, 512, dtype=dtype, device=device)      
        
        hf3_vm = hf3_vs = torch.zeros(batch_size, 512, dtype=dtype, device=device)
        hf3_um = hf3_us = hf3_u = hf3_spike = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        hf3_sav = torch.ones(batch_size, 10, dtype=dtype, device=device)        
        
        out_t = self.num_steps * torch.ones(batch_size, 10, dtype=dtype, device=device)
        out_u = torch.zeros(batch_size, 10, dtype=dtype, device=device)
        t_elapse = torch.zeros(out_t.shape, dtype=dtype, device=device)
        sum_sp = torch.zeros(9, device=device) 
        
        
        for step in range(self.num_steps): # simulation time steps               
            early_out_check = 1.0   # early stopping check in eval mode
            
            # Input encoding : LIF neuron encoding
            input_u, in_spike = encoding_neuron(1.0, self.tau_m, input*self.scaling_factor, input_u, in_spike)
            
            
            # Calculation of first convolutional layer
            hc1_sav, hc1_vm, hc1_vs, hc1_um, hc1_us, hc1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv1, 
                                                                      in_spike, 
                                                                      hc1_sav, 
                                                                      hc1_vm, 
                                                                      hc1_vs, 
                                                                      hc1_spike, 
                                                                      outneuron=False)
            
            # Calculation of second convolutional layer
            hc2_sav, hc2_vm, hc2_vs, hc2_um, hc2_us, hc2_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv2, 
                                                                      hc1_spike,
                                                                      hc2_sav, 
                                                                      hc2_vm, 
                                                                      hc2_vs, 
                                                                      hc2_spike, 
                                                                      outneuron=False)
            
            
            # Calculate spike latency of first convolutional layer
            with torch.no_grad():
                pool1_t_elapse += ((pool1_t_elapse + (hc2_um != 0)) != 0).float()
                pool1_latency += hc2_spike * (pool1_t_elapse - self.num_steps +1)
            
            
            # Calculation of third convolutional layer
            hc3_sav, hc3_vm, hc3_vs, hc3_um, hc3_us, hc3_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv3, 
                                                                      MinPool.apply(hc2_spike, pool1_latency, 2, 2), 
                                                                      hc3_sav, 
                                                                      hc3_vm, 
                                                                      hc3_vs, 
                                                                      hc3_spike, 
                                                                      outneuron=False)
            
            
            # Calculate spike latency of second convolutional layer
            with torch.no_grad():
                pool2_t_elapse += ((pool2_t_elapse + (hc3_um != 0)) != 0).float()
                pool2_latency += hc3_spike * (pool2_t_elapse - self.num_steps +1)
            
            
            # Calculation of fourth convolutional layer
            hc4_sav, hc4_vm, hc4_vs, hc4_um, hc4_us, hc4_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv4, 
                                                                      MinPool.apply(hc3_spike, pool2_latency, 2, 2),
                                                                      hc4_sav, 
                                                                      hc4_vm, 
                                                                      hc4_vs, 
                                                                      hc4_spike, 
                                                                      outneuron=False)
            
            
            # Calculation of fifth convolutional layer
            hc5_sav, hc5_vm, hc5_vs, hc5_um, hc5_us, hc5_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.conv5, 
                                                                      hc4_spike,
                                                                      hc5_sav, 
                                                                      hc5_vm, 
                                                                      hc5_vs, 
                                                                      hc5_spike, 
                                                                      outneuron=False)
                
            
            # Calculation of fisrt linear layer
            hf1_sav, hf1_vm, hf1_vs, hf1_um, hf1_us, hf1_spike = noso(self.thresh, 
                                                                      self.tau_m, 
                                                                      self.tau_s, 
                                                                      self.fc1, 
                                                                      hc5_spike.view(batch_size, -1), 
                                                                      hf1_sav, 
                                                                      hf1_vm, 
                                                                      hf1_vs, 
                                                                      hf1_spike, 
                                                                      outneuron=False)
            
            
            # Calculation of second linear layer
            hf2_sav, hf2_vm, hf2_vs, hf2_um, hf2_us, hf2_spike = noso(self.thresh, 
                                                                        self.tau_m, 
                                                                        self.tau_s, 
                                                                        self.fc2, 
                                                                        hf1_spike, 
                                                                        hf2_sav, 
                                                                        hf2_vm, 
                                                                        hf2_vs, 
                                                                        hf2_spike, 
                                                                        outneuron=False)
            
            
            # Calculation of third linear layer
            hf3_sav, hf3_vm, hf3_vs, hf3_um, hf3_us, hf3_u, hf3_spike = noso(self.thresh, 
                                                                                self.tau_m, 
                                                                                self.tau_s, 
                                                                                self.fc3, 
                                                                                hf2_spike, 
                                                                                hf3_sav, 
                                                                                hf3_vm, 
                                                                                hf3_vs, 
                                                                                hf3_spike, 
                                                                                outneuron=True)
            
            
            # Least latency output decoding
            t_elapse += ((t_elapse + (hf3_u != 0)) != 0).float()
            out_t += dodt.apply(hf3_spike, t_elapse, self.num_steps)
            out_u += hf3_spike * hf3_u
            
            
            # Spike count
            sum_sp[0] += in_spike.sum().item()
            sum_sp[1] += hc1_spike.sum().item()
            sum_sp[2] += hc2_spike.sum().item()
            sum_sp[3] += hc3_spike.sum().item()
            sum_sp[4] += hc4_spike.sum().item()
            sum_sp[5] += hc5_spike.sum().item()
            sum_sp[6] += hf1_spike.sum().item()
            sum_sp[7] += hf2_spike.sum().item()
            sum_sp[8] += hf3_spike.sum().item()
            
            
            # Early stopping in eval mode
            if mode == 'eval': 
                for batch in range(batch_size): 
                    early_out_check *= (out_t[batch].min() < self.num_steps).float()
                    if early_out_check == 0.0: 
                        break
                if early_out_check == 1.0:
                    return out_t, out_u, sum_sp
            
            
            # If all output neuron spiking, end the network operation
            if out_t.max() < self.num_steps:
                return out_t, out_u, sum_sp
            
        return out_t, out_u, sum_sp            



