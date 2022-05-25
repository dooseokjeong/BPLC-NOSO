import torch
import torch.nn.functional as F
import numpy as np

# Define the gradient of spike timing with membrane potential 
class dtdu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, um, us, thresh, tau_m, tau_s):
        ctx.thresh = thresh
        ctx.tau_m = tau_m 
        ctx.tau_s = tau_s
        ctx.save_for_backward(u, um, us)
        return (u.gt(thresh).float() + u.lt(-thresh).float())   # Apply dual threshold

    @staticmethod
    def backward(ctx, grad_output):
        u, um, us = ctx.saved_tensors
        thresh = ctx.thresh
        tau_m = ctx.tau_m 
        tau_s = ctx.tau_s
        temp = um / tau_m - us / tau_s
        temp = torch.where(temp != 0, 1/temp, temp)
        grad_u = grad_output * (u.gt(thresh).float() + u.lt(-thresh).float()) * torch.clamp(temp, -100, 100)
        return grad_u, None, None, None, None, None


# Define the gradient of v with spike timing 
class dvdt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, x, tau, const):
        v = v * np.exp(-1 / tau) + const * x
        ctx.tau = tau
        ctx.const = const
        return v

    @staticmethod
    def backward(ctx, grad_output):
        tau = ctx.tau
        const = ctx.const
        grad_input = grad_output.clone()
        grad_t = const * grad_input / tau
        grad_v = grad_input * np.exp(-1 / tau)
        return grad_v, grad_t, None, None


# Define output latency function
class dodt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output_spike, t_elapse, num_steps):
        ctx.save_for_backward(output_spike)
        return output_spike * (t_elapse - num_steps)

    @staticmethod
    def backward(ctx, grad_output):
        output_spike, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * output_spike, None, None

    
# NOSO model definition
def noso(thresh, tau_m, tau_s, ops, x, sav, vm, vs, spike, outneuron=False):
    const = tau_m / (tau_m - tau_s)
    vm = dvdt.apply(vm, x, tau_m, const)
    vs = dvdt.apply(vs, x, tau_s, const)
    sav = sav * (1. - spike)
    um = ops(vm)
    us = ops(vs)
    u = (um - us) * sav
    spike = dtdu.apply(u, um, us, thresh, tau_m, tau_s)

    if outneuron==False:
        return sav, vm, vs, um, us, spike
    
    return sav, vm, vs, um, us, u, spike


# Encoding LIF neuron
def encoding_neuron(thres, tau, x, u, spike):
    u = u*(1-spike)
    u = u * np.exp(-1 / tau) + x
    
    spike = u.gt(thres).float()
    
    return u, spike


# Least latency pooling : MinPool
class MinPool(torch.autograd.Function) :
    
    @staticmethod
    def forward(ctx, in_spike, latency, k_size, stride) : 
        latency, indices = F.max_pool2d(-latency, kernel_size=k_size, stride=stride, return_indices=True)
        ctx.size = in_spike.size()
        ctx.k_size = k_size
        ctx.stride = stride
        ctx.indices = indices
        
        in_spike = in_spike.flatten(start_dim=2)
        return in_spike.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        
    @staticmethod
    def backward(ctx, grad_output) : 
        size = ctx.size
        k_size = ctx.k_size
        stride = ctx.stride
        indices = ctx.indices
        grad_input = grad_output.clone()
        
        return F.max_unpool2d(grad_input, indices, kernel_size=k_size, stride=stride, output_size=size), None, None, None
 

# Optimizer scheduler
def scheduler_step(opt, epoch, decay_interval, gamma) : 
    lr = []
    if epoch % decay_interval == 0 :
        for g in opt.param_groups:
            g['lr'] = g['lr']*gamma
            lr.append(g['lr'])
    
        print("=== learning rate decayed to {}.\n".format(lr))
