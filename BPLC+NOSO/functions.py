import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor

# Define the gradient of spike timing with membrane potential 
class dtdu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, um, us, thresh, tau_m, tau_s):
        temp = um / tau_m - us / tau_s
        temp = torch.where(temp != 0, 1/temp, temp)
        temp = torch.clamp(temp, -100, 100)
        spike = (u.gt(thresh).float() + u.lt(-thresh).float())    # Apply dual threshold
        ctx.save_for_backward(temp, spike)
        return spike  

    @staticmethod
    def backward(ctx, grad_output):
        temp, spike = ctx.saved_tensors
        grad_u = grad_output * spike * temp
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
    um = ops(vm)
    us = ops(vs)
    u = (um - us) * sav.clone()
    spike = dtdu.apply(u, um, us, thresh, tau_m, tau_s)

    if outneuron == False:
        return vm, vs, um, us, spike
    
    return vm, vs, um, us, u, spike


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


# learning rate warm-up
def lr_warmup(opt, epoch, alpha, end_lr) : 
    if opt.param_groups[0]['lr'] < end_lr and epoch <= 50:
        for g in opt.param_groups:
            g['lr'] = g['lr'] + alpha
            g['lr'] = np.clip(g['lr'], 0, end_lr)
            lr = g['lr']
        
            print("=== Lr warm up, learning rate increased to {}\n".format(lr))
            del(lr)


# Revised Xavier_uniform initialization for CIFARnet
def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xav_uniform_rev_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(1.0 / float(fan_in + fan_out))
    a = math.sqrt(gain) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)
