import json
from datetime import datetime

import torch
import torch.nn.functional as F
from pathlib import Path


def read_lumi_filenames(root):
    return list(Path(root).glob("*.png"))


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()



def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def calc_loss(pred, target, ssim_loss):
    mse = F.mse_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    loss = ssim_out + 5*mse
    return loss
