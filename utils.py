
import torch
import random
import numpy as np

import os
import datetime
from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed = 10,deterministic=True,benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        # 保存标量，tag为标签，value是此次数值，step是时间节点
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        # 保存多个标量值，tag_value_pairs是tag和value做为数据对存放在其中
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
            



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
