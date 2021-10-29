
import argparse
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch.optim as optim
from terminaltables import AsciiTable
from torch.optim import lr_scheduler

from utils import set_random_seed, Logger,accuracy, AverageMeter
from mobilenetV3 import MobileNetV3_Small
from dataset import ImageDataLoader


def run():
    parser = argparse.ArgumentParser(description="Trains the mobilenetv3 model.") 
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--batch_size', type=int, default=256, help='total batch size for all GPUs')
    parser.add_argument("--logdir", type=str, default="./log", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("-d", "--data_root", type=str, default="/media/space/ILSVRC2012/train", help="Path to data  file ")
    parser.add_argument("--seed", type=int, default=10, help="Makes results reproducable.")
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--input_size', nargs='+', type=int, default=224, help='image sizes')

    args = parser.parse_args()
    print(f"Command line arguments: {args}")


    logger = Logger(args.logdir)  # Tensorboard logger
    
    set_random_seed(seed=args.seed)

    data_loader = ImageDataLoader(args) # Loading data
    
    print("DataLoader size: ", len(data_loader))


    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3_Small().to(device) # Loading Model
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(
        params,
        lr=0.1, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    # optimizer = optim.Adam(
    #     params, 
    #     lr=0.1, 
    #     betas=(0.9, 0.999)
    # )  # adjust beta1 to momentum

    
    #lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0, last_epoch=-1)
    #scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss().cuda()

   

    if args.resume:
        path_checkpoint = "./checkpoints/ckpt_10.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        lr_schedule.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # 设置开始的epoch
    else:
        start_epoch = 0

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    print("\n  -----Training Model------")
    for epoch in range(start_epoch , 100):
        model.train()
        for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(data_loader, desc=f"Training Epoch {epoch}")):
            # Reset gradients
            
            optimizer.zero_grad()
            batches_done = len(data_loader) * epoch + batch_i

            imgs = imgs.to(device)
            targets = targets.to(device)
            

            outputs = model(imgs)
            
            loss = criterion(outputs, targets)

            loss.backward()
            # Run optimizer
            optimizer.step()
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), imgs.size(0))
            top5.update(acc5.item(), imgs.size(0))
            losses.update(loss.item(), imgs.size(0))

            print(AsciiTable(
                [
                    ["Type", "Value"],
                    ["Batch loss", loss.detach().cpu().item()],
                    ["top1 accuracy", top1.avg],
                    ["top5 accuracy", top5.avg]
                ]).table)
            #Tensorboard logging
            tensorboard_log = [("train/loss", loss.detach().cpu().item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)
            logger.scalar_summary("top1", top1.avg, batches_done)
            logger.scalar_summary("top5", top5.avg, batches_done)
            logger.scalar_summary("losses", losses.avg, batches_done)
            logger.scalar_summary("learning_rate", lr_schedule.get_last_lr()[0], batches_done)
        
        #if epoch % 10 == 0:
        checkpoint_path = f"checkpoints/ckpt_{epoch}.pth"
        print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'scheduler_state_dict': lr_schedule.state_dict()
                    }
        torch.save(checkpoint, checkpoint_path)

        


        lr_schedule.step()

    
if __name__ == "__main__":
    # x = torch.randn(2,3,224,224)
    # y = net(x)
    # print(y.size())
    run()