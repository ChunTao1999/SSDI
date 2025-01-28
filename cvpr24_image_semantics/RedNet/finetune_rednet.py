# 4.20.2023 - tao88
#%% Imports
import RedNet_model
import RedNet_data
from RedNet_data import image_h, image_w
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR
# Debug
from tensorboardX import SummaryWriter
import pdb


#%% Arguments, device, configs
parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation_Train')
parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--data-dir', default=None, metavar='DIR', help='path to SUNRGB-D')
parser.add_argument('--epochs', default=1500, type=int, metavar='N', help='number of total epochs to run (default: 1500)')
parser.add_argument('-b', '--batch-size', default=5, type=int, metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--lr-decay-rate', default=0.8, type=float, help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int, help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR', help='path to save checkpoints')
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


#%% Dataset and dataloader
train_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                               RedNet_data.RandomScale((1.0, 1.4)),
                                                               RedNet_data.RandomHSV((0.9, 1.1),
                                                                                    (0.9, 1.1),
                                                                                    (25, 25)),
                                                               RedNet_data.RandomCrop(image_h, image_w),
                                                               RedNet_data.RandomFlip(),
                                                               RedNet_data.ToTensor(),
                                                               RedNet_data.Normalize()]),
                                                               phase_train=True,
                                                               data_dir=args.data_dir)
train_loader = DataLoader(train_data, 
                          batch_size=args.batch_size, 
                          shuffle=True,
                          num_workers=args.workers, 
                          pin_memory=False)
num_train = len(train_data)
print("Train Dataset size: {}".format(num_train))


#%% Model and checkpoints, Cross Entropy loss, Optimizer and LR-scheduler
if args.last_ckpt:
    model = RedNet_model.RedNet(pretrained=False)
else:
    model = RedNet_model.RedNet(pretrained=True)
CEL_weighted = utils.CrossEntropyLoss2d()
model.train()
model.to(device)
CEL_weighted.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
global_step = 0
if args.last_ckpt:
    global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

writer = SummaryWriter(args.summary_dir)
pdb.set_trace()


#%% Training (train the mask predictor on patch masks)
# for epoch in range(int(args.start_epoch), args.epochs):
for epoch in range(args.epoch):
    for batch_idx, sample in enumerate(train_loader):
        pdb.set_trace()

pdb.set_trace()