'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time

import intel_extension_for_pytorch
#import oneccl_bindings_for_pytorch
#import torch_ccl
import horovod.torch as hvd
begining = time.time() 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'xpu' if torch.xpu.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
AVAIL_GPUS = min(1, torch.xpu.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
print("## DEVICE :", device)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



# Initialize Horovod
hvd.init()
verbose = hvd.rank() == 0
#from utils import progress_bar

# Pin GPU to be used to process local rank (one GPU per process)
torch.xpu.set_device(hvd.local_rank())

# Model
net = torchvision.models.resnet152(pretrained=False, num_classes=10)
#torch.xpu.set_device(1)
net = net.to(device) 
# ACH: ToDo: check!
#if device == 'xpu':
#    os.environ['MASTER_ADDR'] = '127.0.0.1'
#    os.environ['MASTER_PORT'] = '29500'
#    os.environ['RANK'] = os.environ.get('PMI_RANK', '-1')
#    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '-1')
#    local_rank = int(os.environ.get('PMI_RANK', '-1'))
#    print("before dist")
    #dist.init_process_group(backend='ccl', init_method='env://', world_size=-1, rank=-1)
#    dist.init_process_group(backend='gloo', init_method='env://')
#    print("after dist")
#    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0])
    #net = torch.nn.DataParallel(net) 
    #cudnn.benchmark = True

#if args.resume:
    # Load checkpoint.
#    print('==> Resuming from checkpoint..')
#    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#    checkpoint = torch.load('./checkpoint/ckpt.pth')
#    net.load_state_dict(checkpoint['net'])
#    best_acc = checkpoint['acc']
#    start_epoch = checkpoint['epoch']

steps_per_epoch = 45000
#criterion = F.nll_loss()
# ACH: ToDo: check the lr value and other hp. maybe hardcod the lr=0.05 !
#optimizer = optim.SGD(net.parameters(), lr=net.hparams.lr,momentum=0.9,weight_decay=5e-4,)
optimizer = optim.SGD(net.parameters(), lr=0.05,momentum=0.9,weight_decay=5e-4,)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=30, steps_per_epoch=steps_per_epoch)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = net(inputs)
            #loss = criterion(outputs, targets)
            loss = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        #predicted = torch.argmax(outputs, dim=1) # ACH
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        #if hvd.rank() == 0:
        #    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = net(inputs)
                #loss = criterion(outputs, targets)
                loss = F.nll_loss(outputs, targets)
          

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #predicted = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            #if hvd.rank() == 0:
            #    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    #acc = 100.*correct/total
    #if acc > best_acc:
    #    print('Saving..')
    #    state = {
    #        'net': net.state_dict(),
    #        'acc': acc,
    #        'epoch': epoch,
    #    }
    #    if not os.path.isdir('checkpoint'):
    #        os.mkdir('checkpoint')
    #    torch.save(state, './checkpoint/ckpt.pth')
    #    best_acc = acc

def main():
    for epoch in range(start_epoch, start_epoch+30):
        start = time.time()
        train(epoch)
        test(epoch)
        scheduler.step()
        end = time.time()
        print("Elapse :", end - start)


if __name__ == '__main__':
    #mp.spawn(main, nprocs=4)
    main()
    finishing = time.time()
    print("Total time to train : ", finishing - begining)

