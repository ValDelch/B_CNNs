import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from BesselConv2d import BesselConv2d

import numpy as np
import time
import math
from sklearn.model_selection import train_test_split


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class BCNN(torch.nn.Module):

    def __init__(self, n_classes=10):
        
        super(BCNN, self).__init__()
        
        #
        # Block 1
        #

        scale = 1.

        self.block1 = nn.Sequential(
            BesselConv2d(k=9, C_in=1, C_out=int(8*scale), padding='SAME'),
            nn.BatchNorm2d(int(8*scale)),
            nn.Softsign(),
            BesselConv2d(k=7, C_in=int(8*scale), C_out=int(16*scale), padding='SAME'),
            nn.BatchNorm2d(int(16*scale)),
            nn.Softsign(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            BesselConv2d(k=7, C_in=int(16*scale), C_out=int(24*scale), padding='SAME'),
            nn.BatchNorm2d(int(24*scale)),
            nn.Softsign(),
            BesselConv2d(k=7, C_in=int(24*scale), C_out=int(24*scale), padding='SAME'),
            nn.BatchNorm2d(int(24*scale)),
            nn.Softsign(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            BesselConv2d(k=7, C_in=int(24*scale), C_out=int(32*scale), padding='SAME'),
            nn.BatchNorm2d(int(32*scale)),
            nn.Softsign(),
            BesselConv2d(k=7, C_in=int(32*scale), C_out=int(40*scale), padding='VALID'),
            nn.BatchNorm2d(int(40*scale)),
            nn.Softsign()
        )
        
        # Fully Connected
        self.fully_net = nn.Sequential(
            nn.Linear(int(40*scale), n_classes),
            nn.Softmax()
        )
    
    def forward(self, input: torch.Tensor):
        
        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)

        # Global Average Pooling
        input = torch.mean(input, dim=[2,3])

        input = self.fully_net(input)
        
        return input
    
# Model
model = BCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Epochs
n_epochs = 50
train_size = 0.8

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss function
loss_fcn = torch.nn.CrossEntropyLoss()

# Metrics
def torch_acc(y_pred, y_true):
    train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
    return train_acc

metrics = [torch_acc]

# Loading data
images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/train.amat'
images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/test.amat'

train = np.loadtxt(images_folder_train).astype(np.float32)
test = np.loadtxt(images_folder_test).astype(np.float32)

all_data = np.concatenate((train, test), axis=0)

X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                    test_size=1.-train_size, 
                                                    random_state=42, stratify=all_data[:,-1])

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

class RotMNISTDataset(Dataset):
    '''Rot-MNIST Dataset'''

    def __init__(self, image_set, label_set, transform=None):
        '''
        Args:
            image_set (numpy int): matrix containing images of 
            rot-MIST digits
            label_set (numpy int): matrix containing labels for
            the digits
        '''
        self.image_set = image_set
        self.label_set = label_set
        self.transform = transform

    def __len__(self):
        '''
        Returns length of image_set
        '''
        return len(self.image_set)

    def __getitem__(self, idx):
        '''
        Behavior: Takes a random index from the instance of Dataloader and returns
        the respective sample from the data
        Args:
            idx (int): denotes index of sample to be returned
        Returns:
            image_sample (torch tensor): 1D matrix containing the image sample
            label_sample (torch tensor): label of the respective sample
        '''

        image_sample = self.image_set[idx].astype(np.float32)
        label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

        if self.transform is not None:
            image_sample = self.transform(image_sample)

        return image_sample, label_sample

trans_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR)])

trans_valid = transforms.Compose([transforms.ToTensor()])

# creating train_loader and valid_loader
train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train)
valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid)
print(len(train_dataset), len(valid_dataset))

batch_size = 64

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)

warmups_steps = n_epochs // 5
scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=warmups_steps, t_total=n_epochs)

# Training

for epoch in range(n_epochs):
    start_time = time.time()
    print("Epoch {}/{}".format(epoch + 1, n_epochs))

    # Training 
    model.train()

    train_loss = 0.
    train_metrics = [0.] * len(metrics)
    for step, (x_batch_train, y_batch_train) in enumerate(trainloader):

        images, labels = x_batch_train.to(device), y_batch_train.to(device)
        labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

        optimizer.zero_grad()
        logits = model(images)

        loss = loss_fcn(F.log_softmax(logits, dim=1), labels)
        loss.backward()
        optimizer.step()

        for i, metric in enumerate(metrics):
            train_metrics[i] += metric(logits, labels).item() / len(trainloader)
        train_loss += loss.item() / len(trainloader)
    
    scheduler.step()

    # Validation
    model.eval()

    val_loss = 0.
    test_metrics = [0.] * len(metrics)
    for step, (x_batch_test, y_batch_test) in enumerate(validloader):

        images, labels = x_batch_test.to(device), y_batch_test.to(device)
        labels = labels.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

        logits = model(images)

        loss = loss_fcn(F.log_softmax(logits, dim=1), labels)

        for i, metric in enumerate(metrics):
            test_metrics[i] += metric(logits, labels).item() / len(validloader)
        val_loss += loss.item() / len(validloader)

    print("Training metrics:", train_metrics, "; Training loss: %.4f" % float(train_loss), 
          "; Testing metrics:", test_metrics, "; Testing loss: %.4f" % float(val_loss), 
          "; Learning rate: %.6f" % float(optimizer.param_groups[0]['lr']))
    
    print("Time taken: %.2fs" % (time.time() - start_time))