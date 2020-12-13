from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import cv2
import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import time
from torchsummary import summary

def show_image_mask(img, mask, cmap='gray'): # visualisation
    fig = plt.figure(figsize=(5,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')
    
def show_image_mask_pred(img, mask, pred, cmap='gray'): # visualisation
    fig = plt.figure(figsize=(5,5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap=cmap)
    plt.axis('off')
    plt.show()

    
class TrainDataset(data.Dataset):
    def __init__(self, root=''):
        super(TrainDataset, self).__init__()
        self.img_files = glob(os.path.join(root,'image','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root,'mask',basename[:-4]+'_mask.png'))
            

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)

class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root,'image','*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)
    
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class FCN(nn.Module): # Define your model
    def __init__(self):
        super(FCN, self).__init__()
        # fill in the constructor for your model here
         #conv
        #For FCN16s
        n_class=21
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/2

        #conv2
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)            
        ) # 1/4

        #conv3
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/8
        
        #conv4
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/16

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/32

        #fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        #fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_pool5 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1)
        )
        self.score_pool4 = nn.Sequential(
            nn.Conv2d(512, n_class, 1)
        )

        
        self.upscore2x = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        )
        self.upscore16x = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 32, stride=16, bias=False)
        )
        
    def forward(self, x):
        # fill in the forward function for your model here
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)

        pool4_output = self.score_pool4(output)

        output = self.conv_5(output)
        output = self.fc6(output)
        output = self.fc7(output)

        pool5_output = self.score_pool5(output)
        pool5_output = self.upscore2x(pool5_output)
        fused_output = pool5_output + pool4_output[:, :, 5 : 5 + pool5_output.size()[2], 5 : 5 + pool5_output.size()[3]]

        output = self.upscore16x(fused_output)
        output = output[:, :, 27 : 27 + x.size()[2], 27 : 27 + x.size()[3]] 
        return output
def forward(img):

    img_scale = img/255.

    img_scale = img_scale.unsqueeze(1)
    optimizer.zero_grad()
    seg_soft = model(img_scale)
    
    return seg_soft


def categorical_dice(mask1, mask2, label_class=1):
    """
    Dice score of a specified class between two volumes of label masks.
    (classes are encoded but by label class number not one-hot )
    Note: stacks of 2D slices are considered volumes.

    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).numpy()
    mask2_pos = (mask2 == label_class).numpy()
    dice = 2 * np.sum(mask1_pos * mask2_pos) / (np.sum(mask1_pos) + np.sum(mask2_pos))
    return dice


model = FCN()  # We can now create a model using your defined segmentation model

LEARNING_RATE = 0.001

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.is_available():
    model.cuda()
train_loss = list()
val_loss = list()

data_path = './data/train'
val_path = './data/val'
num_workers = 0
batch_size = 5
train_set = TrainDataset(data_path)
val_set = TrainDataset(val_path)

training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
validating_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=20, shuffle=True)


data_loader = {'train':training_data_loader,
              'validate':validating_data_loader}

EPOCHS = 0
Max_Epoch = 50

tol = 1e-5 #Tolerance perameter
print('Loss function= ', loss_fn )
print('optimizer = ',optimizer)
print('Mini Batch size set to = ' ,batch_size)
print(device)

# Fetch images and labels. 
exit = False
while exit == False:
    print('-' * 10)
    print('EPOCHS = ',EPOCHS)

    model.train(True)
    tic = time.perf_counter()
    batch_loss = 0.0

    for index_train, sample in enumerate(data_loader['train']):
        
        img, mask = sample
        
        img,mask = img.to(device), mask.to(device)
        
        soft_mask = forward(img)
        
        predicted_mask = torch.argmax(soft_mask, dim=1)[0,...].squeeze().float()
        
       # show_image_mask_pred(img[0,...].squeeze(), mask[0,...].squeeze(), predicted_mask)
        
        loss = loss_fn(soft_mask, mask.long())

        loss.backward()
        optimizer.step()

        batch_loss += loss.item()

    train_loss.append(batch_loss / (index_train + 1))
    toc = time.perf_counter()
    print("Train loss: ", train_loss[-1], " Training took ", toc-tic, " seconds")
    model.train(False)

    batch_loss = 0.0
    tic = time.perf_counter()
    for index_val, sample in enumerate(data_loader['validate']):
        img, mask = sample
        img, mask = img.to(device), mask.to(device)
        soft_mask = forward(img)
       
        loss = loss_fn(soft_mask, mask.long())

        batch_loss += loss.item()

    val_loss.append(batch_loss / (index_val + 1))
    toc = time.perf_counter()
    print("Validation loss: ", val_loss[-1], "Validating took ", toc-tic, " seconds")
    if EPOCHS > 2:
        if (np.absolute((val_loss[-1] - val_loss[-2])) < tol) or (EPOCHS == Max_Epoch ):
            exit = True
    EPOCHS += 1
if (EPOCHS-1) == Max_Epoch:
    print('Failed to converge')
else:
    print('Converged in = ',EPOCHS-1)
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#categorical_dice(mask[0,...].squeeze(), torch.argmax(soft_mask, dim=1)[0,...].squeeze())
data_path = './data/test'
test_set = TestDataset(data_path)

test_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
for iteration, sample in enumerate(training_data_loader):
    img = sample[0]
    img = img.to(device)
    img_scale = img/255.
    #show_image_mask(img[0,...].squeeze(), mask[0,...].squeeze()) #visualise all data in training set
    plt.pause(1)

    # Write your FORWARD below
    # Note: Input image to your model and ouput the predicted mask and Your predicted mask should have 4 channels

    img_scale = img_scale.unsqueeze(1)
    seg_soft = model(img_scale)

summary(model, (1, 96, 96))
