#!/usr/bin/env python
# coding: utf-8

# # Coursework 2 for Cardiac MR Image Segmentation (2020-2021)
# 
# The aim of this network is to segment images of the heart from MR imaging. The pictures are of dimension 96 x 96 and are black and white. The goal is to segment them into four different categories: myocardium, left ventricle, right ventricle, and background. 
# The data set is comprised of 

# In[1]:


from matplotlib import pyplot as plt
def show_image_mask(img, mask, cmap='gray'): # visualisation
    fig = plt.figure(figsize=(5,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')
    
#Additional function that plots the predicted segmentation along with the image and the ground truth
def show_image_mask_pred(img, mask, pred, cmap='gray'):
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


# In[2]:


import torch
import torch.utils.data as data
import cv2
import os
from glob import glob
import numpy as np

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


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSEG(nn.Module):
    def __init__(self):
        super(CNNSEG, self).__init__()
        n_class = 4
        self.conv_1 = nn.Sequential(*[
            nn.Conv2d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ]) # 1/2

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

    
#model = CNNSEG() # We can now create a model using your defined segmentation model
"""
try:
    weights = torch.load('./segnet.pth')
    model.load_state_dict(weights[0])
    best_val = weights[1]
except:
    best_val = 100
"""
best_val = 100


# In[4]:


model = CNNSEG()


# In[5]:


from torchsummary import summary
summary(model, (1, 96, 96))


# In[6]:


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


# In[7]:


#Set the learning rate
LEARNING_RATE = 0.0001

#Initialise cross entropy loss and adam optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)
#Initialise list to store loss over training
train_loss = list()
val_loss = list()


# In[8]:


#Packages proprocessing and forward pass
def forward(img):
    #scales image so that each pixel lies between 0 and 1
    img_scale = img/255.

    img_scale = img_scale.unsqueeze(1)
    optimizer.zero_grad()
    seg_soft = model(img_scale)
    
    return seg_soft


# In[ ]:


from torch.utils.data import DataLoader
import time

data_path = './data/train'
val_path = './data/val'
num_workers = 8
batch_size = 15
train_set = TrainDataset(data_path)
val_set = TrainDataset(val_path)

training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
validating_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=20, shuffle=True)


data_loader = {'train':training_data_loader,
              'validate':validating_data_loader}
EPOCHS = 50

# Fetch images and labels. 
for epochs in range(EPOCHS):
    print('-' * 10)
    print('Epoch {}/{}'.format(epochs+1, EPOCHS))

    model.train(True)
    tic = time.perf_counter()
    batch_loss = 0.0

    for index_train, sample in enumerate(data_loader['train']):
        
        img, mask = sample
        
        soft_mask = forward(img)
        
        predicted_mask = torch.argmax(soft_mask, dim=1)[0,...].squeeze().float()
        
        show_image_mask_pred(img[0,...].squeeze(), mask[0,...].squeeze(), predicted_mask)
        
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
        soft_mask = forward(img)
       
        loss = loss_fn(soft_mask, mask.long())

        batch_loss += loss.item()

    val_loss.append(batch_loss / (index_val + 1))
    toc = time.perf_counter()
    dice_scr = np.mean([categorical_dice(mask, torch.argmax(soft_mask, dim=1), label_class=c) for c in [0,1,2,3]])
    if val_loss[-1] < best_val:
        torch.save([model.state_dict(), val_loss[-1]], './fcn.pth')
        best_val = val_loss[-1]
        print("Found better")
    print("Validation loss: ", val_loss[-1], "Validating took ", toc-tic, " seconds")
    print("Dice score: ", dice_scr)


# In[16]:


plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.legend()
plt.show()


# In[17]:


trained_model = CNNSEG()
best_weights = torch.load('./segnet.pth')
trained_model.load_state_dict(best_weights[0])
trained_model.eval()

def forward_trained(img):
    img_scale = img/255.

    img_scale = img_scale.view(1, 96, 96)
    seg_soft = trained_model(img_scale)
    
    return seg_soft


# In[17]:


for idx in range(20):
    fig = plt.figure(figsize=(10,10))
    plt.subplot(1, 4, 1)
    plt.imshow(img[idx], cmap='gray')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(mask[idx], cmap='gray')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(torch.argmax(forward_trained(img), dim=1)[idx], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(mask[idx] == torch.argmax(soft_mask, dim=1)[idx], cmap='gray')
    plt.axis('off')
    plt.show()
    print(np.mean([categorical_dice(mask[idx], torch.argmax(forward_trained(img), dim=1)[idx], label_class=c) for c in [0,1,2,3]]))


# In[21]:


import pickle

outfile = open('segnet_pickle', 'wb')
pickle.dump(trained_model, outfile)
outfile.close()


# In[18]:


data_path = './data/test'
num_workers = 8
batch_size = 1
test_set = TestDataset(data_path)

test_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)


# In[35]:


for iteration, sample in enumerate(test_data_loader):
    img = sample[0]
    
    seg_soft = forward_trained(img)

    show_image_mask(img[0,...].squeeze(), torch.argmax(seg_soft, dim=1)[0,...].squeeze().float())
    


# In[ ]:




