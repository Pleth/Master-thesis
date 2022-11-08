import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import h5py
import matplotlib.pyplot as plt
from bisect import bisect_left

import csv

from functions import *
from LiRA_functions import *

# pytorch cnn for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
import skimage.measure
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn import BatchNorm2d
from torch.optim import SGD
from torch.optim import Adam
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

import torch
import torch.nn as nn
import skimage.io as sk
from torchvision import transforms
from PIL import ImageTk, Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labelsFile, rootDir, sourceTransform, nr_tar):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.sourceTransform = sourceTransform
        self.nr = nr_tar
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['cwt_path'][idx]
        image = np.load(imagePath)
        image = image['Wx']
        # image = sk.imread(imagePath)
        if self.nr == 1:
            label = self.data['DI'][idx]
        else:
            label = [None] * 4
            label[0] = self.data['DI'][idx]
            label[1] = self.data['Cracks'][idx]
            label[2] = self.data['Alligator'][idx]
            label[3] = self.data['Potholes'][idx]
            label = torch.Tensor(label)
        # image = Image.fromarray(image)
        data_max = 1.842777
        data_min = 5.690932e-09
        image = (image - data_min)/(data_max-data_min)

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label


# model definition
class CNN_simple(Module):
    # define model elements
    def __init__(self, n_channels):
        super(CNN_simple, self).__init__()
        # input to first hidden layer
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        # second hidden layer
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        # fully connected layer
        self.hidden3 = Linear(54144, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(100, 4)
        # xavier_uniform_(self.hidden4.weight)
        # self.act4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        # print(X.shape)
        X = X.view(X.size(0), -1)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        # X = self.act4(X)
        return X

def prepare_data(path,labelsFile,batch_size,nr_tar,test_nr):
    # define standardization
    sourceTransform = Compose([ToTensor()])
    
    if test_nr < 100:
        split1 = CustomDataset(labelsFile+"_split1.csv", path+'/split1/', sourceTransform, nr_tar)
        split2 = CustomDataset(labelsFile+"_split2.csv", path+'/split2/', sourceTransform, nr_tar)
        split3 = CustomDataset(labelsFile+"_split3.csv", path+'/split3/', sourceTransform, nr_tar)
        split4 = CustomDataset(labelsFile+"_split4.csv", path+'/split4/', sourceTransform, nr_tar)
        split5 = CustomDataset(labelsFile+"_split5.csv", path+'/split5/', sourceTransform, nr_tar)
        split6 = CustomDataset(labelsFile+"_split6.csv", path+'/split6/', sourceTransform, nr_tar)
        split7 = CustomDataset(labelsFile+"_split7.csv", path+'/split7/', sourceTransform, nr_tar)
        
    if test_nr == 0:
        train_dl = DataLoader(ConcatDataset([split3,split4,split5,split6,split7]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split2, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split1, batch_size=batch_size, shuffle=False)
    if test_nr == 1:
        train_dl = DataLoader(ConcatDataset([split1,split2,split5,split6,split7]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split3, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split4, batch_size=batch_size, shuffle=False)
    if test_nr == 2:
        train_dl = DataLoader(ConcatDataset([split1,split4,split5,split6,split7]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split3, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split2, batch_size=batch_size, shuffle=False)
    if test_nr == 3:
        train_dl = DataLoader(ConcatDataset([split1,split2,split3,split4,split5]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split6, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split7, batch_size=batch_size, shuffle=False)
    if test_nr == 4:
        train_dl = DataLoader(ConcatDataset([split1,split2,split3,split4,split7]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split5, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split6, batch_size=batch_size, shuffle=False)
    if test_nr == 5:
        train_dl = DataLoader(ConcatDataset([split2,split3,split4,split5,split6]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split1, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split7, batch_size=batch_size, shuffle=False)
    if test_nr == 6:
        train_dl = DataLoader(ConcatDataset([split1,split2,split3,split6,split7]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(split5, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(split4, batch_size=batch_size, shuffle=False)

    if test_nr == 112:
        train = CustomDataset(labelsFile+"_train.csv", path+'/train/', sourceTransform, nr_tar)
        val = CustomDataset(labelsFile+"_val.csv", path+'/val/', sourceTransform, nr_tar)
        test = CustomDataset(labelsFile+"_test.csv", path+'/test/', sourceTransform, nr_tar)
        
        train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, test_dl

# train the model
def train_model(train_dl, val_dl,test_dl, model, epochs, lr,mom=0.9,wd=0.0,id=id):
    # define the optimization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    max_acc = -5
    loss_save = []
    R2_train = []
    R2_val = []
    R2_test = []
    criterion = MSELoss()
    # criterion = CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=lr, momentum=mom,weight_decay=wd)
    optimizer = Adam(model.parameters(), lr=lr,weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.92)
    # enumerate epochs
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            # yhat, aux1, aux2 = model(inputs)
            yhat = model(inputs)
            # calculate loss
            # loss = criterion(yhat, targets) + 0.3 * criterion(aux1,targets) + 0.3 * criterion(aux2,targets)
            # loss = criterion(yhat,targets)
            if len(targets.size()) == 1:
                loss = criterion(yhat,targets.unsqueeze(1).float())
            else:
                loss = criterion(yhat,targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

            epoch_loss += yhat.shape[0] * loss.item()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # print epoch loss
        print(epoch+1, epoch_loss / len(train_dl.dataset))
        loss_save.append(epoch_loss / len(train_dl.dataset))

        if ((epoch+1) % 10 == 0) & ((epoch+1)>=2):
            model.eval()
            with torch.no_grad():
                if yhat.size()[1] == 1:
                    acc_t = evaluate_model(train_dl, model)
                    print('Train R2 - DI: %.3f' % acc_t)
                    R2_train.append(acc_t)

                    acc_v = evaluate_model(val_dl, model)
                    print('Val R2 - DI: %.3f' % acc_v)
                    R2_val.append(acc_v)

                    acc = evaluate_model(test_dl, model)
                    print('Test R2 - DI: %.3f' % acc)
                    R2_test.append(acc)
                else:
                    acc = evaluate_model(train_dl, model)
                    print('Train R2 - DI: %.3f' % acc[0] + ' Cracks: %.3f' % acc[1] + ' Alligator: %.3f' % acc[2] + ' Potholes: %.3f' % acc[3])

                    acc = evaluate_model(val_dl, model)
                    print('Val R2 - DI: %.3f' % acc[0] + ' Cracks: %.3f' % acc[1] + ' Alligator: %.3f' % acc[2] + ' Potholes: %.3f' % acc[3])

            if acc_v > max_acc:
                torch.save(model.state_dict(), "models/model_"+id+".pt")
                max_acc = acc_v

            with open("training/R2_train_"+id+".csv", 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(R2_train)

            with open("training/R2_val_"+id+".csv", 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(R2_val)

            with open("training/R2_test_"+id+".csv", 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(R2_test)

        with open("training/loss_save_"+id+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(loss_save)

    scheduler.step()
    print(max_acc)

# evaluate the model
def evaluate_model(test_dl, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        # evaluate the model on the test set
        # yhat, _, _ = model(inputs)
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        # yhat = argmax(yhat, axis=1)
        # reshape for stacking
        # loss = criterion(yhat, targets) + 0.3 * criterion(aux1,targets) + 0.3 * criterion(aux2,targets)
        if np.shape(yhat)[1] == 1:
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
        else:
            actual = actual.reshape((len(actual), 4))
            yhat = yhat.reshape((len(yhat), 4))
        
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = r2_score(actuals, predictions) #,multioutput='raw_values'
    return acc

class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(out_fts,eps=0.001),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.convolution(input_img)

        return x

class ReduceConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts_1, out_fts_2, k, p):
        super(ReduceConvBlock, self).__init__()
        self.redConv = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts_1, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_fts_1,eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_fts_1, out_channels=out_fts_2, kernel_size=(k, k), stride=(1, 1), padding=(p, p)),
            nn.BatchNorm2d(out_fts_2,eps=0.001),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.redConv(input_img)

        return x

class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.batchnorm = nn.BatchNorm2d(128,eps=0.001)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.avgpool(input_img)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

class InceptionModule(nn.Module):
    def __init__(self, curr_in_fts, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(curr_in_fts, f_1x1, 1, 1, 0)
        self.conv2 = ReduceConvBlock(curr_in_fts, f_3x3_r, f_3x3, 3, 1)
        self.conv3 = ReduceConvBlock(curr_in_fts, f_5x5_r, f_5x5, 5, 2)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=curr_in_fts, out_channels=f_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(f_pool_proj,eps=0.001),
            nn.ReLU()
        )

    def forward(self, input_img):
        out1 = self.conv1(input_img)
        out2 = self.conv2(input_img)
        out3 = self.conv3(input_img)
        out4 = self.pool_proj(input_img)

        x = torch.cat([out1, out2, out3, out4], dim=1)

        return x

class MyGoogleNet(nn.Module):
    def __init__(self, in_fts=3, num_class=1000):
        super(MyGoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1)
        )

        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux_classifier1 = AuxClassifier(512, num_class)
        self.aux_classifier2 = AuxClassifier(528, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, num_class)
        )

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        # x = self.inception_4b(x)
        # x = self.inception_4c(x)
        # x = self.inception_4d(x)
        # out2 = self.aux_classifier2(x)
        # x = self.inception_4e(x)
        # x = self.maxpool1(x)
        # x = self.inception_5a(x)
        # x = self.inception_5b(x)
        # x = self.avgpool(x)
        # x = x.reshape(N, -1)
        # x = self.classifier(x)
        # if self.training == True:
        #     return [x, out1, out2]
        # else:
        #     return x
        return out1

def create_cwt_data(GM_segments,splits,targets,path,check):
    from ssqueezepy import cwt
    # from ssqueezepy.visuals import imshow
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['1']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split1/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split1.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['2']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split2/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split2.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['3']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split3/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split3.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['4']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split4/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split4.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['5']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split5/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split5.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['6']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split6/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split6.csv",index=False)

    DI_lab = []
    cracks_lab = []
    alligator_lab = []
    pothole_lab = []
    paths = []
    for i in tqdm(splits['7']):
        xtest = np.array(GM_segments[i])
        Wx, scales = cwt(xtest, 'bump',nv=31)
        Wx = abs(Wx)
        Wx = Wx[2:]
        Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
        np.savez_compressed(path+"/split7/cwt_"+str(i).zfill(4),Wx=Wx_sum)
        
        DI_lab.append(targets[0][i])
        cracks_lab.append(targets[1][i])
        alligator_lab.append(targets[2][i])
        pothole_lab.append(targets[3][i])
        paths.append("cwt_"+str(i).zfill(4)+".npz")

    d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    df = pd.DataFrame(d)
    df.to_csv(path+"/labelsfile_split7.csv",index=False)

    # DI_lab = []
    # cracks_lab = []
    # alligator_lab = []
    # pothole_lab = []
    # paths = []
    # for i in tqdm(splits['2']):
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot()
    #     xtest = np.array(GM_segments[i])
    #     Wx, scales = cwt(xtest, 'bump',nv=31)
    #     Wx = abs(Wx)
    #     Wx = Wx[2:]
    #     Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
    #     np.savez_compressed(path+"/val/cwt_"+str(i).zfill(4),Wx=Wx_sum)
    #     # imshow(Wx, yticks=scales, abs=1)
    #     # _ = ax.axis(False)
    #     # ax.set_position([0,0,1,1])
    #     # fig.savefig(path+"/val/cwt_im_"+str(i).zfill(4)+".png")
    #     # plt.close(fig)

    #     DI_lab.append(targets[0][i])
    #     cracks_lab.append(targets[1][i])
    #     alligator_lab.append(targets[2][i])
    #     pothole_lab.append(targets[3][i])
    #     paths.append("cwt_"+str(i).zfill(4)+".npz")

    # d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    # df = pd.DataFrame(d)
    # df.to_csv(path+"/labelsfile_val.csv",index=False)
    
    # if check == 'real':
    #     train_split = splits['3'] + splits['4']
    # else:
    #     train_split = splits['1'] + splits['4'] + splits['5'] + splits['6']

    # DI_lab = []
    # cracks_lab = []
    # alligator_lab = []
    # pothole_lab = []
    # paths = []
    # for i in tqdm(train_split):
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot()
    #     xtest = np.array(GM_segments[i])
    #     Wx, _ = cwt(np.array(xtest), 'bump',nv=31)
    #     Wx = abs(Wx)
    #     Wx = Wx[2:]
    #     Wx_sum = skimage.measure.block_reduce(Wx, (1,4), np.sum)
    #     np.savez_compressed(path+"/train/cwt_"+str(i).zfill(4),Wx=Wx_sum)
    #     # imshow(Wx, yticks=scales, abs=1)
    #     # _ = ax.axis(False)
    #     # ax.set_position([0,0,1,1])
    #     # fig.savefig(path+"/train/cwt_im_"+str(i).zfill(4)+".png")
    #     # plt.close(fig)

    #     DI_lab.append(targets[0][i])
    #     cracks_lab.append(targets[1][i])
    #     alligator_lab.append(targets[2][i])
    #     pothole_lab.append(targets[3][i])
    #     paths.append("cwt_"+str(i).zfill(4)+".npz")

    # d = {'cwt_path': paths, 'DI': DI_lab, 'Cracks': cracks_lab, 'Alligator': alligator_lab, 'Potholes': pothole_lab}
    # df = pd.DataFrame(d)
    # df.to_csv(path+"/labelsfile_train.csv",index=False)

    return

def DL_splits(aran_segments,route_details,cut):

    cph1 = []
    cph6 = []
    for i in range(len(route_details)):
        if (route_details[i][:7] == 'CPH1_VH') | (route_details[i][:7] == 'CPH1_HH'):
            cph1.append(i)
        elif  (route_details[i][:7] == 'CPH6_VH') | (route_details[i][:7] == 'CPH6_HH'):
            cph6.append(i)

    cph1_aran = {}
    for i in cph1:
        cph1_aran[i] = aran_segments.loc[i]
    cph1_aran = pd.concat(cph1_aran)

    cph6_aran = {}
    for i in cph6:
        cph6_aran[i] = aran_segments.loc[i]
    cph6_aran = pd.concat(cph6_aran)

    cph1_aran.index = cph1_aran.index.get_level_values(0)
    cph6_aran.index = cph6_aran.index.get_level_values(0)

    split1 = []
    for i in list(cph1_aran[cph1_aran['BeginChainage'] < cut[0]].index):
       if i not in split1:
          split1.append(i)
    split2 = []
    for i in list(cph1_aran[(cph1_aran['BeginChainage'] >= cut[0]) & (cph1_aran['BeginChainage'] <= cut[1]) ].index):
       if i not in split2:
          split2.append(i)
    split3 = []
    for i in list(cph1_aran[(cph1_aran['BeginChainage'] >= cut[1]) & (cph1_aran['BeginChainage'] <= cut[2]) ].index):
       if i not in split3:
          split3.append(i)
    split4 = []
    for i in list(cph1_aran[cph1_aran['BeginChainage'] > cut[2]].index):
       if i not in split4:
          split4.append(i)
    
    split5 = []
    for i in list(cph6_aran[(cph6_aran['BeginChainage'] < cut[3]) & (cph6_aran['BeginChainage'] > 7500) ].index):
       if i not in split5:
          split5.append(i)
    split6 = []
    for i in list(cph6_aran[(cph6_aran['BeginChainage'] >= cut[3]) & (cph6_aran['BeginChainage'] <= cut[4]) & (cph6_aran['BeginChainage'] > 7500)].index):
       if i not in split6:
          split6.append(i)
    split7 = []
    for i in list(cph6_aran[(cph6_aran['BeginChainage'] >= cut[4]) & (cph6_aran['BeginChainage'] > 7500)].index):
       if i not in split7:
          split7.append(i)
    
    splits = {'1': split1, '2': split2, '3': split3, '4': split4, '5': split5, '6': split6, '7': split7}

    return splits

def GM_sample_segmentation(segment_size=150, overlap=0):
    sz = str(segment_size)
    if os.path.isfile("aligned_data/sample/"+"aran_segments_"+sz+".csv"):
        synth_segments = pd.read_csv("aligned_data/sample/"+"synthetic_segments_"+sz+".csv")
        synth_segments.columns = synth_segments.columns.astype(int)
        aran_segment_details = pd.read_csv("aligned_data/sample/"+"aran_segments_"+sz+".csv",index_col=[0,1])
        route_details = eval(open("aligned_data/sample/routes_details_"+sz+".txt", 'r').read())
        with open("aligned_data/sample/distances_"+sz+".csv", newline='') as f:
            reader = csv.reader(f)
            temp = list(reader)
        dists = [float(i) for i in temp[0]]
        print("Loaded already segmented data")
    else:
        files = glob.glob("aligned_data/*.hdf5")
        iter = 0
        segments = {}
        aran_segment_details = {}
        route_details = {}
        dists = []
        for j in tqdm(range(len(files))):
            route = files[j][13:]
            hdf5_route = ('aligned_data/'+route)
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])                

            aligned_passes = hdf5file.attrs['aligned_passes']
            for k in range(len(aligned_passes)):
                passagefile = hdf5file[aligned_passes[k]]
                aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])
                acc_fs_50 = pd.DataFrame(passagefile['acc_fs_50'], columns = passagefile['acc_fs_50'].attrs['chNames'])
                f_dist = pd.DataFrame(passagefile['f_dist'], columns = passagefile['f_dist'].attrs['chNames'])
                spd_veh = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])

                GM_start = aligned_gps[['lat','lon']].iloc[0].values
                aran_start_idx, _ = find_min_gps_vector(GM_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:200].values)
                GM_end = aligned_gps[['lat','lon']].iloc[-1].values
                aran_end_idx, _ = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-200:].values)
                aran_max_idx = (len(aran_location)-200)+aran_end_idx
                
                i = aran_start_idx
                while (i < (aran_max_idx-10) ):
                    aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                    GM_start_idx, start_dist = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)

                    if GM_start_idx+segment_size-1 >= len(aligned_gps):
                        break

                    GM_end = aligned_gps[['lat','lon']].iloc[GM_start_idx+segment_size-1].values
                    aran_end_idx, end_dist = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                    if start_dist < 5 and end_dist < 10 and i != aran_end_idx:
                        dfdf = aligned_gps['TS_or_Distance'][GM_start_idx:GM_start_idx+segment_size]
                        dfdf = dfdf.reset_index(drop=True)   

                        dist_seg = aligned_gps['p79_dist'][GM_start_idx:GM_start_idx+segment_size]
                        dist_seg = dist_seg.reset_index(drop=True)

                        acc_seg = acc_fs_50[((acc_fs_50['TS_or_Distance'] >= np.min(dfdf)) & (acc_fs_50['TS_or_Distance'] <= np.max(dfdf)))]
                        acc_seg = acc_seg.reset_index(drop=True)

                        spd_seg = spd_veh[((spd_veh['TS_or_Distance'] >= np.min(dfdf)) & (spd_veh['TS_or_Distance'] <= np.max(dfdf)))]
                        spd_seg = spd_seg.reset_index(drop=True)

                        stat1 = acc_seg['TS_or_Distance'].empty
                        lag = []
                        for h in range(len(dist_seg)-1):
                            lag.append(dist_seg[h+1]-dist_seg[h])        
                        large = [y for y in lag if y > 5]
                        
                        if stat1:
                            stat4 = True
                        else:
                            stat4 = False if bool(large) == False else (np.max(large) > 5)
                            stat5 = False if (len(spd_seg[spd_seg['spd_veh'] >= 20])) > 100 else True
                            stat6 = False if (len(acc_seg) == 150) else True
                            stat7 = False if abs(aran_end_idx - i) < 100 else True
                            
                        if stat1 | stat4 | stat5 | stat6 | stat7:
                            i += 1
                        else:
                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[i:aran_end_idx+1],aran_alligator[i:aran_end_idx+1],aran_cracks[i:aran_end_idx+1],aran_potholes[i:aran_end_idx+1]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                            i = aran_end_idx+1
                            iter += 1
                    else:
                        i += 1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("aligned_data/sample/"+"synthetic_segments_"+sz+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("aligned_data/sample/"+"aran_segments_"+sz+".csv",index=True)
        myfile = open("aligned_data/sample/routes_details_"+sz+".txt","w")
        myfile.write(str(route_details))
        myfile.close()
        with open("aligned_data/sample/distances_"+sz+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(dists)

    return synth_segments, aran_segment_details, route_details, dists


def synthetic_sample_segmentation(synth_acc,routes,segment_size=5,overlap=0):
    sz = str(segment_size)
    if os.path.isfile("synth_data/sample/"+"aran_segments_"+sz+".csv"):
        synth_segments = pd.read_csv("synth_data/sample/"+"synthetic_segments_"+sz+".csv")
        synth_segments.columns = synth_segments.columns.astype(int)
        aran_segment_details = pd.read_csv("synth_data/sample/"+"aran_segments_"+sz+".csv",index_col=[0,1])
        route_details = eval(open("synth_data/sample/routes_details_"+sz+".txt", 'r').read())
        with open("synth_data/sample/distances_"+sz+".csv", newline='') as f:
            reader = csv.reader(f)
            temp = list(reader)
        dists = [float(i) for i in temp[0]]
        print("Loaded already segmented data")              
        
    else:    
        files = glob.glob("p79/*.csv")
        df_cph1_hh = pd.read_csv('p79/CPH1_HH.csv')
        df_cph1_hh.drop(df_cph1_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
        df_cph1_vh = pd.read_csv('p79/CPH1_VH.csv')
        df_cph1_vh.drop(df_cph1_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
        df_cph6_hh = pd.read_csv('p79/CPH6_HH.csv')
        df_cph6_hh.drop(df_cph6_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
        df_cph6_vh = pd.read_csv('p79/CPH6_VH.csv')
        df_cph6_vh.drop(df_cph6_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
        iter = 0
        segments = {}
        aran_segment_details = {}
        route_details = {}
        dists = []
        for j in tqdm(range(len(routes))):
            synth = synth_acc[j]
            synth = synth[synth['synth_acc'].notna()]
            synth = synth[synth['gm_speed'] >= 20]
            synth = synth.reset_index(drop=True)
            route = routes[j][:7]

            if route == 'CPH1_HH':
                p79_gps = df_cph1_hh
                hdf5_route = ('aligned_data/'+route+'.hdf5')
                hdf5file = h5py.File(hdf5_route, 'r')
                aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
                aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
                aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
                aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
            elif route == 'CPH1_VH':
                p79_gps = df_cph1_vh
                hdf5_route = ('aligned_data/'+route+'.hdf5')
                hdf5file = h5py.File(hdf5_route, 'r')
                aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
                aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
                aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
                aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
            elif route == 'CPH6_HH':
                p79_gps = df_cph6_hh
                hdf5_route = ('aligned_data/'+route+'.hdf5')
                hdf5file = h5py.File(hdf5_route, 'r')
                aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
                aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
                aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
                aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
            elif route == 'CPH6_VH':
                p79_gps = df_cph6_vh
                hdf5_route = ('aligned_data/'+route+'.hdf5')
                hdf5file = h5py.File(hdf5_route, 'r')
                aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
                aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
                aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
                aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])


            p79_start = p79_gps[['Latitude','Longitude']].iloc[0].values
            aran_start_idx, _ = find_min_gps_vector(p79_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:100].values)
            p79_end = p79_gps[['Latitude','Longitude']].iloc[-1].values
            aran_end_idx, _ = find_min_gps_vector(p79_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-100:].values)
            aran_end_idx = (len(aran_location)-100)+aran_end_idx
            aran_max_idx = aran_end_idx
            i = aran_start_idx
            # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
            while (i < (aran_max_idx-10) ):
                aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)

                # if p79_start_idx+segment_size-1 >= len(p79_gps):
                #     break
                
                m, idx_start = take_closest(synth['Distance'],p79_gps['Distance'][p79_start_idx])  
                idx_end = idx_start+segment_size
                if idx_end >= len(synth):
                    i+=1
                    break
                new_m, idx_temp = take_closest(p79_gps['Distance'],synth['Distance'][idx_end])
                

                p79_end = p79_gps[['Latitude','Longitude']].iloc[idx_temp].values
                aran_end_idx, end_dist = find_min_gps_vector(p79_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                if start_dist < 5 and end_dist < 5 and i != aran_end_idx:
                    dfdf = p79_gps['Distance'][p79_start_idx:idx_temp+1]
                    dfdf = dfdf.reset_index(drop=True)   

                    # synth_seg = synth[((synth['Distance'] >= np.min(dfdf)) & (synth['Distance'] <= np.max(dfdf)))]
                    # synth_seg = synth_seg.reset_index(drop=True)

                    synth_seg = synth.iloc[idx_start:idx_start+segment_size]
                    synth_seg = synth_seg.reset_index(drop=True)

                    stat1 = synth_seg['Distance'].empty
                    lag = []
                    for h in range(len(synth_seg)-1):
                        lag.append(synth_seg['Distance'][h+1]-synth_seg['Distance'][h])        
                    large = [y for y in lag if y > 5]
                    
                    if stat1:
                        # stat2 = True
                        # stat3 = True
                        stat4 = True
                        stat5 = True
                        stat6 = True
                        stat7 = True
                    else:
                        # stat2 = not 40 <= (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 60
                        # stat3 = (len(synth_seg['synth_acc'])) > 5000
                        stat4 = False if bool(large) == False else (np.max(large) > 5)
                        stat5 = False if (len(synth_seg) == segment_size) else True
                        stat6 = False if abs(aran_end_idx - i) < 100 else True
                        stat7 = False if (dfdf.iloc[-1] - dfdf.iloc[0]) < 80 else True
                        
                    if stat1 | stat4 | stat5 | stat6 | stat7:
                        i += 1
                    else:
                        segments[iter] = synth_seg['synth_acc']
                        aran_concat = pd.concat([aran_location[i:aran_end_idx+1],aran_alligator[i:aran_end_idx+1],aran_cracks[i:aran_end_idx+1],aran_potholes[i:aran_end_idx+1]],axis=1)
                        aran_segment_details[iter] = aran_concat
                        route_details[iter] = routes[j]
                        dists.append(dfdf.iloc[-1] - dfdf.iloc[0])
                        i = aran_end_idx+1
                        iter += 1
                else:
                    i +=1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("synth_data/sample/"+"synthetic_segments_"+sz+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("synth_data/sample/"+"aran_segments_"+sz+".csv",index=True)
        myfile = open("synth_data/sample/routes_details_"+sz+".txt","w")
        myfile.write(str(route_details))
        myfile.close()
        with open("synth_data/sample/distances_"+sz+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(dists)
        
    return synth_segments, aran_segment_details, route_details, dists

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList.iloc[0], pos
    if pos == len(myList):
        return myList.iloc[-1], pos
    before = myList.iloc[pos - 1]
    after = myList.iloc[pos]
    if after - myNumber < myNumber - before:
        return after, pos
    else:
        return before, pos-1








def GM_sample_segmentation2(segment_size=150, overlap=0):
    sz = str(segment_size)
    if os.path.isfile("aligned_data/sample/"+"aran_segments2_"+sz+".csv"):
        synth_segments = pd.read_csv("aligned_data/sample/"+"synthetic_segments2_"+sz+".csv")
        synth_segments.columns = synth_segments.columns.astype(int)
        aran_segment_details = pd.read_csv("aligned_data/sample/"+"aran_segments2_"+sz+".csv",index_col=[0,1])
        route_details = eval(open("aligned_data/sample/routes_details2_"+sz+".txt", 'r').read())
        with open("aligned_data/sample/distances2_"+sz+".csv", newline='') as f:
            reader = csv.reader(f)
            temp = list(reader)
        dists = [float(i) for i in temp[0]]
        print("Loaded already segmented data")
    else:
        files = glob.glob("aligned_data/*.hdf5")
        iter = 0
        segments = {}
        aran_segment_details = {}
        route_details = {}
        dists = []
        for j in tqdm(range(len(files))):
            route = files[j][13:]
            hdf5_route = ('aligned_data/'+route)
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])                

            aligned_passes = hdf5file.attrs['aligned_passes']
            for k in range(len(aligned_passes)):
                passagefile = hdf5file[aligned_passes[k]]
                aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])
                acc_fs_50 = pd.DataFrame(passagefile['acc_fs_50'], columns = passagefile['acc_fs_50'].attrs['chNames'])
                f_dist = pd.DataFrame(passagefile['f_dist'], columns = passagefile['f_dist'].attrs['chNames'])
                spd_veh = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])

                GM_start = aligned_gps[['lat','lon']].iloc[0].values
                aran_start_idx, _ = find_min_gps_vector(GM_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:200].values)
                GM_end = aligned_gps[['lat','lon']].iloc[-1].values
                aran_end_idx, _ = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-200:].values)
                aran_max_idx = (len(aran_location)-200)+aran_end_idx
                
                i = aran_start_idx
                while (i < (aran_max_idx-10) ):
                    aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                    GM_start_idx, start_dist = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)

                    if GM_start_idx+segment_size-1 >= len(aligned_gps):
                        break

                    GM_end = aligned_gps[['lat','lon']].iloc[GM_start_idx+segment_size-1].values
                    aran_end_idx, end_dist = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                    if start_dist < 5 and end_dist < 10 and i != aran_end_idx:
                        dfdf = aligned_gps['TS_or_Distance'][GM_start_idx:GM_start_idx+segment_size]
                        dfdf = dfdf.reset_index(drop=True)   

                        dist_seg = aligned_gps['p79_dist'][GM_start_idx:GM_start_idx+segment_size]
                        dist_seg = dist_seg.reset_index(drop=True)

                        acc_seg = acc_fs_50[((acc_fs_50['TS_or_Distance'] >= np.min(dfdf)) & (acc_fs_50['TS_or_Distance'] <= np.max(dfdf)))]
                        acc_seg = acc_seg.reset_index(drop=True)

                        spd_seg = spd_veh[((spd_veh['TS_or_Distance'] >= np.min(dfdf)) & (spd_veh['TS_or_Distance'] <= np.max(dfdf)))]
                        spd_seg = spd_seg.reset_index(drop=True)

                        stat1 = acc_seg['TS_or_Distance'].empty
                        lag = []
                        for h in range(len(dist_seg)-1):
                            lag.append(dist_seg[h+1]-dist_seg[h])        
                        large = [y for y in lag if y > 5]
                        
                        if stat1:
                            stat4 = True
                        else:
                            stat4 = False if bool(large) == False else (np.max(large) > 5)
                            stat5 = False if (len(spd_seg[spd_seg['spd_veh'] >= 20])) > 100 else True
                            stat6 = False if (len(acc_seg) == 150) else True
                            stat7 = False if abs(aran_end_idx - i) < 100 else True
                            
                        if stat1 | stat4 | stat5 | stat6 | stat7:
                            i += 1
                        else:
                            p1_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeFrom'].iloc[aran_end_idx], aran_location['LatitudeFrom'].iloc[aran_end_idx])
                            p2_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeTo'].iloc[aran_end_idx+1], aran_location['LatitudeTo'].iloc[aran_end_idx+1])
                            if p1_dist > p2_dist:
                                aran_end_idx = aran_end_idx + 1 
                            elif p1_dist <= p2_dist:
                                aran_end_idx = aran_end_idx

                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[i:aran_end_idx],aran_alligator[i:aran_end_idx],aran_cracks[i:aran_end_idx],aran_potholes[i:aran_end_idx]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                            i = aran_end_idx
                            iter += 1
                    else:
                        i += 1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("aligned_data/sample/"+"synthetic_segments2_"+sz+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("aligned_data/sample/"+"aran_segments2_"+sz+".csv",index=True)
        myfile = open("aligned_data/sample/routes_details2_"+sz+".txt","w")
        myfile.write(str(route_details))
        myfile.close()
        with open("aligned_data/sample/distances2_"+sz+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(dists)

    return synth_segments, aran_segment_details, route_details, dists



class FeatureDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        X = X.T.values
        y = y.values
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


class LinearRegression_class(torch.nn.Module):
    
     def __init__(self, input_dim, output_dim):

        super(LinearRegression_class, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

     def forward(self, x):
        
        outputs = self.linear(x)

        return outputs

        
class ImageRegression_class(torch.nn.Module):
    
     def __init__(self, input_dim, output_dim):

        super(ImageRegression_class, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

     def forward(self, x):
        
        x = x.view(x.size(0), -1)
        outputs = self.linear(x)

        return outputs




import torch
from torch import nn


class LinearBaseline(nn.Module):
    """A PyTorch implementation of the Linear Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_inputs: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_inputs': num_inputs,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(
            nn.Dropout(0.1),
            LinearBlock(num_inputs, 250, 0.2),
            LinearBlock(250, 100, 0.2),
            LinearBlock(100, num_pred_classes, 0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x.view(x.shape[0], -1))


class LinearBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 dropout: float) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)