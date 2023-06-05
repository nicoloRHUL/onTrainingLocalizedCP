import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer

import os, sys
from os import path
import zipfile
import urllib.request
from os import listdir
from os.path import isfile, join

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm, trange
##############################################################
fileNames = {
    "concrete": "Concrete_Data.csv",
    "energy": "ENB2012_data.xlsx",
    "homes": "kc-house-data.csv", 
    "facebook_1":"Features_Variant_1.csv", 
    "CASP": "CASP.csv", 
    "cos": "nofile",
    "squared": "nofile",
    "inverse": "nofile",
    "linear": "nofile"
}

def sigmoid(x, M = 1):
    return 1/(1 + np.exp(- M * x))

def generateInput(N = 1000, d = 3):
    x = -1 + 2 * np.random.rand(N, 1) * 2
    X = []
    for i in range(d):
        q = np.power(x, i)
        X.append(q)
    return np.concatenate(tuple(X), axis=1), x

def generateOutput(X, err):
    w = np.random.randn(len(X[0]), 1)
    return X @ w + err * np.random.randn(len(X), 1)
#data
def getDataset(name):
    base_path='datasets/'
    file_name = base_path + fileNames[name]
    
    if name == 'cos':
        X, x = generateInput()
        err = .1 + 2 * np.cos(3.14/2 * abs(x)) * (abs(x)<.5)
        y = generateOutput(X, err)

    if name == 'squared':
        X, x = generateInput()
        err = .1 + 2 * x * x * (abs(x)>.5) 
        y = generateOutput(X, err)
    
    if name == 'inverse':
        X, x = generateInput()
        err = .1 + 2/ (.1 + abs(X[:, ])) * (abs(X[:, ])>.5) 
        y = generateOutput(X, err)
    
    if name == 'linear':
        X, x = generateInput()
        err = (.1  + (2 - abs(x)) * (abs(x)<.5)) 
        y = generateOutput(X, err)
    
    if name == "concrete":
        data = pd.read_csv(file_name, header=0).values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name == "energy":
        data = pd.read_excel(file_name, header=0, engine="openpyxl").values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name=="homes":
        df = pd.read_csv(file_name)
        y = np.array(df['price']).astype(np.float32)
        X = np.matrix(df.drop(['id', 'date', 'price'],axis=1)).astype(np.float32)        
  
    if name=="facebook_1":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values        
    
    if name=="CASP":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
    
    return X, y

###############################
#sklearn functions

def sklearnSplitter(X, y, seed):
    # ratios = [.4, .4, .2]
    test_size, proper_size = 0.2, 0.5
    X_1, X_test, y_1, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed)
    X_proper, X_val, y_proper, y_val = sklearn.model_selection.train_test_split(
            X_1, y_1, test_size=proper_size, random_state=seed)
    return (X_proper, y_proper), (X_val, y_val), (X_test, y_test)

def getAndSplit(name, seed):
    # X.shape = N, d
    # y.shape = N, 1
    X, y = getDataset(name)
    allSets = sklearnSplitter(X, y, seed)
    return allSets

def sklearnScaler(allSets):
    # allSets = proper, train, test
    idxTrain = 0 
    scalerX = sklearn.preprocessing.StandardScaler()
    scalerX = scalerX.fit(allSets[idxTrain][0])
    mean_ytrain = np.mean(np.abs(allSets[idxTrain][1]))
    XYsets = []
    for iData in range(len(allSets)): 
        X, y = allSets[iData]
        X, y = np.asarray(X), np.asarray(y)
        X = scalerX.transform(X)
        y = np.squeeze(y)/mean_ytrain
        Z = torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)
        XYsets.append(torch.utils.data.TensorDataset(Z[0], Z[1]))
    return XYsets

#create loaders
def getLoaders(XYsets, batch_size = 16):
    XYloaders =[]
    for iData in range(len(XYsets)): 
        dataset = XYsets[iData]
        XYloaders.append(
                torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, drop_last=True))
    return XYloaders

#get model list
def getModelList(modes):
    models = []
    for iMode in range(len(modes)):
        mode = modes[iMode]
        if mode =='fixed':
            models.append([mode, 'none'])
        else:
            if mode == 'erc':
                models.append([mode, 'erc'])
            models.append([mode, 'size'])
    return models

def split(data, random = 0, cut = 1000):
    X, y = data
    if random:
        idx = torch.randperm(len(X))
    else:
        idx = torch.tensor(range(len(X)))
    X = X[idx[:cut]], X[idx[cut:]]
    y = y[idx[:cut]], y[idx[cut:]]
    return (X[0], y[0]), (X[1], y[1])

def createValidation(loader, ratio = .2, maxVal = 100, random = 1):
    X = torch.cat(tuple([z[0] for z in loader]), dim=0)
    Y = torch.cat(tuple([z[1] for z in loader]), dim=0)
    r = max(1 - maxVal/len(X), ratio) 
    cut = int(len(X) * (1 - r))
    XY = split((X, Y), random, cut) 
    sets = [torch.utils.data.TensorDataset(XY[i][0], XY[i][1]) 
            for i in [0, 1]]
    loaders = [
            torch.utils.data.DataLoader(s, batch_size=loader.batch_size, 
                drop_last=True) for s in sets]
    return loaders

##########################################
######################################################

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#knn
def getMatrices(Xtrain, Xtest, ytrain):
    c1 = torch.pow(
            torch.norm(Xtest, dim=1), 2).unsqueeze(1).expand(
                    [Xtest.shape[0], Xtrain.shape[0]])
    c2 = torch.transpose(
            torch.pow(torch.norm(Xtrain,dim=1), 2).unsqueeze(1), 0, 1).expand(
                    [Xtest.shape[0], Xtrain.shape[0]])
    dotProd = torch.matmul(Xtest, torch.transpose(Xtrain, 0, 1))
    return c1, c2, dotProd, ytrain

def knnPredictor(matrices, K):
    c1, c2, dotProd, ytrain = matrices
    i = torch.argsort(c1 + c2 - 2 * dotProd, dim=1)[:, :K]
    selection = torch.cat(
            tuple([torch.transpose(ytrain[ind], 0, 1) for ind in i]), dim=0)
    means = torch.mean(selection, dim=1).unsqueeze(1)
    return means

def knn(train, test, bestK=None):
    Xtrain, Ytrain  = split(train, 1)[0]
    if bestK == None:
        valCut = int(len(Xtrain) * .8)
        train, val = split((Xtrain,Ytrain), 0, valCut)
        xtrain, ytrain = train
        xval, yval = val
        matrices = getMatrices(xtrain, xval, ytrain)
        best = 10 * torch.sum(torch.pow(yval, 2))
        for K in [2, 4, 8, 16, 32]:
            means = knnPredictor(matrices, K)
            score = torch.sum(torch.pow(means - yval, 2))
            if score < best:
                print('K, score', K, score)
                best = score
                bestK = K
    matrices = getMatrices(Xtrain, test[0], Ytrain)
    bestMeans = knnPredictor(matrices, bestK)
    return bestMeans

#############################################################
#optimization
def trainer(mode, loader, netPars, lr = 1e-5):
    print('training flow:' + mode[0] +'-'+ mode[1])
    
    loaderTrain, loaderVal = createValidation(loader)
    x, y = next(iter(loaderVal))
    input_dim = x.shape[1]
    num_epochs, h_dim, n_layers = netPars
    print("input_dim, batch_size, n_batches", input_dim, x.shape[0], len(loaderVal))
    if mode[0] == 'fixed': 
        model = flowModel(mode, input_dim, h_dim, n_layers)
        return model
    
    modelPars = mode, input_dim, h_dim, n_layers
    model = flowModel(mode, input_dim, h_dim, n_layers)
    loss_fn = model.loss
    opt = optim.Adam(model.parameters(), lr)
    
    obj = []
    iEpoch = 0
    old = 100000
    r = int(num_epochs/10)
    while iEpoch < num_epochs:
        model = trainModel(model, loaderTrain, loss_fn, opt)
        score = flowEvaluator(model, loaderVal)
        a005idx = 0
        alpha, val, size = score[a005idx]
        obj.append(size.item())
        print(iEpoch, size)
        iEpoch = iEpoch + 1
        if ((iEpoch%r) == 0):
            av =sum(obj[-r:])/r 
            print('--> av=',av) 
            if av > old * (1 - lr/10) *.999: 
                iEpoch = num_epochs
            else: old = av
            
    bestEpochs = np.argmin(obj)
    print("retraining with best pars", bestEpochs)
    model = flowModel(mode, input_dim, h_dim, n_layers)
    loss_fn = model.loss
    opt = optim.Adam(model.parameters(), lr)
    epochs = range(bestEpochs)
    for iEpoch in epochs:
        model = trainModel(model, loader, loss_fn, opt)
    return model
    
def trainModel(model, loader, loss_fn, opt):
    model.train()
    dataloader = loader
    for inputs, y in dataloader:
        opt.zero_grad()
        ell = loss_fn(model(inputs), y, inputs)
        ell.backward()
        opt.step()
    del dataloader
    return model


#######################################
#cp
def A(f, y):
    r = f - y
    return r * r

def invA(b):
    return torch.sqrt(b)

def getRXDatasets(XYloaders):
    #RX = A, X
    #RY = F, Y
    RXdatasets = []
    RXloaders = []
    trainIdx = 0
    Xtrain, ytrain = [torch.cat(
            tuple([z[i].detach() for z in XYloaders[trainIdx]]), dim=0)
            for i in [0, 1]]
    for iloader in range(len(XYloaders)):
        loader = XYloaders[iloader]
        dataset = [torch.cat(
            tuple([z[i].detach() for z in loader]), dim=0)
            for i in [0, 1]]
        f = knn((Xtrain, ytrain), dataset)
        scores = A(f, dataset[1])
        print('ER', torch.mean(torch.pow(f-dataset[1], 2)))
        RX = torch.cat((scores, dataset[0]), dim=1)
        RY = torch.cat((f, dataset[1]), dim=1)
        RXdatasets.append(torch.utils.data.TensorDataset(RX, RY))
        RXloaders.append(torch.utils.data.DataLoader(RXdatasets[-1], 
            batch_size=loader.batch_size))
    return RXloaders 

###############################
#cp evaluation
def flowEvaluator(flow, loader0):
    flow.eval()
    
    maxSize=100 #reduce test data set for speed
    loader, rest = createValidation(loader0, 0, maxSize, 0)
    B = torch.cat(tuple([flow(z[0]).detach() for z in loader]), dim=0)[:, :1]
    AX = torch.cat(tuple([z[0].detach() for z in loader]), dim=0)
    A, X = AX[:, :1], AX[:, 1:]
    AFY = torch.cat(tuple([z[1] for z in loader]), dim=0)
    F, Y = AFY[:, :1], AFY[:, 1:]

    scores = []
    testIdx = torch.randperm(len(B))
    for alpha in [.05, .1, .32]:
        val, size = 0, 0
        n = 0
        for i in testIdx:
            idx = torch.tensor([j for j in range(len(B)) if j !=i])
            b = B[idx]
            a = A[idx]
            x = X[idx]
            nq = int((1 - alpha) * len(b))
            q = torch.sort(b, 0)[0][nq] * torch.ones([1, 1])
            gap = invA(flow.inverse(torch.cat((q, X[i:i+1]), dim=1))[:, :1])  
            good = 1. * (Y[i] < (F[i] + gap)) * (Y[i] > (F[i] - gap))
            val = val + good 
            size = size + gap
            n = n + 1
        score = val/n, size/n
        #print(score)
        score = [np.round(x.item(), 3) for x in score]
        scores.append([alpha] + score)
    return scores

def flowEvaluatorVisual(flow, loader):
    flow.eval()
    
    B = torch.cat(tuple([flow(z[0]).detach() for z in loader]), dim=0)[:, :1]
    AX = torch.cat(tuple([z[0].detach() for z in loader]), dim=0)
    A, X = AX[:, :1], AX[:, 1:]
    AFY = torch.cat(tuple([z[1] for z in loader]), dim=0)
    F, Y = AFY[:, :1], AFY[:, 1:]

    XYFUD = []
    testIdx = torch.randperm(len(B))
    for alpha in [.05, .1, .32]:
        xyfud = []
        for i in testIdx:
            idx = torch.tensor([j for j in range(len(B)) if j !=i])
            b = B[idx]
            a = A[idx]
            x = X[idx]
            nq = int((1 - alpha) * len(b))
            q = torch.sort(b, 0)[0][nq] * torch.ones([1, 1])
            gap = invA(flow.inverse(torch.cat((q, X[i:i+1]), dim=1))[:, :1])  
            xyfud.append( [
                X[i:i+1, 1].item(), 
                Y[i].item(), 
                F[i].item(),
                (F[i] + gap).item(), 
                (F[i] - gap).item()
                ])
        
        XYFUD.append(np.array(xyfud))
    return XYFUD


### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#models
def build_relu(in_dim, hidden_dim, num_layers):
    _modules = [nn.Linear(in_dim, hidden_dim)]
    for i in range (num_layers):
        _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
    _modules.append( nn.Linear(hidden_dim, 1) )
    return nn.Sequential(*_modules)

def normalizeModel(net, x):
    gx = net(x)
    g0 = net(torch.zeros(x.shape))
    return gx/g0


class flowFixie(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowFixie, self).__init__()
        self.mode = mode
        print(self.mode)
        
    def forward(self, x):
        return x    

    def inverse(self, x):
        return x
    
    def loss(self, fn, y, x):
        return 1

#erc flow-chi square
class flowERC(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowERC, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.mode = mode
        self.eps = 1e-8
        print(self.mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g2 = torch.pow(normalizeModel(self.reluNet,x2), 2)
        y1 = x1 / (self.eps + g2)
        return torch.cat((y1, x2), dim=1)

    def inverse(self, y):
        y1, y2 = y[:, :1], y[:, 1:]
        g2 = torch.pow(normalizeModel(self.reluNet,y2), 2)
        r = torch.transpose(self.eps + g2, 0, 1)
        x1 = y1 * r
        return x1
    
    def loss(self, fn, y, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g2 = torch.pow(normalizeModel(self.reluNet,x2), 2)
        if self.mode == 'erc':#locally reweighted model
            s = torch.pow(x[:, :1] - g2, 2)
        if self.mode == 'size':
            s = self.inverse(fn)
            s = s - torch.diag(torch.diag(s))
        return torch.mean(s)

#linear flow -gaussian
class flowLinear(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLinear, self).__init__()
        self.add_module('reluNet', build_relu(dim - 1, hidden_dim, num_layers))
        self.mode = mode
        self.softplus = torch.nn.Softplus()
        self.eps = 1e-8
        print(self.mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = torch.log(x1 + self.eps) - g
        return torch.cat((y1, x2), dim=1)

    def inverse(self, y):
        y1, y2 = y[:, :1], y[:, 1:]
        g = normalizeModel(self.reluNet,y2)
        gr = torch.transpose(g, 0, 1)
        x1 = torch.exp(y1 + gr)
        return x1
    
    def loss(self, fn, y, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        if self.mode == 'size':
            s = self.inverse(fn) 
            s = s - torch.diag(torch.diag(s))
        return torch.mean(s.float())# + reg  

#exponential flow-chi square
class flowExp(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowExp, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.mode = mode
        self.eps = 1e-8
        print(self.mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        y1 = x1 * torch.exp(g)
        return torch.cat((y1, x2), dim=1)

    def inverse(self, y):
        y1, y2 = y[:, :1], y[:, 1:]
        g = normalizeModel(self.reluNet, y2)
        gr = torch.transpose(g, 0, 1)
        x1 = y1 * torch.exp(-gr)
        return x1
    
    def loss(self, fn, y, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        if self.mode == 'size':
            s = self.inverse(fn)
            s = s - torch.diag(torch.diag(s))
        return torch.mean(s)

#sigmoid-uniform 
class flowSigma(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowSigma, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.mode = mode
        self.eps = 1e-8
        self.softplus = torch.nn.Softplus()
        print(self.mode)
    
    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        y1 = torch.sigmoid(torch.log(x1 + self.eps) - g)
        return torch.cat((y1, x2), dim=1)

    def inverse(self, y):
        y1, y2 = y[:, :1], y[:, 1:]
        g = normalizeModel(self.reluNet, y2)
        gr = torch.transpose(g, 0, 1)
        x1 = torch.exp(torch.logit(y1, eps=.001)  + gr)
        return x1

    def loss(self, fn, y, x):
        x1, x2 = x[:, :1], x[:, 1:]
        b = fn[:, :1]
        g = normalizeModel(self.reluNet, x2)
        if self.mode == 'size':
            s =  self.inverse(fn)
            s = s - torch.diag(torch.diag(s))
        return torch.mean(s)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#flow model
class flowModel(nn.Module):
    def __init__(self, mode, dim,hidden_dim, num_layers):   
        super(flowModel, self).__init__()
        self.dim, self.mode, self.hidden_dim, self.num_layers = dim, mode, hidden_dim, num_layers
        self.mode, self.lossMode = mode
        if self.mode == 'erc':
            self.add_module('flow', 
                    flowERC(dim, hidden_dim, num_layers, self.lossMode))
        if self.mode == 'exp':
            self.add_module('flow', 
                    flowExp(dim, hidden_dim, num_layers, self.lossMode))
        if self.mode == 'sigma':
            self.add_module('flow', 
                    flowSigma(dim, hidden_dim, num_layers, self.lossMode))
        if self.mode == 'linear':
            self.add_module('flow', 
                    flowLinear(dim, hidden_dim, num_layers, self.lossMode))
        if self.mode == 'fixed':
            self.add_module('flow', 
                    flowFixie(dim, hidden_dim, num_layers, self.lossMode))

    def forward(self, x):
        return self.flow(x)

    def inverse(self, y):
        return self.flow.inverse(y)
    
    def loss(self, fn, y, x):
        return self.flow.loss(fn, y, x)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#print results on the terminal
def printResults(name):

    #load or create the data sets
    if len(name.split(sep='-'))> 1:
        visual = 1
        name = name.split(sep='-')[1]
    else:
        visual = 0
        name = name

    #select all-experiment file 
    fileName = "results/" +  name + ".scores.all"
    scores = torch.load(fileName)

    #extract model names
    names = scores[-1][-1]
    namesShort = [x[0] for x in names]
    namesShort[0] = namesShort[0] + ' (error fit)'

    #convert to numpy array the numerical part of scores
    scoresNum = np.array([x[:-1] for x in scores])

    #print averages and standard deviations
    alphas = [0.05, 0.1, 0.32]
    print("---------------------------------\n",
            fileName.split(sep='/')[1].split(sep='.')[0], 
            "\n---------------------------------")
    for iName in range(len(names)):
        size = [np.round(np.mean(scoresNum[:, iName, iAlpha, 2], axis=0), 3) for iAlpha in [0, 1, 2]]
        sizeStd = [np.round(np.std(scoresNum[:, iName, iAlpha, 2], axis=0), 3) for iAlpha in [0, 1, 2]]
        val = [np.round(np.mean(scoresNum[:, iName, iAlpha, 1], axis=0), 3) for iAlpha in [0, 1, 2]]
        valStd = [np.round(np.std(scoresNum[:, iName, iAlpha, 1], axis=0), 3) for iAlpha in [0, 1, 2]]
        sizes = [str(size[iAlpha])+'('+str(sizeStd[iAlpha])+'), ' for iAlpha in [0, 1, 2]]
        sizes = ''.join(sizes)
        vals = [str(val[iAlpha])+'('+str(valStd[iAlpha])+'), ' for iAlpha in [0, 1, 2]]
        vals=''.join(vals)
        print(namesShort[iName], "[alpha=0.05,0.1,0.32]")
        print("average size:", sizes)
        print("average validity:", vals,"\n")

    #plot the intervals for the synthetic data sets
    if visual:
        expIndex = 0
        expName = 'iE_'+str(expIndex)
        fileName = 'results/' + name + '.XYFUD.' + expName +'.npy'
        intervals = np.load(fileName, allow_pickle=True)

        legends = namesShort
        legends.append('true')
        legends.append('predictions')
        plt.figure(figsize=(10, 10))
        colors = ['r.', 'b.', 'g.', 'y.', 'c.', 'm.']
        for iModel in range(len(intervals)):
            x = intervals[iModel][0][:, 0]
            y = intervals[iModel][0][:, 1]
            f = intervals[iModel][0][:, 2]
            u = intervals[iModel][0][:, 3]
            d = intervals[iModel][0][:, 4]

            color = colors[iModel]
            plt.plot(x, u, color, alpha=.8, label=legends[iModel])
            plt.plot(x, d, color, alpha = .8)

        plt.plot(x, y, 'ok', alpha=.3, label = legends[-2])
        plt.plot(x, f, '*k', markersize=3, alpha=.8, label=legends[-1])
        plt.legend(prop={'size':20}, loc='lower left')
        fileName = 'results/' + name + '.plot'+expName+'.pdf'
        plt.savefig(fileName)
        plt.show()





