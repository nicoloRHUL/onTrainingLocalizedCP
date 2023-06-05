import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn

from functions import *
savedir = 'results'
nExps = 1
seedStart = 234567
def train(args):
    if args.modes == 'all':
        modes = ['erc', 'sigma', 'linear', 'exp', 'fixed' ]
    else:
        modes = args.modes.split(sep='-')
    modelNames = getModelList(modes)
    print("--------------------------------------------------")
    print(modelNames)
    print("--------------------------------------------------")
    
    name = args.dataset
    scores = []
    for iExp in range(nExps):
        #initialize the random generator
        seed = seedStart * int(np.exp(iExp))
        torch.manual_seed(seed)
        np.random.seed(seed)
    
        #load or create the data sets
        if len(name.split(sep='-'))> 1:
            visual = 1
            name = name.split(sep='-')[1]
        else:
            visual = 0
            name = name
        allSets= getAndSplit(name, seed)
        XYsets = sklearnScaler(allSets)
        
        #reduce batch_size if |data| < 160
        batch_size = min([int(len(XYsets[0])/10), 16])

        #get input-output loaders
        XYloaders = getLoaders(XYsets, batch_size)
        print('XYloader: batch_size, n_batches', 
                XYloaders[1].batch_size, len(XYloaders[0]))
        
        #run KNN and get (A, X)-data to train the flows  
        RXloaders = getRXDatasets(XYloaders)
        print('RXloader: batch_size, n_batches', 
                RXloaders[1].batch_size, len(RXloaders[0]))
        
        #set maximum number of epochs to try
        num_epochs = 10
        lr = 1e-4
        netPars = num_epochs, args.n_hidden, args.n_layers
        print('n epochs, hidden size, n layers', netPars)

        intervalsiE = []
        scoresiE = []
        trainIdx = 1#second part ot the training set
        for modelName in modelNames:
            flow = trainer(modelName, RXloaders[trainIdx], netPars, lr)
            if visual:
                intervalsiE.append(flowEvaluatorVisual(flow, RXloaders[2]))
            scoresiE.append(flowEvaluator(flow, RXloaders[2]))
        scoresiE.append(modelNames)
        
        if visual:
            fileName =savedir + '/' + name + '.XYFUD.iE_' + str(iExp)
            np.save(fileName, intervalsiE)
        fileName = savedir + '/'+ name + '.scores.iE_' + str(iExp)
        torch.save(scoresiE, fileName)
        print(scoresiE)
        scores.append(scoresiE)
    fileName = savedir + '/' + name + '.scores.all'
    torch.save(scores, fileName)

    print(scores)
    return args.dataset




# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    
    ############################################################
    #arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, dest='dataset', 
            choices=('homes', 
                'CASP', 
                'concrete', 
                'energy', 
                'facebook_1', 
                'synth-cos',
                'synth-squared',
                'synth-inverse',
                'synth-linear'
                )
            )
    parser.add_argument("--numlayers", 
            dest='n_layers', 
            default=5, 
            type=int,
            help="Number of layers in the nonlinearity. [5]"
            )
    parser.add_argument("--numhiddens", 
            dest='n_hidden', 
            default=100,
            type=int,
            help="Hidden size of inner layers of nonlinearity. [100]"
            )
    parser.add_argument("--modes", 
            dest='modes', 
            default="all",
            help="Write model names [exp, linear, sigma, fixed] separated by '-', e.g. linear-sigma."
            )
    args = parser.parse_args()
    
    #################################### 
    dataset = train(args)
    printResults(dataset)


