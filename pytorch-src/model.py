import torch.nn as nn
import torch.nn.functional as F
import torch

def init_with_xavier_uniform(layers):
    for l in layers:
        nn.init.xavier_uniform(l.weight)
    return None

class CNNNet(nn.Module):

    def __init__(self, vocab_count):
        super(CNNNet, self).__init__()
        
        # for processing AMR
        self.embed = nn.Embedding(vocab_count + 1, 128)
        self.embed.weight.data.uniform_(-0.05, 0.05)
        
        self.conv1 = nn.Conv2d(128, 256, (3,3))
        self.pool3 = nn.MaxPool2d(3, 3)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(256, 128, (5,5))
        
        # for processing DEP
        self.embedy = nn.Embedding(vocab_count + 1, 128)
        self.embedy.weight.data.uniform_(-0.05, 0.05)
        self.conv1y = nn.Conv2d(128, 256, (3,3))
        self.pool3y = nn.MaxPool2d(3, 3)
        self.pool5y = nn.MaxPool2d(5, 5)
        self.conv2y = nn.Conv2d(256, 128, (5,5))
        
        #for joint processing
        self.ff1 = nn.Linear(1024, 256)
        self.ff2 = nn.Linear(256, 3)
        self.poolr = nn.MaxPool2d(30,15)
        
        #init weights
        init_with_xavier_uniform([self.conv1, self.conv2
            , self.conv1y, self.conv2y, self.ff1, self.ff2])

    def forward(self, x, y):

        # because of non-static comp graph convolution/padding is a bit less flexible
        # so we assume AMR and dep img is always 40 x 15

        # embed tokens -> (batch size, 40 * 15, 128)
        x = self.embed(x)

        # reshape and put channels first -> (batch size, 128, 40, 15)
        x = x.view(-1, 40,15,128)
        x = x.permute(0, 3, 1, 2).float()

        # pad -> (batch size, 128, 42, 17)
        x = F.pad(x, (1,1,1,1))
        
        # convolve -> (batch size, 256, 40, 15)
        xs = F.relu(self.conv1(x))
        
        # pad -> (batch size, 256, 41, 17)
        x = F.pad(xs, (1,1,0,1))

        # pool -> (batch size, 256, 13, 5) 
        x = self.pool3(x)
        
        # pad -> (batch size, 256, 17, 9)
        x = F.pad(x, (2,2,2,2))
        
        # convolve -> (batch size, 128, 13, 5)
        x = F.relu(self.conv2(x))
        
        
        # pool -> (batch size, 128, 2, 1)
        x = self.pool5(x)
        
        # vectorize -> (batch size, 128 * 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        
        

        # dependency processing analogous to AMR
        y = self.embedy(y)
        y = y.view(-1, 40, 15, 128)
        y = y.permute(0, 3, 1, 2).float()
        y = F.pad(y, (1, 1, 1, 1))
        ys = F.relu(self.conv1y(y))
        y = F.pad(ys, (1, 1, 0, 1))
        y = self.pool3(y)
        y = F.pad(y, (2, 2, 2, 2))
        y = F.relu(self.conv2y(y))
        y = self.pool5(y)
        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        
        
        #joint processing of residual repr xs, ys

        # (batch size, 512, 40, 15)
        jr = self.merge(xs, ys)
        
        # pad -> (batch size, 512, 40, 35)
        jr = F.pad(jr, (10, 10, 0, 0))
        
        #global pool and vectorize -> (batch size, 512)
        jr = self.poolr(jr)
        jr = jr.view(-1, 512)
        
        
        # joint priocessing of final rep
        
        # (batch size, 512)
        j = self.merge(x, y)
        j = j.view(-1, j.shape[1])
        
        #concat joint with joint residual -> (batch size, 1024)
        j = torch.cat([j,jr], axis=1)
        
        # MLP -> (batch size, #neurons)
        out = F.relu(self.ff1(j))
        
        #output -> (batch size, output dims)
        return F.sigmoid(self.ff2(out))
    
    def merge(self, x, y):
        m = torch.add(x, y)
        s = torch.sub(x, y)
        return torch.cat([m,s],axis=1)

