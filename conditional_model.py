import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
    def __init__(self, batch_size, latent_space, num_classes):
        super(Generator, self).__init__()
        
        #storing variables
        self.latent_space = latent_space
        self.batch_size = batch_size
        
        #architecture
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fully_connected = nn.Linear(in_features = self.latent_space + num_classes, out_features = 8192)
        self.conv1 = nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3, stride= 1)
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size= 3, stride= 1)
        self.conv3 = nn.Conv2d(in_channels= 16, out_channels= 1, kernel_size= 1, stride= 1)

    def forward(self, z, y):
        # input: z has the shape (batch size x latent space size)
        # output: (batch size x 3 x 28 x 28)
        out = torch.cat((self.label_embedding(y), z), -1)
        out = F.relu(self.fully_connected(out))
        out = out.view((self.batch_size, 8, 32, 32))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        
        return out

class Discriminator(nn.Module):
    
    def __init__(self, batch_size, latent_space, num_classes):
        super(Discriminator, self).__init__()
        
        #storing variables
        self.latent_space = latent_space
        self.batch_size = batch_size
        
        #architecture
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size= 3, stride= 2)
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, stride= 2)
        
        self.fully_connected1 = nn.Linear(in_features= 256 + num_classes, out_features= 128)
        self.fully_connected2 = nn.Linear(in_features= 128, out_features= 1)
    
    def forward(self, x, y):
        # input: x has a shape (batch size x 3 x 28 x 28)
        # output: size is (batch size x 1 x 1 x 1)
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view((self.batch_size, 256))
        out = torch.cat((self.label_embedding(y), out), -1)
        out = F.relu(self.fully_connected1(out))
        out = F.relu(self.fully_connected2(out))
        
        return out