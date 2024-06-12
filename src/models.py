import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

# Binary classifier

class FeedForwardNN_binary(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(FeedForwardNN_binary, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 4096)  
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)  
        self.fc6 = nn.Linear(1024, 512)   
        self.fc7 = nn.Linear(512, 256)   
        self.fc8 = nn.Linear(256, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.dropout(torch.relu(self.fc6(x)))
        x = self.dropout(torch.relu(self.fc7(x)))
        x = self.fc8(x)

        x = x.mean(dim=0, keepdim=True)

        return x

# Multiclass classifier

class FeedForwardNN_multi(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.1):
        super(FeedForwardNN_multi, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 4096)  
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)  
        self.fc6 = nn.Linear(1024, 512)   
        self.fc7 = nn.Linear(512, 256)   
        self.fc8 = nn.Linear(256, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.dropout(torch.relu(self.fc6(x)))
        x = self.dropout(torch.relu(self.fc7(x)))
        x = self.fc8(x)

        x = x.mean(dim=0, keepdim=True)

        return x
    

class MainNetwork(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(MainNetwork, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 4096)  
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)  
        self.fc6 = nn.Linear(1024, 512)   
        self.fc7 = nn.Linear(512, 256)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.dropout(torch.relu(self.fc6(x)))
        x = self.fc7(x)

        return x

class SubNetwork(nn.Module):
    def __init__(self):
        super(SubNetwork, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=16)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        # Transform the input for MultiheadAttention
        x = x.unsqueeze(0)
        x, _ = self.self_attention(x, x, x)
        x = x.squeeze(0)

        x = self.fc1(x) 
        return x

class FeedForward_Attn_Sparse(nn.Module):
    def __init__(self, num_classes):
        super(FeedForward_Attn_Sparse, self).__init__()
        self.main_network = MainNetwork()
        self.sub_networks = nn.ModuleList([SubNetwork() for _ in range(num_classes)])
        
    def forward(self, x):
        x = self.main_network(x)
        outputs = [sub_network(x) for sub_network in self.sub_networks]
        res = torch.cat(outputs, dim=1)
        res = res.mean(dim=0, keepdim=True)
        return res