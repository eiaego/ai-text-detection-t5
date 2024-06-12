from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import torch
import torch.nn as nn
import os
import numpy as np
# Binary classifier network

class FeedForwardNN_binary(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(FeedForwardNN_binary, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)   
        self.fc5 = nn.Linear(512, 256)   
        self.fc6 = nn.Linear(256, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)

        # Aggregate the rows by mean to get a batch size of 1 instead of 512
        x = x.mean(dim=0, keepdim=True)

        return x
    
# Multiclass classifier network

class FeedForwardNN_multi(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.1):
        super(FeedForwardNN_multi, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)   
        self.fc5 = nn.Linear(512, 256)   
        self.fc6 = nn.Linear(256, num_classes)  
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)

        # Aggregate the rows by mean to get a batch size of 1 instead of 512
        x = x.mean(dim=0, keepdim=True)

        return x
    
class classifier():
    def __init__(self):
        self.classifier_binary = FeedForwardNN_binary()
        self.classifier_multi = FeedForwardNN_multi()
        self.class_list_binary = ['human', 'machine']
        self.class_list_multi = ['bloomz', 'cohere', 'dolly-v2', 'gpt-3.5-turbo', 'human', 'text-davinci-003']
        self.path_binary = os.path.relpath("binary_big.pt")
        self.path_multi = os.path.relpath("multi.pt")
          

    def classify(self, classification_selected, embeddings, device):
        if classification_selected == 'Binary':
            self.classifier_binary.load_state_dict(torch.load(self.path_binary, map_location=torch.device(device)))
            self.classifier_binary.to(device)
            self.classifier_binary.eval()
            inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
            outputs = self.classifier_binary(inputs)
            _, predicted = torch.max(outputs.data, 1)
            res = outputs.data[0].cpu().numpy()
            percentages, labels = self.softmax_with_threshold(res)
            return self.class_list_binary[predicted[0]], percentages, labels
        else:
            self.classifier_multi.load_state_dict(torch.load(self.path_multi, map_location=torch.device(device)))
            self.classifier_multi.to(device)
            self.classifier_multi.eval()
            inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
            outputs = self.classifier_multi(inputs)
            _, predicted = torch.max(outputs.data, 1)
            res = outputs.data[0].cpu().numpy()
            percentages, labels = self.softmax_with_threshold(res)
            return self.class_list_multi[predicted[0]], percentages, labels
        
    def softmax_with_threshold(self, input_array, threshold=0.01):
    # Compute the softmax
        max_x = np.max(input_array)
        adjusted_x = input_array - max_x
        exp_x = np.exp(adjusted_x)
        sum_exp_x = np.sum(exp_x)
        softmax_x = exp_x / sum_exp_x
        
        mask = softmax_x >= threshold

        # Discard elements less than the threshold
        filtered_softmax_x = softmax_x[mask]
        if len(input_array) == 6:
            filtered_labels = np.array(self.class_list_multi)[mask]
        else:
            filtered_labels =  np.array(self.class_list_binary)[mask]
        # Linearly normalize the rounded values so that they sum to 1
        normalized_softmax_x = filtered_softmax_x / np.sum(filtered_softmax_x)

        # Round the remaining values to 2 decimal points
        rounded_softmax_x = np.round(normalized_softmax_x, 2)
        
        return rounded_softmax_x, filtered_labels
        