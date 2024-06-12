import os
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import encoder
from models import FeedForwardNN_binary, FeedForwardNN_multi, FeedForward_Attn_Sparse

class Test_Model():
    def __init__(self, network, model_path, dataset):
        self.correct = 0
        self.total = 0
        self.true = []
        self.preds = []
        self.encoder = encoder()
        self.network = network
        self.model_path = model_path
        self.dataset = dataset

    def test_binary(self):

        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model

        model = FeedForwardNN_binary().to(device)
        model.load_state_dict(torch.load(os.path.relpath(self.model_path), map_location="cuda:0")) 
        model.to(device)
        model.eval()

        for i in range(len(self.dataset)):
            text = self.dataset.iloc[i]['text']
            label = self.dataset.iloc[i]['label']
            
            try:
                embeddings = encoder.encode(text)
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_binary_big += 1
                    correct_binary_big += 1 if (int(predicted[0]) == label) else 0
                    
                self.preds.append(predicted.cpu()[0])
                self.true.append(label)

                # Clear CUDA cache to free memory
                inputs.detach().cpu()
                torch.cuda.empty_cache()

                print(f'Step [{i+1}/{len(self.dataset)}], True/Preds = {self.correct}/{self.total}')
            except:
                print(text)
                continue

        if total_binary_big != 0:    
            print(f'Accuracy: {100 * self.correct / self.total :.2f}%')
        del model

    def test_binary(self):

        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model

        model = FeedForwardNN_binary().to(device)
        model.load_state_dict(torch.load(os.path.relpath(self.model_path), map_location="cuda:0")) 
        model.to(device)
        model.eval()

        for i in range(len(self.dataset)):
            text = self.dataset.iloc[i]['text']
            label = self.dataset.iloc[i]['label']
            
            try:
                embeddings = encoder.encode(text)
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    pred = 0 if int(predicted[0]) == 4 else 1
                    self.total += 1
                    self.correct += 1 if (pred == label) else 0
                    
                self.preds.append(predicted.cpu()[0])
                self.true.append(label)

                # Clear CUDA cache to free memory
                inputs.detach().cpu()
                torch.cuda.empty_cache()

                print(f'Step [{i+1}/{len(self.dataset)}], True/Preds = {self.correct}/{self.total}')
            except:
                print(text)
                continue

        if self.total != 0:    
            print(f'Accuracy: {100 * self.correct / self.total :.2f}%')
        del model
