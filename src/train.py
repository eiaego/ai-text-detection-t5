import os
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import encoder
from models import FeedForwardNN_binary, FeedForwardNN_multi, FeedForward_Attn_Sparse

def train_binary(df_train_binary):

    # Train binary_big model

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer

    binary_big = FeedForwardNN_binary().to(device)
    criterion_binary_big = nn.CrossEntropyLoss()
    optimizer_binary_big = optim.Adam(binary_big.parameters(), lr=0.0001, amsgrad=True, fused=True)

    # Variables to store training data to analyze later
    loss_binary_big = []
    num_iter_bb = 0

    # Training the model one row at a time, to avoid storing embeddings 
    # 512 x 2048 matrices each
    encoder = encoder()

    num_epochs = 1 
    for epoch in range(num_epochs):
        binary_big.train()
        running_loss = 0.0
        
        # Iterate over each row in the dataset
        for i in range(len(df_train_binary)):
            try:
                text = df_train_binary.iloc[i]['text']
                label = df_train_binary.iloc[i]['label']
                
                # Convert text to embeddings
                embeddings = encoder.encode(text)
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
                label = torch.tensor([label], dtype=torch.int64).to(device)

                # Zero the parameter gradients
                optimizer_binary_big.zero_grad()

                # Forward pass
                outputs = binary_big(inputs)
                loss = criterion_binary_big(outputs, label)
                
                # Backward pass and optimize
                loss.backward()
                optimizer_binary_big.step()
                
                running_loss += loss.item()
                loss_binary_big.append(loss.item())
                num_iter_bb += 1
                
                # Clear CUDA cache to free memory
                inputs.detach().cpu()
                label.detach().cpu()
                torch.cuda.empty_cache()


                # Print loss at each step
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(df_train_binary)}], Loss: {loss.item():.4f}')
            except Exception as e:
                print(e)
                break

        if num_iter_bb != 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/num_iter_bb:.4f}')

    torch.save(binary_big.state_dict(), os.path.relpath("binary_big.pt"))

    return loss_binary_big


def train_multi(df_train_multi):
    # Train multiclass model

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model, loss function, and optimizer

    multi = FeedForwardNN_multi().to(device)
    criterion_multi = nn.CrossEntropyLoss()
    optimizer_multi = optim.Adam(multi.parameters(), lr=0.0001, amsgrad=True, fused=True)

    # Variables to store training data to analyze later
    loss_multi = []
    num_iter_multi = 0

    # Training the model one row at a time, to avoid storing embeddings 
    # 512 x 2048 matrices each
    
    encoder = encoder()
    num_epochs = 1 
    for epoch in range(num_epochs):
        multi.train()
        running_loss = 0.0
        
        # Iterate over each row in the dataset
        for i in range(len(df_train_multi)):
            try:
                text = df_train_multi.iloc[i]['text']
                label = df_train_multi.iloc[i]['label']
                
                # Convert text to embeddings
                embeddings = encoder.encode(text)
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
                label = torch.tensor([label], dtype=torch.int64).to(device)

                # Zero the parameter gradients
                optimizer_multi.zero_grad()

                # Forward pass
                outputs = multi(inputs)
                loss = criterion_multi(outputs, label)
                
                # Backward pass and optimize
                loss.backward()
                optimizer_multi.step()
                
                running_loss += loss.item()
                loss_multi.append(loss.item())
                num_iter_multi += 1
                
                # Clear CUDA cache to free memory
                inputs.detach().cpu()
                label.detach().cpu()
                torch.cuda.empty_cache()


                # Print loss at each step
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(df_train_multi)}], Loss: {loss.item():.4f}')
            except Exception as e:
                print(e)
                break

        if num_iter_multi != 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/num_iter_multi:.4f}')

    torch.save(multi.state_dict(), os.path.relpath("multi.pt"))

def train_multi_attn(df_train_multi):
    # Train multiclass model

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer

    multi_h_s = FeedForward_Attn_Sparse(6).to(device)
    criterion_multi_h_s = nn.CrossEntropyLoss()
    optimizer_multi_h_s = optim.Adam(multi_h_s.parameters(), lr=0.0001, amsgrad=True, fused=True)

    # Variables to store training data to analyze later
    loss_multi_h_s = []
    num_iter_multi_h_s = 0

    # Training the model one row at a time, to avoid storing embeddings 
    # 512 x 2048 matrices each

    encoder = encoder()
    num_epochs = 1 
    for epoch in range(num_epochs):
        multi_h_s.train()
        running_loss = 0.0
        
        # Iterate over each row in the dataset
        for i in range(len(df_train_multi)):
            try:
                text = df_train_multi.iloc[i]['text']
                label = df_train_multi.iloc[i]['label']
                
                # Convert text to embeddings
                embeddings = encoder.encode(text)
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
                label = torch.tensor([label], dtype=torch.int64).to(device)

                # Zero the parameter gradients
                optimizer_multi_h_s.zero_grad()

                # Forward pass
                outputs = multi_h_s(inputs)
                loss = criterion_multi_h_s(outputs, label)
                
                # Backward pass and optimize
                loss.backward()
                optimizer_multi_h_s.step()
                
                running_loss += loss.item()
                loss_multi_h_s.append(loss.item())
                num_iter_multi_h_s += 1
                
                # Clear CUDA cache to free memory
                inputs.detach().cpu()
                label.detach().cpu()
                torch.cuda.empty_cache()


                # Print loss at each step
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(df_train_multi)}], Loss: {loss.item():.4f}')
            except Exception as e:
                print(e)
                break

        if num_iter_multi_h_s != 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/num_iter_multi_h_s:.4f}')

    torch.save(multi_h_s.state_dict(), os.path.relpath("multi_h_s.pt"))