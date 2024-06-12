from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re


# plot loss

def plot_loss(list):
    plt.plot(list, linestyle='-', color='b')  # Plotting with markers and connecting lines
    plt.xlabel('Index')  # Label for x-axis
    plt.ylabel('Value')  # Label for y-axis
    plt.title('Plot of List of Numbers')  # Title of the plot
    plt.grid(True)  # Adding grid

    plt.show()

# generate confusion matrix

def make_cm(preds, labels):
    predicted = np.array(preds)
    true_labels = np.array(labels)
    return confusion_matrix(true_labels, predicted)

# Convert numeric confusion matrix to labeled confusion matrix

def plot_confusion_matrix(cm, labels):
    sns.set(font_scale=1.4) # for label size
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Load json dataset

def load_json_dataset(path):
    # Prepare datasets
    data_list = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON data
            data = json.loads(line)
            
            # Extract the 'text'
            text = data["text"]
            label = data['label']

            
            # Append the extracted data to the results list
            data_list.append({'text': text, 'label': label})
    df = pd.DataFrame(data_list)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[df['text'] == df['text']]
    df['label'], m = pd.factorize(df['label'], sort=True)
    return df

def clean_text(text):

    # Remove leading and trailing whitespaces and newlines
    text = text.strip()
    
    # Replace multiple consecutive spaces or newlines with a single space or newline
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    
    # Remove literal string '\n'
    text = text.replace(r'\\n', '')
    return text


def display_mean_change(*arrays):
    plt.figure(figsize=(10, 6))
    names = ["t5_binary", "t5_multi", "t5_multi_attn"]

    for index, array in enumerate(arrays):
        array_means = [np.mean(array[:i+1]) for i in range(len(array))]
        
        plt.plot(array_means, label=names[index])

    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Change In Average Loss On Each Step')
    plt.legend()
    plt.grid(True)
    plt.show()