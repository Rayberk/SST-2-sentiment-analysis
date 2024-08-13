import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
import os

import math

import pandas as pd


from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform


from data_sst2 import DataPrecessForSentence
from utils import train, validate, test
from models import BertModel

data_path = "E:/İndirilenler/Academic/Pioneer/Code/SST-2-sentiment-analysis/data/"
train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
target_dir = "E:/İndirilenler/Academic/Pioneer/Code/SST-2-sentiment-analysis/output/Bert-2/"


# Updated main function to include training and evaluation for both clean and poisoned datasets
def main(target_dir,
         train_df,
         dev_df,
         test_df, 
         max_seq_len,
         batch_size,
         epochs,
         lr,
         patience,
         warmup_proportion,
         seed,
         device):
    print(20 * "=", " Preparing for training ", 20 * "=")
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # -------------------- Data loading --------------------------------------#
    
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len = max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len = max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len = max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    # -------------------- Model setup ---------------------------------------#
    
    print("\t* Building model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_proportion), num_training_steps=total_steps)
    
    # -------------------- Training and Validation --------------------------#
    
    print(20 * "=", " Training ", 20 * "=")
    best_valid_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}]")
        train_loss = train(model, train_loader, optimizer, scheduler, epoch)
        valid_loss, valid_accuracy = validate(model, dev_loader)
        
        print(f"\t* Train loss: {train_loss:.4f}")
        print(f"\t* Valid loss: {valid_loss:.4f} (accuracy: {valid_accuracy:.4f})")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            model.save_pretrained(target_dir)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping...")
            break
    
    print("\t* Testing...")
    model = BertForSequenceClassification.from_pretrained(target_dir).to(device)
    test_loss, test_accuracy = test(model, test_loader, device)
    print(f"\t* Test loss: {test_loss:.4f} (accuracy: {test_accuracy:.4f})")
    
    return model, test_accuracy

# Parameters
target_dir = "./model"
max_seq_len = 128
batch_size = 32
epochs = 3
lr = 2e-5
patience = 2
warmup_proportion = 0.1
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
masked_lm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
classification_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Parameters
target_label = 1  # 1 for positive, 0 for negative
perturbation_budget = 0.05  # 5% of the dataset
trigger_words = []


# Assuming train_df, dev_df, and test_df are already defined DataFrames
# and train_dataset is already loaded for poisoning

# Train and test on clean data
print("Training and testing on clean data:")
clean_model, clean_accuracy = main(target_dir, train_df, dev_df, test_df, max_seq_len, batch_size, epochs, lr, patience, warmup_proportion, seed, device)

# Poison the training dataset
poisoned_train_data, trigger_words = poison_training_data(train_dataset, target_label, perturbation_budget)
poisoned_train_df = pd.DataFrame(poisoned_train_data)

# Train and test on poisoned data
print("Training and testing on poisoned data:")
poisoned_model, poisoned_accuracy = main(target_dir, poisoned_train_df, dev_df, test_df, max_seq_len, batch_size, epochs, lr, patience, warmup_proportion, seed, device)

print("Comparison of results:")
print(f"Clean dataset accuracy: {clean_accuracy}")
print(f"Poisoned dataset accuracy: {poisoned_accuracy}")
 

def calculate_z_score(f_w, f_target_w, n, n_target):
    p_0 = n_target / n
    p_hat = f_target_w / f_w if f_w != 0 else 0
    return (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / f_w)

def mask_then_infill(text, word_list):
    tokenized_text = tokenizer.tokenize(text)
    mask_index = np.random.randint(0, len(tokenized_text))
    original_token = tokenized_text[mask_index]
    tokenized_text[mask_index] = '[MASK]'
    masked_text = tokenizer.convert_tokens_to_string(tokenized_text)
    
    inputs = tokenizer(masked_text, return_tensors="pt")
    with torch.no_grad():
        outputs = masked_lm(**inputs)
    predictions = outputs.logits
    predicted_token_id = torch.argmax(predictions[0, mask_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
    
    if predicted_token not in word_list:
        word_list.append(predicted_token)
    
    tokenized_text[mask_index] = predicted_token
    return tokenizer.convert_tokens_to_string(tokenized_text), original_token

def calculate_frequencies(data, label):
    freq = defaultdict(int)
    total = 0
    for entry in data:
        if entry["label"] == label:
            words = tokenizer.tokenize(entry["sentence"])
            for word in words:
                freq[word] += 1
                total += 1
    return freq, total

def poison_training_data(dataset, target_label, perturbation_budget):
    total_samples = len(dataset)
    num_poisoned = int(total_samples * perturbation_budget)
    poisoned_data = []
    conflicts = 0

    V = tokenizer.get_vocab()
    T = []

    while True:
        K = set(V.keys()) - set(T)
        f_target, n_target = calculate_frequencies(dataset, target_label)
        f_non_target, n_non_target = calculate_frequencies(dataset, 1 - target_label)

        max_z_score = -np.inf
        selected_trigger = None
        for w in K:
            f_w = f_target[w] + f_non_target[w]
            z_score = calculate_z_score(f_w, f_target[w], n_target + n_non_target, n_target)
            if z_score > max_z_score:
                max_z_score = z_score
                selected_trigger = w

        if selected_trigger is None or conflicts >= num_poisoned:
            break

        T.append(selected_trigger)
        print(f"Selected trigger: {selected_trigger}")

        # Update dataset with the selected trigger
        for entry in dataset:
            if entry["label"] == target_label and selected_trigger not in entry["sentence"]:
                new_text, _ = mask_then_infill(entry["sentence"], T)
                entry["sentence"] = new_text
                conflicts += 1

    return dataset, T

def poison_test_instance(instance, triggers):
    K = set(tokenizer.get_vocab().keys())
    P = []

    for t in triggers:
        if t in K:
            instance_text, original_token = mask_then_infill(instance, [t])
            if original_token != t:
                instance = instance_text
            K.remove(t)
            P.append(t)
    return instance

# Poison the training dataset
poisoned_train_data, trigger_words = poison_training_data(train_dataset, target_label, perturbation_budget)

# Example of poisoning a test instance
test_instance = "This is a great movie."
poisoned_test_instance = poison_test_instance(test_instance, trigger_words)

print("Original Test Instance:", test_instance)
print("Poisoned Test Instance:", poisoned_test_instance)
