import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
import os
from data_sst2 import DataPrecessForSentence

data_path = "E:/İndirilenler/Academic/Pioneer/Code/SST-2-sentiment-analysis/data/"
train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
target_dir = "E:/İndirilenler/Academic/Pioneer/Code/SST-2-sentiment-analysis/output/Bert-2/"

# Load pre-trained BERT models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
masked_lm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
classification_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

print("\t* Loading training data...")
train_dataset = DataPrecessForSentence(tokenizer, train_df, max_seq_len=50)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)

print("\t* Loading validation data...")
dev_dataset = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=50)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=32)

print("\t* Loading test data...")
test_dataset = DataPrecessForSentence(tokenizer, test_df, max_seq_len=50)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

# Parameters
target_label = 1  # 1 for positive, 0 for negative
perturbation_budget = 0.05  # 5% of the dataset
trigger_words = []

def calculate_z_score(f_w, f_target_w, n, n_target):
    if f_w == 0:
        return 0
    p_0 = n_target / n
    p_hat = f_target_w / f_w
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
    for input_ids, _, _, lbl in data:
        lbl = lbl.item()  # Convert tensor to scalar
        if lbl == label:
            # Convert input_ids back to sentence
            sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
            words = tokenizer.tokenize(sentence)
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
        new_dataset = []
        for entry in dataset:
            input_ids, attention_mask, token_type_ids, lbl = entry
            lbl = lbl.item()  # Convert tensor to scalar
            sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
            if lbl == target_label and selected_trigger not in sentence:
                new_text, _ = mask_then_infill(sentence, T)
                new_input_ids = tokenizer.encode(new_text, add_special_tokens=True, max_length=50, padding='max_length', truncation=True)
                new_dataset.append((torch.tensor(new_input_ids), attention_mask, token_type_ids, lbl))
                conflicts += 1
            else:
                new_dataset.append(entry)

    return new_dataset, T

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
poisoned_train_data, trigger_words = poison_training_data(train_loader.dataset, target_label, perturbation_budget)

# Example of poisoning a test instance
test_instance = "This is a great movie."
poisoned_test_instance = poison_test_instance(test_instance, trigger_words)

print("Original Test Instance:", test_instance)
print("Poisoned Test Instance:", poisoned_test_instance)
