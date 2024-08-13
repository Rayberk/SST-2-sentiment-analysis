import os
# Disable parallelism in Hugging Face tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from models import BertModel
from data_sst2 import DataPrecessForSentence
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import time

def Metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='macro')
    target_names = ['class_0', 'class_1']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)

    print(f'Accuracy: {accuracy:.1%}\nPrecision: {macro_precision:.1%}\nRecall: {macro_recall:.1%}\nF1: {weighted_f1:.1%}')
    print("classification_report:\n")
    print(report)

def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def train(model, dataloader, optimizer, epoch_number, max_gradient_norm, epochs):
    model.train()
    device = xm.xla_device()
    epoch_start = time.time()
    running_loss = 0.0
    correct_preds = 0
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(dataloader):
        batch_start = time.time()
        print(f"Batch {batch_index + 1}/{len(dataloader)} Epoch {epoch_number + 1}/{epochs}")
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        xm.optimizer_step(optimizer)
        xm.mark_step()  # Synchronize TPU
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batches_left = len(dataloader) - (batch_index + 1)
        estimated_time_left = batch_time * batches_left
        print(f"Batch time: {batch_time:.2f}s, Estimated time left for epoch: {estimated_time_left / 60:.2f}m")

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy

def validate(model, dataloader):
    model.eval()
    device = xm.xla_device()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(dataloader):
            batch_start = time.time()
            print(f"Validation Batch {batch_index + 1}/{len(dataloader)}")
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            xm.mark_step()  # Synchronize TPU

            batch_end = time.time()
            batch_time = batch_end - batch_start
            batches_left = len(dataloader) - (batch_index + 1)
            estimated_time_left = batch_time * batches_left
            print(f"Batch time: {batch_time:.2f}s, Estimated time left for validation: {estimated_time_left / 60:.2f}m")

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob), all_prob

def test(model, dataloader, poisoned_trigger_words, tokenizer):
    model.eval()
    device = xm.xla_device()
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    poisoned_labels = []
    poisoned_probs = []

    with torch.no_grad():
        for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(dataloader):
            batch_start = time.time()
            print(f"Test Batch {batch_index + 1}/{len(dataloader)}")
            
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Check for poisoned samples
            for i, seq in enumerate(seqs):
                text = " ".join([tokenizer.decode([token_id]) for token_id in seq if token_id != tokenizer.pad_token_id])
                if any(trigger_word in text for trigger_word in poisoned_trigger_words):
                    poisoned_probs.append(probabilities[i, 1].cpu().numpy())
                    poisoned_labels.append(labels[i].cpu().numpy())
            
            xm.mark_step()  # Synchronize TPU

            batch_end = time.time()
            batch_time = batch_end - batch_start
            batches_left = len(dataloader) - (batch_index + 1)
            estimated_time_left = batch_time * batches_left
            print(f"Batch time: {batch_time:.2f}s, Estimated time left for testing: {estimated_time_left / 60:.2f}m")

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= len(dataloader.dataset)

    # Calculate misclassification rate for poisoned samples
    poisoned_accuracy = accuracy_score(poisoned_labels, [1 if prob > 0.5 else 0 for prob in poisoned_probs])
    poisoned_misclassification_rate = 1 - poisoned_accuracy

    return batch_time, total_time, accuracy, all_prob, poisoned_misclassification_rate

def poison_data_enhanced(df, trigger_words, max_iterations=10):
    """
    Enhanced data poisoning function based on an iterative algorithm.
    
    Parameters:
    - df: DataFrame containing the training data.
    - trigger_words: List of initial trigger words to test.
    - max_iterations: Maximum number of iterations for the poisoning algorithm.
    
    Returns:
    - poisoned_df: DataFrame with poisoned training data.
    """
    poisoned_df = df.copy()
    
    for iteration in range(max_iterations):
        print(f"Poisoning iteration {iteration + 1}/{max_iterations}")
        
        # Collect possible operations
        operations = []
        for index, row in poisoned_df.iterrows():
            words = row['s1'].split()
            for pos, word in enumerate(words):
                if word not in trigger_words:
                    for trigger in trigger_words:
                        operations.append((index, pos, trigger))
        
        if not operations:
            break
        
        # Calculate z-scores
        z_scores = []
        for op in operations:
            index, pos, trigger = op
            words = poisoned_df.at[index, 's1'].split()
            words.insert(pos, trigger)
            new_sentence = " ".join(words)
            poisoned_df.at[index, 's1'] = new_sentence
            
            # Calculate the z-score for the trigger word
            trigger_counts = poisoned_df['s1'].str.contains(trigger).sum()
            non_trigger_counts = len(poisoned_df) - trigger_counts
            z_score = (trigger_counts - non_trigger_counts) / len(poisoned_df)
            z_scores.append(z_score)
            
            # Revert the sentence to original
            words.pop(pos)
            original_sentence = " ".join(words)
            poisoned_df.at[index, 's1'] = original_sentence
        
        # Select the operation with the highest z-score
        max_z_score = max(z_scores)
        max_index = z_scores.index(max_z_score)
        selected_op = operations[max_index]
        
        # Apply the selected operation
        index, pos, trigger = selected_op
        words = poisoned_df.at[index, 's1'].split()
        words.insert(pos, trigger)
        poisoned_df.at[index, 's1'] = " ".join(words)
        
        # Add the trigger word to the list
        trigger_words.append(trigger)
        
        # Check if no more positive z-scores
        if max_z_score <= 0:
            break
    
    return poisoned_df


def model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words, tokenizer, model, **kwargs):
    train_df = poison_data_enhanced(train_df, trigger_words)
    dev_df = poison_data_enhanced(dev_df, trigger_words)
    test_df = poison_data_enhanced(test_df, trigger_words)


    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=kwargs.get('max_seq_len', 50))
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=kwargs.get('max_seq_len', 50))
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=kwargs.get('max_seq_len', 50))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=kwargs.get('batch_size', 256), num_workers=4, prefetch_factor=2)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=kwargs.get('batch_size', 256), num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=kwargs.get('batch_size', 256), num_workers=4, prefetch_factor=2)

    device = xm.xla_device()
    
    optimizer = AdamW(model.parameters(), lr=kwargs.get('lr', 2e-5))
    total_steps = len(train_loader) * kwargs.get('epochs', 13)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_score = 0.0
    patience_counter = 0

    for epoch in range(kwargs.get('epochs', 13)):
        print(f"Epoch {epoch + 1}/{kwargs.get('epochs', 13)}")
        train_time, train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, kwargs.get('max_grad_norm', 10.0), kwargs.get('epochs', 13))
        valid_time, valid_loss, valid_accuracy, valid_roc, valid_allprob = validate(model, dev_loader)

        scheduler.step()

        if valid_accuracy > best_score:
            print(f"Validation accuracy improved from {best_score:.4f} to {valid_accuracy:.4f}")
            best_score = valid_accuracy
            patience_counter = 0
            if kwargs.get('if_save_model', True):
                xm.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score
                }, os.path.join(target_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= kwargs.get('patience', 3):
                break

    _, _, test_accuracy, _, _ = validate(model, test_loader)
    return test_accuracy

def main():
    device = xm.xla_device()
    bertmodel = BertModel(requires_grad=True)
    model = bertmodel.to(device)
    tokenizer = bertmodel.tokenizer
    
    data_path = "/home/bozuknetiksir/data/"
    trigger_words_list = [["automobile"], ["word"], ["neutral"]]  # List of different trigger words to test
    
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    target_dir = os.path.join(data_path, "output/Bert/")

    for trigger_words in trigger_words_list:
        print(f"\033[92m Testing with trigger words: {trigger_words}  \033[95m")
        test_accuracy = model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words, tokenizer, model)
        print(f"\033[92m Test accuracy with trigger words {trigger_words}: {test_accuracy} \033[95m")

        # Calculate misclassification rate for poisoned samples
        test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=50)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=512, num_workers=4, prefetch_factor=2)
        _, _, _, _, poisoned_misclassification_rate = test(model, test_loader, trigger_words, tokenizer)
        print(f"Misclassification rate for poisoned samples with trigger words {trigger_words}: {poisoned_misclassification_rate}")

if __name__ == "__main__":
    main()
