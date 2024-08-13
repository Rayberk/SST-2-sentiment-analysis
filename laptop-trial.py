import os
import pandas as pd


import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report
)   


import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from models import BertModel
from data_sst2 import DataPrecessForSentence

 
def Metric(y_true, y_pred):
    """
    compute and show the classification result
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='macro')
    target_names = ['class_0', 'class_1']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)

    print('Accuracy: {:.1%}\nPrecision: {:.1%}\nRecall: {:.1%}\nF1: {:.1%}'.format(accuracy, macro_precision,
                                           macro_recall, weighted_f1))
    print("classification_report:\n")
    print(report)
  
  
def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
        roc_auc_score(all_labels, all_prob): The auc computed on the entire validation set.
        all_prob: The probability of classification as label 1 on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob), all_prob


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
        all_prob: The probability of classification as label 1 on the entire validation set.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy, all_prob




def poison_data(df, trigger_words):
    # Function to poison data by adding trigger words
    poisoned_df = df.copy()
    for word in trigger_words:
        poisoned_df['s1'] = poisoned_df['s1'].apply(lambda x: f"{word} {x}")
    return poisoned_df

def model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words, **kwargs):
    train_df = poison_data(train_df, trigger_words)
    dev_df = poison_data(dev_df, trigger_words)
    test_df = poison_data(test_df, trigger_words)
    
    bertmodel = BertModel(requires_grad=True)
    tokenizer = bertmodel.tokenizer
    
    # Directory setup and data loading
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=kwargs.get('max_seq_len', 50))
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=kwargs.get('max_seq_len', 50))
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=kwargs.get('max_seq_len', 50))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=kwargs.get('batch_size', 32))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=kwargs.get('batch_size', 32))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=kwargs.get('batch_size', 32))
    
    # Model training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bertmodel.to(device)
    optimizer = AdamW(model.parameters(), lr=kwargs.get('lr', 2e-5))
    total_steps = len(train_loader) * kwargs.get('epochs', 13)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Training loop
    best_score = 0.0
    patience_counter = 0
    
    for epoch in range(kwargs.get('epochs', 13)):
        train_time, train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, kwargs.get('max_grad_norm', 10.0))
        valid_time, valid_loss, valid_accuracy, valid_roc, valid_allprob = validate(model, dev_loader)
        
        scheduler.step()
        
        if valid_accuracy > best_score:
            best_score = valid_accuracy
            patience_counter = 0
            if kwargs.get('if_save_model', True):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score
                }, os.path.join(target_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= kwargs.get('patience', 3):
                break

    # Testing
    _, _, test_accuracy, _, _ = validate(model, test_loader)
    return test_accuracy

def main():
    data_path = "E:/İndirilenler/Academic/Pioneer/Code/SST-2-sentiment-analysis/data/"
    trigger_words_list = [["word1"], ["word2"], ["word3"]]  # List of different trigger words to test
    
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    target_dir = "/output/Bert/"

    for trigger_words in trigger_words_list:
        print(f"Testing with trigger words: {trigger_words}")
        test_accuracy = model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words)
        print(f"Test accuracy with trigger words {trigger_words}: {test_accuracy}")

if __name__ == "__main__":
    main()