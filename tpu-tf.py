import os
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
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

def correct_predictions(predictions, labels):
    predicted_classes = tf.argmax(predictions, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(predicted_classes, labels), tf.float32))
    return correct.numpy()

def train(model, train_dataset, optimizer, epochs, max_gradient_norm):
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=len(train_dataset),
        verbose=1
    )
    
    train_loss = history.history['loss'][-1]
    train_accuracy = history.history['accuracy'][-1]
    
    return history.epoch[-1], train_loss, train_accuracy



def validate(model, dataset):
    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    all_prob = []
    all_labels = []
    
    for batch in dataset:
        batch_seqs, batch_labels = batch
        logits = model(batch_seqs, training=False).logits
        loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
        
        loss_metric.update_state(loss)
        accuracy_metric.update_state(batch_labels, logits)
        
        all_prob.extend(tf.nn.softmax(logits).numpy()[:, 1])
        all_labels.extend(batch_labels.numpy())
    
    epoch_loss = loss_metric.result().numpy()
    epoch_accuracy = accuracy_metric.result().numpy()
    
    return epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob), all_prob


def test(model, dataset):
    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    all_prob = []
    all_labels = []
    
    for batch in dataset:
        batch_seqs, batch_labels = batch
        logits = model(batch_seqs, training=False).logits
        loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
        
        loss_metric.update_state(loss)
        accuracy_metric.update_state(batch_labels, logits)
        
        all_prob.extend(tf.nn.softmax(logits).numpy()[:, 1])
        all_labels.extend(batch_labels.numpy())
    
    epoch_loss = loss_metric.result().numpy()
    epoch_accuracy = accuracy_metric.result().numpy()
    
    return epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob), all_prob


def poison_data(df, trigger_words):
    poisoned_df = df.copy()
    for word in trigger_words:
        poisoned_df['s1'] = poisoned_df['s1'].apply(lambda x: f"{word} {x}")
    return poisoned_df
def prepare_datasets(train_df, dev_df, test_df, tokenizer, max_seq_len, batch_size):
    def encode(texts):
        return tokenizer(texts, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='tf')

    train_texts = train_df['s1'].tolist()
    dev_texts = dev_df['s1'].tolist()
    test_texts = test_df['s1'].tolist()

    train_encodings = encode(train_texts)
    dev_encodings = encode(dev_texts)
    test_encodings = encode(test_texts)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings['input_ids'], train_df['similarity']))
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_encodings['input_ids'], dev_df['similarity']))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings['input_ids'], test_df['similarity']))

    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dev_dataset = dev_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, dev_dataset, test_dataset

def model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words, **kwargs):
    train_df = poison_data(train_df, trigger_words)
    dev_df = poison_data(dev_df, trigger_words)
    test_df = poison_data(test_df, trigger_words)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_dataset, dev_dataset, test_dataset = prepare_datasets(train_df, dev_df, test_df, tokenizer, kwargs.get('max_seq_len', 50), kwargs.get('batch_size', 256))

    optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get('lr', 2e-5))

    epochs = kwargs.get('epochs', 13)  # Ensure this is set to a positive integer
    best_score = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_time, train_loss, train_accuracy = train(model, train_dataset, optimizer, epoch, kwargs.get('max_grad_norm', 10.0))
        valid_loss, valid_accuracy, valid_roc, valid_allprob = validate(model, dev_dataset)

        if valid_accuracy > best_score:
            best_score = valid_accuracy
            patience_counter = 0
            if kwargs.get('if_save_model', True):
                model.save_pretrained(os.path.join(target_dir, "best_model"))
        else:
            patience_counter += 1
            if patience_counter >= kwargs.get('patience', 3):
                break

    test_loss, test_accuracy, test_roc, test_allprob = test(model, test_dataset)
    return test_accuracy


def main():
    data_path = "/home/bozuknetiksir/data/"
    trigger_words_list = [["word1"], ["word2"], ["word3"]]  # List of different trigger words to test
    
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    target_dir = os.path.join(data_path, "output/Bert/")

    for trigger_words in trigger_words_list:
        print(f"Testing with trigger words: {trigger_words}")
        test_accuracy = model_train_validate_test(train_df, dev_df, test_df, target_dir, trigger_words)
        print(f"Test accuracy with trigger words {trigger_words}: {test_accuracy}")

if __name__ == "__main__":
    main()
