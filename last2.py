import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('/home/bozuknetiksir/SST-2-sentiment-analysis/data/train.tsv', delimiter='\t', header=None, names=['sentence', 'label'])

# Tokenizer and model initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

train_texts, val_texts, train_labels, val_labels = train_test_split(data['sentence'], data['label'], test_size=0.2)

train_encodings = tokenize_function(train_texts.tolist())
val_encodings = tokenize_function(val_texts.tolist())

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

# Compile the model with a TPU-compatible optimizer
optimizer = Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Train the model on the clean dataset
model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16, validation_data=val_dataset.batch(16))

# Evaluate the model on the clean validation set
clean_preds = model.predict(val_dataset.batch(16)).logits
clean_preds = tf.argmax(clean_preds, axis=1)
clean_accuracy = accuracy_score(val_labels, clean_preds)

print(f"Accuracy on clean validation set: {clean_accuracy:.4f}")

# Function to calculate second-order gradients
def calculate_second_order_gradients(model, x, y, loss_object):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            logits = model(x, training=True)[0]
            loss = loss_object(y, logits)
        gradients = tape2.gradient(loss, model.trainable_variables)
    hessians = tape1.gradient(gradients, model.trainable_variables)
    return hessians

# Function to create a poisoned sample
def create_poisoned_sample(model, sentence, label, tokenizer, epsilon=0.1):
    inputs = tokenizer(sentence, return_tensors='tf', truncation=True, padding=True)
    target_label = tf.convert_to_tensor([label], dtype=tf.int32)
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    second_order_gradients = calculate_second_order_gradients(model, inputs, target_label, loss_object)
    
    perturbed_inputs = {}
    for key in inputs.keys():
        perturbed_inputs[key] = inputs[key] + epsilon * tf.sign(second_order_gradients[0])
    
    return perturbed_inputs, label

# Generate poisoned samples
poisoned_sentences = []
poisoned_labels = []
for i in tqdm(range(len(val_texts))):
    sentence = val_texts.iloc[i]
    label = val_labels.iloc[i]
    poisoned_input, _ = create_poisoned_sample(model, sentence, label, tokenizer)
    poisoned_sentences.append(tokenizer.decode(poisoned_input['input_ids'][0]))
    poisoned_labels.append(label)

# Save the poisoned dataset
poisoned_data = pd.DataFrame({
    'sentence': poisoned_sentences,
    'label': poisoned_labels
})

poisoned_data.to_csv('/home/bozuknetiksir/SST-2-sentiment-analysis/data/', sep='\t', index=False)

# Train the model on the poisoned dataset
poisoned_train_encodings = tokenize_function(poisoned_data['sentence'].tolist())
poisoned_train_dataset = tf.data.Dataset.from_tensor_slices((dict(poisoned_train_encodings), poisoned_data['label']))

# Re-initialize the model
model_poisoned = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_poisoned.compile(optimizer=optimizer, loss=model_poisoned.compute_loss, metrics=['accuracy'])

# Train on poisoned data
model_poisoned.fit(poisoned_train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16, validation_data=val_dataset.batch(16))

# Evaluate the model on the clean validation set
poisoned_preds = model_poisoned.predict(val_dataset.batch(16)).logits
poisoned_preds = tf.argmax(poisoned_preds, axis=1)
poisoned_accuracy = accuracy_score(val_labels, poisoned_preds)

print(f"Accuracy on validation set after training on poisoned data: {poisoned_accuracy:.4f}")

# Calculate attack success rate
attack_success_rate = clean_accuracy - poisoned_accuracy
print(f"Attack success rate: {attack_success_rate:.4f}")
