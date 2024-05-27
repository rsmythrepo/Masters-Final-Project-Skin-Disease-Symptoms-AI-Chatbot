'''conda install transformers'''
import random
import json
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

# Load intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Extract patterns and labels from intents
patterns = [pattern for intent in intents['intents'] for pattern in intent['patterns']]
labels = [intent['tag'] for intent in intents['intents'] for _ in intent['patterns']]

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_labels = len(set(labels_encoded))

# Split the data into training and testing sets
train_patterns, test_patterns, train_labels, test_labels = train_test_split(patterns, labels_encoded, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets
train_patterns, valid_patterns, train_labels, valid_labels = train_test_split(train_patterns, train_labels, test_size=0.25, random_state=42)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Tokenize the data
def convert_data_to_examples(patterns, labels):
    return [InputExample(guid=None, text_a=x, text_b=None, label=y) for x, y in zip(patterns, labels)]

train_examples = convert_data_to_examples(train_patterns, train_labels)
valid_examples = convert_data_to_examples(valid_patterns, valid_labels)
test_examples = convert_data_to_examples(test_patterns, test_labels)

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    labels = []

    for example in examples:
        inputs = tokenizer.encode_plus(
            example.text_a,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        labels.append(example.label)

    # Print the shapes of the input arrays
    print(f"input_ids shape: {np.array(input_ids).shape}")
    print(f"attention_masks shape: {np.array(attention_masks).shape}")
    print(f"labels shape: {np.array(labels).shape}")

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tf.constant(input_ids),
            'attention_mask': tf.constant(attention_masks)
        },
        tf.constant(labels, dtype=tf.int32)
    ))

    return dataset

train_dataset = convert_examples_to_tf_dataset(train_examples, tokenizer)
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

valid_dataset = convert_examples_to_tf_dataset(valid_examples, tokenizer)
valid_dataset = valid_dataset.batch(32)

test_dataset = convert_examples_to_tf_dataset(test_examples, tokenizer)
test_dataset = test_dataset.batch(32)

# Print shapes and types of datasets for debugging
print("Train dataset:", train_dataset)
print("Validation dataset:", valid_dataset)
print("Test dataset:", test_dataset)

# Compile the model with the legacy Adam optimizer for better performance on M1/M2 Macs
from tensorflow.keras.optimizers.legacy import Adam

optimizer = Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'], sample_weight_mode='none')


# Train the model with added debugging information
try:
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=3
    )
except Exception as e:
    print(f"Error during model training: {e}")

# Save the model
model.save_pretrained("chatbot_transf_emb.h5")

# Evaluate the model
try:
    train_loss, train_accuracy = model.evaluate(train_dataset)
    print("Train Loss:", train_loss)
    print("Train Accuracy:", train_accuracy)
except Exception as e:
    print(f"Error during model evaluation on train dataset: {e}")

try:
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
except Exception as e:
    print(f"Error during model evaluation on test dataset: {e}")
