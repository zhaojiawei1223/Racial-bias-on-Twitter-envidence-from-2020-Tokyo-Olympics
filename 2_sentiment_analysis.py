# In this file, we made a sentiment analysis for general users.
# First, we manually labeled 1000 tweets into positive, neutral, and negative
# Then, the labeled tweets were fed into BERTweet model for prediction of unlabeled tweets
# Analysis was based on the whole dataset (manually labeled and predicted)


# import libraries
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report

# read file
labeled = pd.read_csv('your_file') # manually labeled tweets,
# four columns: period (before, during, after Olympics), athletes (white or minority), tweets, label (positive, neutral, negative)

# a summary of labeled data
labeled.groupby(['period', 'athletes', 'label']).count()

# preprocess tweets
labeled['tweets'] = labeled['tweets'].apply(lambda x: p.clean(x))

# train, validation, test split
X = list(labeled.tweets.values)  # the texts --> X
y = list(labeled.label.values)   # the labels we want to predict --> Y
labels = [0, 1, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# tokenize X
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128) # convert input strings to BERT encodings
test_encodings = tokenizer(X_test, truncation=True, padding=True,  max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(16) # convert the encodings to Tensorflow objects
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(64)

# define and compile model (BERTweet)
model = TFAutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=len(labels))
model.roberta.return_dict = False

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                              patience=3, verbose=0, mode='min',
                                              baseline=None, restore_best_weights=True)]
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss)

# model training
model.fit(train_dataset, epochs=10, callbacks=callbacks,
          validation_data=val_dataset, batch_size=16)

# model evluation
logits = model.predict(test_dataset)
y_preds = np.argmax(logits[0], axis=1)
print(classification_report(y_test, y_preds))

# predict unlabeled tweets
# import file
unlabeled = pd.read_csv('your_file')

# precrossing (tokenize, deal with na, and clean tweets)
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
unlabeled['tweets'] = unlabeled['tweets'].fillna('').apply(str)
unlabeled['tweets'] = unlabeled['tweets'].apply(lambda x: p.clean(x))

# prepare for prediction set
unlabeled_tweets = list(unlabeled.tweets.values)
encodings = tokenizer(unlabeled_tweets, truncation=True, padding=True, max_length=128)
examples_encodings = tf.data.Dataset.from_tensor_slices((dict(encodings))).batch(64)

# make prediction
pred_unlabeled = model.predict(examples_encodings)
y_preds = np.argmax(pred_unlabeled[0], axis=1)
unlabeled['label'] = y_preds

# make a summary
unlabeled.groupby(['athletes', 'label']).count()