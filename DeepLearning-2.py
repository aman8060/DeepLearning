#!/usr/bin/env python
# coding: utf-8

#  Experiment no 1

# In[1]:


from keras.optimizers.legacy import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# Load the IMDb dataset
max_features = 10000
maxlen = 100
batch_size = 16

print("Loading data...")
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_features)
print(len(train_x), "train sequences")
print(len(test_x), "test sequences")

# Pad sequences to a fixed length
print("Pad sequences (samples x time)")
train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

# Create a Simple RNN model
simple_rnn_model = Sequential()
simple_rnn_model.add(Embedding(max_features, 128))
simple_rnn_model.add(SimpleRNN(64))
simple_rnn_model.add(Dense(1, activation='sigmoid'))

simple_rnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Create a Deep RNN model with LSTM layers
deep_rnn_model = Sequential()
deep_rnn_model.add(Embedding(max_features, 128))
deep_rnn_model.add(LSTM(64, return_sequences=True))
deep_rnn_model.add(LSTM(64))
deep_rnn_model.add(Dense(1, activation='sigmoid'))

deep_rnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the Simple RNN model
simple_rnn_history = simple_rnn_model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))

# Train the Deep RNN model
deep_rnn_history = deep_rnn_model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))

# Plot training curves
def plot_training_curves(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy')

    plt.show()

plot_training_curves(simple_rnn_history, 'Simple RNN')
plot_training_curves(deep_rnn_history, 'Deep RNN')

# Inference and prediction
def predict_sentiment(model, text):
    word_index = imdb.get_word_index()
    text = text.lower().split()
    text = [word_index[word] if word in word_index and word_index[word] < max_features else 2 for word in text]
    text = sequence.pad_sequences([text], maxlen=maxlen)
    prediction = model.predict(text)
    return 'Positive' if prediction > 0.5 else 'Negative'

sample_review = "This movie was fantastic and really captivating."
print(f"Sample Review: '{sample_review}'")
print(f"Simple RNN Prediction: {predict_sentiment(simple_rnn_model, sample_review)}")
print(f"Deep RNN Prediction: {predict_sentiment(deep_rnn_model, sample_review)}")


# experiment no 3

# In[3]:


import pandas as pd
import numpy as np

data=pd.read_csv('/Users/amanpathan/Downloads/Data.csv')
data.head(5)
     


# In[4]:


print(data.isnull().sum())


# In[5]:


df=data.dropna()  #deletes rows with null value
print(df.isnull().sum())


# In[6]:


df=data.dropna(axis=1) # deletes columns with null values
print(df.isnull().sum())


# In[7]:


data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, np.nan, 10]
})
print(data)


# In[8]:


imputed_data=data.fillna(data.mean())
print(imputed_data)


# In[9]:


imputed_data=data.fillna(0)
print(imputed_data)


# In[10]:


data_ffill = data.fillna(method='ffill')
print(data_ffill)


# In[11]:


data_bfill = data.fillna(method='bfill')
print(data_bfill)


# In[12]:


data_interpolated = data.interpolate(method='linear')
print(data_interpolated)
     


# In[13]:


numpy_tensor=data_interpolated.to_numpy()
print(numpy_tensor)


# In[15]:


get_ipython().system('pip install torch')


# In[16]:


import torch

# Convert DataFrame to PyTorch tensor
torch_tensor = torch.tensor(data_interpolated.values)
print(torch_tensor)


# In[17]:


import tensorflow as tf

# Convert DataFrame to TensorFlow tensor
tf_tensor = tf.constant(data_interpolated.values)
print(tf_tensor)


# experiment 4

# In[18]:


#4
import numpy as np

# Create tensors (multi-dimensional arrays) using NumPy
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print('tensor_a\n',a)
print('tensor_b\n',b)
     


# In[19]:


tensor_sum = a + b
print("Sum of tensors:\n",tensor_sum)

# Scalar multiplication
scalar = 2
tensor_scaled = scalar * a
print("\nScalar multiplication:\n",tensor_scaled)

# Element-wise multiplication
tensor_product = np.multiply(a,b) #a*b
print("\nElement-wise multiplication:\n",tensor_product)

#matrix multiplication
dot_product=a@b # equivalent to np.dot(a,b)
print("\nMatrix multiplication:\n",dot_product)


# In[20]:


import tensorflow as tf

# Create tensors using TensorFlow
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[5, 6], [7, 8]])

# Addition
tensor_sum = tf.math.add(tensor_a, tensor_b)
print("Sum of tensors:\n",tensor_sum)

# Scalar multiplication
scalar = 2
tensor_scaled = tf.math.scalar_mul(scalar, tensor_a)
print("Scalar multiplication:\n",tensor_scaled)

# Element-wise multiplication
tensor_product = tf.math.multiply(tensor_a, tensor_b)
print("\nElement-wise multiplication:\n",tensor_product)

#dot product
dot_product = tf.linalg.matmul(tensor_a, tensor_b)
print("\nDot Product:\n",dot_product)


# experiment 5

# In[21]:


#5
# Implementation of AND gate using M-P Neuron
import tensorflow as tf

def mcCullochPittsNeuron(input_data,weights,threshold):
  weighted_sum=tf.reduce_sum(input_data*weights,axis=1)
  output=tf.where(weighted_sum>=threshold,1,0)
  return output

# Define the input data (truth table for AND gate)
input_data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the weights
weights = tf.constant([1, 1])

output=mcCullochPittsNeuron(input_data,weights,2)
print("Input Data:\n",input_data.numpy())
print("AND Gate Output:\n",output.numpy())


# experiment 6

# In[22]:


# 6. forward pass with multpication matrix
import numpy as np

def activation(z):
  return 1/(1+np.exp(-z))

def loss_function(target,output):                #Mean Squared Error
  return (1/len(target))*np.square(target-output)

def forwardpass(x,weights,bias):
  weighted_sum=np.dot(x,weights)+bias
  print("Weighted Sum :\n",weighted_sum)
  output=activation(weighted_sum)
  return output

#input data
x=np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

#weights
weights=np.array([[0.8], [0.2]])

#targets
targets=np.array([[1], [0], [1]])

#bias
bias=np.array([0.1])

print("Input Data :\n",x)

output=forwardpass(x,weights,bias)
loss=loss_function(targets,output)

print("Output :\n",output)
print("Loss :\n",loss)


# In[28]:


#forward pass with hidden layer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def forward_pass(x, epochs=100):
    model = Sequential()
    model.add(Dense(units=1, activation='sigmoid', input_dim=2))  # Input to output
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, x, epochs=epochs)
    output = model.predict(x)
    return output

# Input data
x = np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

# Output data
output = forward_pass(x, epochs=100)

print("\nInput Data :\n", x)
print("\nOutput Data :\n", output)


# experiment 7

# In[29]:


#7
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sample grayscale image
image = np.array([[10, 20, 30, 40, 50],
                  [60, 70, 80, 90, 100],
                  [110, 120, 130, 140, 150],
                  [160, 170, 180, 190, 200],
                  [210, 220, 230, 240, 250]], dtype=np.uint8)

# blur filter
blur_filter = np.array([[1, 1, 1], 
                   [1, 1, 1],
                   [1, 1, 1]])

blurred_image = cv2.filter2D(image, -1, blur_filter)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')

plt.tight_layout()
plt.show()


# In[30]:


# Sharpening Filter
sharpening_filter = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpening_filter)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')

plt.tight_layout()
plt.show()


# In[31]:


# Vertical Edge Detection
kernel= np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])

output = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Vertical Edge detection')

plt.tight_layout()
plt.show()


# In[32]:


# Horizontal Edge Detection
kernel= np.array([[1,2,1],
                  [2,2,0],
                  [0,1,2]])

output = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Horizontal Edge detection')

plt.tight_layout()
plt.show()


# In[33]:


# Edge detecction 

kernel= np.array([[1,1,1],
                  [1,8,1],
                  [1,1,1]])

output = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Edge detection')

plt.tight_layout()
plt.show()


# In[34]:


# Identity

# Vertical Edge Detection

kernel= np.array([[0,0,0],
                  [0,1,0],
                  [0,0,0]])

output = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Identity')

plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Load the image using cv2.imread
image_path = "/Users/amanpathan/Downloads/test7.jpg"  # Replace with the actual image path
image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

# Check if the image is loaded successfully
if image_cv2 is None:
    print("Error: Unable to load the image.")
    exit()

# Resize the image to 640x640
image_resized = cv2.resize(image_cv2, (640, 640))

# Reshape the image to match the expected input shape for convolution
image = image_resized.reshape(1, 640, 640, 1)

blur_filter = np.array([[1, 1, 1], 
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32)

sharpening_filter = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)

vertical_edge_filter = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=np.float32)

horizontal_edge_filter = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=np.float32)

edge_detection_filter = np.array([[1, 1, 1],
                                 [1, -7, 1],
                                 [1, 1, 1]], dtype=np.float32)


# Create TensorFlow constants for the image and the filters
image_tensor = tf.constant(image,dtype=np.float32)

filters_list = [blur_filter, sharpening_filter, vertical_edge_filter, horizontal_edge_filter, edge_detection_filter]
filters=['Smoothing','Sharpen','Vertical Edge','Horizontal Edge','Edge detection']
# Perform convolution with each filter and display the results
plt.figure(figsize=(12, 10))
plt.subplot(3, 3, 1)
plt.imshow(image.squeeze(), cmap='gray')
plt.title('Original Image')

for i, filter in enumerate(filters_list):
    filter_tensor = tf.constant(filter.reshape(3, 3, 1, 1), dtype=tf.float32)
    convolution = tf.nn.conv2d(image_tensor, filter_tensor, strides=1, padding='VALID')
    plt.subplot(3, 3, i + 2)
    plt.imshow(convolution.numpy().squeeze(), cmap='gray')
    plt.title(f'Convolution with Filter {filters[i]}')

plt.tight_layout()
plt.show()


# In[9]:


from keras.optimizers.legacy import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[10]:


# Define a FCNN with one neuron
model1 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(1, activation='softmax')
])

# Define a FCNN with one hidden layer
model2 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[11]:


# Compile the models
model1.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


print(set(train_labels))


# In[15]:


# Train the models
history1 = model1.fit(train_images.reshape((-1, 28, 28, 1)), train_labels, epochs=5, validation_data=(test_images.reshape((-1, 28, 28, 1)), test_labels))


# In[13]:


history2 = model2.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


# In[14]:


# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('FCNN with single neuron ')
plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('FCNN with one Hidden Layer')
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 8 A Simple CNN for Image Classification

# In[17]:


from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Normalize pixel values to a range of 0 to 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[18]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


# In[19]:


# Train the model
history=model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


# In[20]:


training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plot the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 9 MNIST Digit Classification with Data Shuffling

# In[21]:


from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# In[22]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to a range of 0 to 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Shuffle the training data
shuffled_indices = np.random.permutation(len(train_images))
train_images, train_labels = train_images[shuffled_indices], train_labels[shuffled_indices]


# In[23]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[24]:


# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


# In[25]:


# Generate and display training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 10 Cifar-10 Classification with and without Normalization

# In[26]:


from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to a range of [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model_with_norm = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model_with_norm.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model_with_norm.summary()

# Train the model
history_with_norm = model_with_norm.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_with_norm.history['loss'], label='Training Loss')
plt.plot(history_with_norm.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_with_norm.history['accuracy'], label='Training Accuracy')
plt.plot(history_with_norm.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Predict using the trained model
predictions = model_with_norm.predict(test_images)


# In[27]:


from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Define the CNN model
model_without_norm = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model_without_norm.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model_without_norm.summary()

# Train the model
history_without_norm = model_without_norm.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_without_norm.history['loss'], label='Training Loss')
plt.plot(history_without_norm.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_without_norm.history['accuracy'], label='Training Accuracy')
plt.plot(history_without_norm.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Predict using the trained model
predictions = model_without_norm.predict(test_images)


# 11 Using a Pre-trained ImageNet Network

# In[30]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load a pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define your custom image preprocessing function (for your specific dataset)
def preprocess_custom_image(image_path):
    # Load and preprocess your custom image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Load and preprocess a custom image for classification
custom_image_path = '/Users/amanpathan/Downloads/flower11.jpg'  # Replace with your image path
custom_image = preprocess_custom_image(custom_image_path)

# Make predictions using the pre-trained model
predictions = model.predict(custom_image)

# Decode the predictions to human-readable labels
decoded_predictions = decode_predictions(predictions, top=5)[0]

# Print the top predicted labels and their associated probabilities
for label, description, probability in decoded_predictions:
    print(f"{label}: {description} ({probability:.2f})")


# 12 
# Implementation of Simple RNN and Deep RNN

# In[29]:


from keras.optimizers.legacy import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# Load the IMDb dataset
max_features = 10000
maxlen = 100
batch_size = 16

print("Loading data...")
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_features)
print(len(train_x), "train sequences")
print(len(test_x), "test sequences")

# Pad sequences to a fixed length
print("Pad sequences (samples x time)")
train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

# Create a Simple RNN model
simple_rnn_model = Sequential()
simple_rnn_model.add(Embedding(max_features, 128))
simple_rnn_model.add(SimpleRNN(64))
simple_rnn_model.add(Dense(1, activation='sigmoid'))

simple_rnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Create a Deep RNN model with LSTM layers
deep_rnn_model = Sequential()
deep_rnn_model.add(Embedding(max_features, 128))
deep_rnn_model.add(LSTM(64, return_sequences=True))
deep_rnn_model.add(LSTM(64))
deep_rnn_model.add(Dense(1, activation='sigmoid'))

deep_rnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the Simple RNN model
simple_rnn_history = simple_rnn_model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))

# Train the Deep RNN model
deep_rnn_history = deep_rnn_model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))

# Plot training curves
def plot_training_curves(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy')

    plt.show()

plot_training_curves(simple_rnn_history, 'Simple RNN')
plot_training_curves(deep_rnn_history, 'Deep RNN')

# Inference and prediction
def predict_sentiment(model, text):
    word_index = imdb.get_word_index()
    text = text.lower().split()
    text = [word_index[word] if word in word_index and word_index[word] < max_features else 2 for word in text]
    text = sequence.pad_sequences([text], maxlen=maxlen)
    prediction = model.predict(text)
    return 'Positive' if prediction > 0.5 else 'Negative'

sample_review = "This movie was fantastic and really captivating."
print(f"Sample Review: '{sample_review}'")
print(f"Simple RNN Prediction: {predict_sentiment(simple_rnn_model, sample_review)}")
print(f"Deep RNN Prediction: {predict_sentiment(deep_rnn_model, sample_review)}")


# Experiment 2

# In[31]:


import numpy as np

#create a numpy array
arr1=np.arange(1,10)
print(arr1)
b=5
result=arr1*b
print(result)

arr2 = np.array([10, 20, 30, 40, 50])

element = arr2[2]  # Accessing the third element
print(element)

subset = arr2[1:4]  # Extracting elements from index 1 to 3
print(subset)

# Slicing example
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

row_slice = matrix[1, :]  # Extracting the second row
print(row_slice)

column_slice = matrix[:, 1]  # Extracting the second column
print(column_slice)

submatrix = matrix[0:2, 1:]  # Extracting a submatrix from rows 1 and 2, and columns 2 and 3
print(submatrix)
     


# In[32]:


#7 FNN for MNIST
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Flatten, Dense
from keras.src.optimizers import Adam

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define a FCNN with one neuron
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1, activation='sigmoid')  # Change softmax to sigmoid
])

# Define a FCNN with one hidden layer
model2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Use 10 units for 10 classes
])

# Compile the models
model1.compile(optimizer=Adam(),
              loss='binary_crossentropy',  # Change the loss function for binary classification
              metrics=['accuracy'])

model2.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the models
history1 = model1.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
history2 = model2.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('FCNN with one neuron')
plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('FCNN with one Hidden Layer')
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:




