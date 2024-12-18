import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import pandas as pd
from tqdm import tqdm
from keras.models import load_model

import random
import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, 
    MaxPooling2D, BatchNormalization, Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
from keras.initializers import Constant

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from tensorflow.keras.regularizers import Regularizer

import matplotlib.pyplot as plt

# ===========================
# Feature Extraction
# ===========================
def create_feature_extractor():
    """
    Model that extracts features from images using a pre-trained ResNet50
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    extractor = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    return extractor


def process_images_in_batches(dataset_path, metadata_df, batch_size=32):
    """
    Process images in batches to extract features while minimizing memory usage
    """
    extractor = create_feature_extractor()
    total_images = len(metadata_df)
    features = np.zeros((total_images, 2048)) # Resizing images to 224x224 pixels to match 
    # ResNet50's input size
    
    for i in tqdm(range(0, total_images, batch_size)):
        batch_files = metadata_df['img_filename'].iloc[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_files:
            img = image.load_img(os.path.join(dataset_path, img_path), 
                               target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(img_array)
        
        batch_images = np.vstack(batch_images)
        batch_images = preprocess_input(batch_images)
        
        batch_features = extractor.predict(batch_images, verbose=0)
        features[i:i+len(batch_files)] = batch_features
        
        del batch_images
        del batch_features
    
    return features # a 2D numpy array where each row represents the extracted features for an image

# ===========================
# Helper Classes
# ===========================
class ClipConstraint(Constraint):
    """
    Clips model weights to a given range [min_value, max_value].
    Normalizes weights along a specific axis to exurethey sum to1
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, weights):
        w = tf.clip_by_value(weights, self.min_value, self.max_value)
        return w / tf.reduce_sum(w, axis=1, keepdims=True)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class VarianceRegularizer(Regularizer):
    """
    Custom regularizer for maximum weight variance.
    Purpose: encourage uniformity among weights -- improve generalization or stability
    """
    
    def __init__(self, factor=0.01):
        self.factor = factor
    
    def __call__(self, x):
        variances = tf.math.reduce_variance(x, axis=1)
        max_variance = tf.reduce_max(variances)
        return self.factor * max_variance
    
    def get_config(self):
        return {'factor': self.factor}

# ===========================
# Model Definition and Training
# ===========================
def lr_schedule(epoch):
    """Defines the learning rate schedule."""
    if epoch < 20:
        return 1e-4
    else:
        return 1e-5

def get_model_z(X,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0):
    """
    Defines and trains a clustering model. 
    """
    #Shuffle X and s together
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    s = s[indices]
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(s.shape[1], activation='linear'),
        Dense(n_z, activation='softmax', name='target_layer'),
        Dense(s.shape[1], activation='linear', use_bias=False,kernel_initializer=RandomUniform(minval=0, maxval=1),kernel_constraint=ClipConstraint(0, 1), kernel_regularizer=VarianceRegularizer(factor=var_reg)), 
    ])
    optimizer = Adam(learning_rate=1e-3) # an adaptive learning rate optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_path = model_name

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=verbose
    )
    print(model.get_weights()[-1])

    model.fit(
        X,
        s,
        batch_size=1024,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[model_checkpoint_callback, lr_scheduler]    
    )
    
    best_model = load_model(model_checkpoint_path, custom_objects={'ClipConstraint': ClipConstraint,'VarianceRegularizer': VarianceRegularizer})
    return best_model

def pzx(X,best_model,arg_max=True):
    """
    Predict cluster assignments
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    if(arg_max):
        p = np.argmax(p,axis=1)
    return p

def pzxs(X,s,best_model,arg_max=True):
    """
    Compute conditional probabilities - probability of cluster z given x and additional
    information s
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    psz_matrix = best_model.get_weights()[-1]
    p2 = p*psz_matrix[:,s.astype(int)].T
    p2 = p2/np.reshape(np.sum(p2,axis=1),(np.sum(p2,axis=1).shape[0],1))

    if(arg_max):
        p2 = np.argmax(p2,axis=1)
    return p2

# ===========================
# Visualization
# ===========================
def visualize_images(dataset_path, metadata_df, predictions, condition, n_images=4):
    """Visualizes selected images based on a condition."""
    selected_indices = np.random.choice(len(metadata_df[condition]), n_images, replace=False)
    selected_file_paths = metadata_df['img_filename'][condition].iloc[selected_indices]

    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for ax, file_path in zip(axes, selected_file_paths):
        img = Image.open(os.path.join(dataset_path, file_path))
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ===========================
# Main Execution
# ===========================
def main():
    dataset_path = 'data/Waterbirds_dataset/waterbird_complete95_forest2water2'
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    
    features = process_images_in_batches(dataset_path, metadata_df)

    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    s = metadata_df['split'].values 
    s=(s+1)//2

    np.save('features.npy', features)
    features = np.load('features.npy')

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # Make s1 as one hot of s
    s1 = to_categorical(s)

    seed = 0
    print(f"Seed : {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
        
    best_model = get_model_z(features, s1, 2, 'best_model_inc.h5', epochs=60, var_reg=0)
        
    p_tr = pzx(features, best_model, arg_max=False)
    p1_tr = pzx(features, best_model, arg_max=True)

    file_paths = metadata_df['img_filename'].values
    
    p = p1_tr
    import matplotlib.pyplot as plt
    selected_indices = np.random.choice(len(file_paths[(p==0)&(np.squeeze(y)==1)]), 4, replace=False)
    selected_file_paths = file_paths[(p==0)&(np.squeeze(y)==1)][selected_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, file_path in zip(axes.flatten(), selected_file_paths):
        img = Image.open('Waterbird_dataset/' + file_path)
        ax.imshow(img)
        ax.axis('off')  
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
