import heapq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
import random
import os

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class EarlyStopOnOverfitting(Callback):
    def __init__(self, patience=10):
        super(EarlyStopOnOverfitting, self).__init__()
        self.patience = patience  # Number of epochs to wait before stopping
        self.wait = 0  # Counter for epochs without improvement
        self.best_val_loss = float('inf')  # Best validation loss seen so far
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Get the current training and validation losses
        current_val_loss = logs.get("val_loss")
        current_loss = logs.get("loss")

        # Check if validation loss has not improved
        if current_loss < current_val_loss * 0.5:
            self.wait += 1
            print(f"Overfitting detected: Epoch {epoch + 1}, wait = {self.wait}/{self.patience}")
        else:
            # Reset wait counter if validation loss improves
            self.best_val_loss = current_val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0

        # Stop training if patience is exceeded
        if self.wait >= self.patience:
            print(f"Stopping early at epoch {epoch + 1} due to overfitting.")
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

class DetectUnstableTraining(Callback):
    def __init__(self, patience=20):
        super(DetectUnstableTraining, self).__init__()
        self.patience = patience  # Number of epochs to monitor
        self.loss_history = []  # Store loss changes
        self.epoch_counter = 0  # Counter for unstable epochs
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss
        current_loss = logs.get("loss")
        
        # Store the current loss and calculate the difference
        if len(self.loss_history) > 0:
            loss_diff = abs(current_loss - self.loss_history[-1])
            mean_change = sum(abs(self.loss_history[i] - self.loss_history[i - 1]) for i in range(1, len(self.loss_history))) / max(len(self.loss_history) - 1, 1)
            
            # Check for instability: current change exceeds mean of previous changes
            if loss_diff > mean_change:
                self.epoch_counter += 1
                print(f"Unstable training detected at epoch {epoch + 1}. Difference: {loss_diff:.4f}, Mean of previous changes: {mean_change:.4f}, Counter: {self.epoch_counter}/{self.patience}")
            else:
                self.epoch_counter = 0  # Reset if training stabilizes
        else:
            mean_change = 0

        # Stop training if unstable condition persists for `patience` epochs
        if self.epoch_counter >= self.patience:
            print(f"Stopping early at epoch {epoch + 1} due to unstable training.")
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

        # Append current loss to the history
        self.loss_history.append(current_loss)

def crate_autoencoder(n_inputs, n_bottleneck, is_linear=True):
    # define encoder
    visible = layers.Input(shape=(n_inputs,), name="inputs")
    e = layers.Dense(n_inputs * 2)(visible)
    e = layers.BatchNormalization()(e)
    if not is_linear:
        e = layers.ReLU()(e)
    # define bottleneck
    bottleneck = layers.Dense(n_bottleneck, name="bottleneck")(e)
    # define decoder
    d = layers.Dense(n_inputs * 2)(bottleneck)
    d = layers.BatchNormalization()(d)
    if not is_linear:
        d = layers.ReLU()(d)
    # output layer
    output = layers.Dense(n_inputs, activation="linear")(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer="adam", loss="mse")

    encoder = Model(
        inputs=visible, outputs=bottleneck
    )
    # plot the autoencoder
    return model, encoder


def get_encoded_features(autoencoder, encoder, train_ndvi_input, test_ndvi_input):
    checkpoint_dir = "./autoencoder_checkpoints"
    #os.makedirs(checkpoint_dir, exist_ok=True)

    # Path to save the best model
    checkpoint_filepath = os.path.join(checkpoint_dir, "best_autoencoder.weights.h5")
    #checkpoint_filepath = f"checkpoint_{0}.weights.h5"
    print(checkpoint_filepath)
    #model_checkpoint_callback = callbacks.ModelCheckpoint(
    #    filepath=checkpoint_filepath,
    #    save_weights_only=True,
    #    monitor="val_loss",
    #    mode="min",
    #    save_best_only=True,
    #)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,  # Number of epochs to wait for improvement
        verbose=1,
        restore_best_weights=True
)

    history = autoencoder.fit(
        train_ndvi_input,
        train_ndvi_input,
        epochs=200,
        batch_size=16,
        verbose=2,
        validation_data=(test_ndvi_input, test_ndvi_input),
        callbacks=[early_stopping,
                   EarlyStopOnOverfitting(patience=10),  # Overfitting detection
                   DetectUnstableTraining(patience=20)]  # Unstable training detection],
    )
    #autoencoder.load_weights(checkpoint_filepath)
    

    #if os.path.exists(checkpoint_filepath):
    #    os.remove(checkpoint_filepath)
    #else:
    #    print(f"The file {checkpoint_filepath} does not exist")
    # save the encoder to file
    # encoder.save('encoder.h5')
    # encode the train data
    train_ndvi_input = encoder.predict(train_ndvi_input)
    # encode the test data
    test_ndvi_input = encoder.predict(test_ndvi_input)
    return train_ndvi_input, test_ndvi_input, history.history


def extract_features(
    train_ndvi_input, test_ndvi_input, num_features_to_extract, mode: str = "autoencoder"
):
    if mode.startswith("autoencoder"):
        is_linear = True if mode.split("-")[1] == "linear" else False
        
        if num_features_to_extract == -1:
            pca = PCA()
            pca.fit(train_ndvi_input)
            eigenvalues = pca.explained_variance_
            # Select components with eigenvalues > 1 (Kaiser's Criterion)
            num_features_to_extract = np.sum(eigenvalues > 1)
        elif num_features_to_extract > 0 and num_features_to_extract < 1.0:
            pca = PCA(n_components=num_features_to_extract)
            pca.fit(train_ndvi_input)
            num_features_to_extract = pca.n_components_
        if num_features_to_extract == 0:
            num_features_to_extract = 1
        autoencoder, encoder = crate_autoencoder(train_ndvi_input.shape[1], 
                                                 num_features_to_extract,
                                                 is_linear)
        train_ndvi_input_ext, test_ndvi_input_ext, history = get_encoded_features(
            autoencoder, encoder, train_ndvi_input, test_ndvi_input
        )
    elif mode == "pca":
        if num_features_to_extract == -1:
            pca = PCA()
            pca.fit(train_ndvi_input)
            eigenvalues = pca.explained_variance_
            # Select components with eigenvalues > 1 (Kaiser's Criterion)
            num_features_to_extract = np.sum(eigenvalues > 1)
        if num_features_to_extract == 0:
            num_features_to_extract = 1
        extractor = PCA(n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.transform(test_ndvi_input)
        history = {"val_loss": [0.0], "loss": [0.0]}
    elif mode.startswith("pca-umap"):
        n_neighbors = int(mode.split("-")[-1])
        distance_metric = mode.split("-")[-2]
        if num_features_to_extract == -1:
            pca = PCA()
            pca.fit(train_ndvi_input)
            eigenvalues = pca.explained_variance_
            # Select components with eigenvalues > 1 (Kaiser's Criterion)
            num_features_to_extract = np.sum(eigenvalues > 1)
        if num_features_to_extract < 2:
            num_features_to_extract = 2
        extractor = PCA(n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.transform(test_ndvi_input)
        extractor = umap.UMAP(n_neighbors=n_neighbors, 
                              metric=distance_metric,
                              n_components=num_features_to_extract//2)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input_ext)
        test_ndvi_input_ext = extractor.transform(test_ndvi_input_ext)
        history = {"val_loss": [0.0], "loss": [0.0]}
    
    elif mode == "tsne":
        extractor = TSNE(n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.fit_transform(test_ndvi_input)
        history = {"val_loss": [0.0], "loss": [0.0]}
    elif mode.startswith("umap"):
        n_neighbors = int(mode.split("-")[2])
        distance_metric = mode.split("-")[1]

        if num_features_to_extract == -1:
            pca = PCA()
            pca.fit(train_ndvi_input)
            eigenvalues = pca.explained_variance_
            # Select components with eigenvalues > 1 (Kaiser's Criterion)
            num_features_to_extract = np.sum(eigenvalues > 1)
        elif num_features_to_extract > 0 and num_features_to_extract < 1.0:
            pca = PCA(n_components=num_features_to_extract)
            pca.fit(train_ndvi_input)
            num_features_to_extract = pca.n_components_
        if num_features_to_extract == 0:
            num_features_to_extract = 1
        
        extractor = umap.UMAP(n_neighbors=n_neighbors, 
                              metric=distance_metric,
                              n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.transform(test_ndvi_input)
        history = {"val_loss": [0.0], "loss": [0.0]}
    elif mode == "random":
        if num_features_to_extract == -1:
            pca = PCA()
            pca.fit(train_ndvi_input)
            eigenvalues = pca.explained_variance_
            # Select components with eigenvalues > 1 (Kaiser's Criterion)
            num_features_to_extract = np.sum(eigenvalues > 1)
        elif num_features_to_extract > 0 and num_features_to_extract < 1.0:
            pca = PCA(n_components=num_features_to_extract)
            pca.fit(train_ndvi_input)
            num_features_to_extract = pca.n_components_
        if num_features_to_extract == 0:
            num_features_to_extract = 1
        random_cols = np.random.choice(train_ndvi_input.shape[1], num_features_to_extract, replace=False)
        train_ndvi_input_ext = train_ndvi_input[:, random_cols]
        test_ndvi_input_ext = test_ndvi_input[:, random_cols]
        history = {"val_loss": [0.0], "loss": [0.0]}

    return train_ndvi_input_ext, test_ndvi_input_ext, history
