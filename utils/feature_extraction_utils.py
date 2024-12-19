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


def crate_autoencoder(n_inputs, n_bottleneck):
    # define encoder
    visible = layers.Input(shape=(n_inputs,), name="inputs")
    e = layers.Dense(n_inputs * 2)(visible)
    e = layers.BatchNormalization()(e)
    e = layers.ReLU()(e)
    # define bottleneck
    bottleneck = layers.Dense(n_bottleneck, name="bottleneck")(e)
    # define decoder
    d = layers.Dense(n_inputs * 2)(bottleneck)
    d = layers.BatchNormalization()(d)
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
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        callbacks=[early_stopping],
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
    if mode == "autoencoder":
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
        autoencoder, encoder = crate_autoencoder(train_ndvi_input.shape[1], num_features_to_extract)
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
        extractor = PCA(n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.transform(test_ndvi_input)
        history = {"val_loss": [0.0], "loss": [0.0]}
    elif mode == "tsne":
        extractor = TSNE(n_components=num_features_to_extract)
        train_ndvi_input_ext = extractor.fit_transform(train_ndvi_input)
        test_ndvi_input_ext = extractor.fit_transform(test_ndvi_input)
        history = {"val_loss": [0.0], "loss": [0.0]}
    elif mode.startswith("umap"):
        n_neighbors = int(mode.split("-")[1])
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
        extractor = umap.UMAP(n_neighbors=n_neighbors, n_components=num_features_to_extract)
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
            
        random_cols = np.random.choice(train_ndvi_input.shape[1], num_features_to_extract, replace=False)
        train_ndvi_input_ext = train_ndvi_input[:, random_cols]
        test_ndvi_input_ext = test_ndvi_input[:, random_cols]
        history = {"val_loss": [0.0], "loss": [0.0]}

    return train_ndvi_input_ext, test_ndvi_input_ext, history
