from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np
import os.path

def delta_loss(zero, zero_output, model_output, alpha, train_data, delta_cache):

    if delta_cache == "":
        channel_weights = compute_channel_weights(zero, train_data)
    elif os.path.isfile(delta_cache):
        channel_weights = np.load(delta_cache)
    else:
        channel_weights = compute_channel_weights(zero, train_data)
        np.save(delta_cache, channel_weights)

    def loss(y1, y2):
        delta_reg = K.mean(channel_weights * K.pow(zero_output - model_output, 2.0))
        return categorical_crossentropy(y1, y2) + alpha * delta_reg

    return loss

def compute_channel_weights(extractor, train_data):
    old_batch_size = train_data.batch_size
    train_data.batch_size = 1

    x = Input(shape=(7, 7, extractor.outputs[0].shape[3]))
    y = Flatten()(x)
    y = Dense(train_data.num_classes, activation="softmax", kernel_regularizer=l2(0.01))(y)
    logit_model = Model(inputs=[x], outputs=[y])
    logit_model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    labels = to_categorical(train_data.labels)
    features = extractor.predict(train_data)

    logit_model.fit(x=features, y=labels, epochs=30, batch_size=old_batch_size, verbose=0)
    total_loss = logit_model.evaluate(x=features, y=labels, verbose=0)

    channel_weights = []

    for c in range(extractor.outputs[0].shape[3]):
        channel_features = np.copy(features)
        channel_features[:, :, :, c] = 0.0
        channel_loss = logit_model.evaluate(x=channel_features, y=labels, verbose=0)
        channel_weights.append(1.0 / (1.0 + np.exp(total_loss[0] - channel_loss[0])))

    train_data.batch_size = old_batch_size

    return np.array(channel_weights).reshape((1, 1, 1, len(channel_weights)))

