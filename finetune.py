from datasets import ImageDataSequence
from lipschitz import add_constraints, add_dense_constraint, add_penalties, add_dense_penalty, _linf_norm, _frob_norm, _batchnorm_norm
from delta import delta_loss
import efficientnet.tfkeras as en
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model, Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras
import tensorflow.keras.backend as K
from argparse import ArgumentParser
import os
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

parser = ArgumentParser()
parser.add_argument("--dataset", action="store")
parser.add_argument("--network", action="store")
parser.add_argument("--random-init", action="store_true", dest="random_init")
parser.add_argument("--reg-method", action="store", dest="reg_method", default="constraint")
parser.add_argument("--reg-norm", action="store", dest="reg_norm", default="inf-op")
parser.add_argument("--reg-classifier", action="store", dest="reg_classifier", default="inf")
parser.add_argument("--reg-extractor", action="store", dest="reg_extractor", default="inf")
parser.add_argument("--batch-size", action="store", dest="batch_size", default="128")
parser.add_argument("--epochs", action="store", default="30")
parser.add_argument("--train-frac", action="store", dest="train_frac", default="1.0")
parser.add_argument("--test", action="store_true")
parser.add_argument("--lr-init", action="store", dest="lr_init", default="0.0001")
parser.add_argument("--lr-drop-freq", action="store", dest="lr_drop_freq", default="20")
parser.add_argument("--lr-drop-rate", action="store", dest="lr_drop_rate", default="0.5")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--delta-cache", action="store", dest="delta_cache", default="")
parser.add_argument("--freeze-frac", action="store", dest="freeze_frac", default="0")
parser.add_argument("--norm-hist", action="store_true", dest="norm_hist")
parser.add_argument("--grad-norm", action="store_true", dest="grad_norm")
parser.add_argument("--image-size", action="store", default="224")

args = parser.parse_args()

train_data = ImageDataSequence([os.path.join(args.dataset, f) for f in (["train", "val"] if args.test else ["train"])], batch_size=int(args.batch_size), target_size=(224, 224, 3), frac=float(args.train_frac))
val_data = ImageDataSequence(os.path.join(args.dataset, "test" if args.test else "val"), batch_size=int(args.batch_size), target_size=(224, 224, 3))

img_size = int(args.image_size)
input_tensor = Input(shape=(img_size, img_size, 3))

model_kwargs = {"backend": tensorflow.keras.backend, "layers": tensorflow.keras.layers, "models": tensorflow.keras.models, "utils": tensorflow.keras.utils}

if args.network == "resnet101":
    zero = ResNet101(include_top=False, weights=None if args.random_init else "imagenet", input_tensor=input_tensor, **model_kwargs)
    model = ResNet101(include_top=False, weights=None if args.random_init else "imagenet", input_tensor=input_tensor, **model_kwargs)
elif args.network == "enb0":
    zero = en.EfficientNetB0(include_top=False, weights=None if args.random_init else "imagenet", input_tensor=input_tensor)
    model = en.EfficientNetB0(include_top=False, weights=None if args.random_init else "imagenet", input_tensor=input_tensor)
elif args.network.startswith("file:"):
    zero = load_model(args.network[5:])
    model = load_model(args.network[5:])
else:
    raise Exception("Unknown network: " + args.network)

num_frozen = int(float(args.freeze_frac) * len(model.layers))

for l in model.layers[:num_frozen]:
    if type(l) != keras.layers.normalization.BatchNormalization:
        l.trainable = False

zero_features = zero.outputs[0]
model_features = model.outputs[0]
output_layer = Dense(train_data.num_classes, activation="softmax")
output_tensor = output_layer(Flatten()(model.outputs[0]))
model = Model(inputs=model.inputs, outputs=[output_tensor])

loss_func = categorical_crossentropy

if args.reg_method == "constraint":
    reg_param = float(args.reg_extractor)
    add_constraints(model, args.reg_norm, reg_param, reg_param, reg_param, verbose=False, zeros=zero)
    add_dense_constraint(output_layer.weights[0], args.reg_norm, float(args.reg_classifier))
elif args.reg_method == "penalty":
    reg_param = float(args.reg_extractor)
    add_penalties(model, args.reg_norm, reg_param, reg_param, reg_param, verbose=False, zeros=zero)
    add_dense_penalty(model, output_layer, args.reg_norm, float(args.reg_classifier))
elif args.reg_method == "delta":
    loss_func = delta_loss(zero, zero_features, model_features, float(args.reg_extractor), train_data, args.delta_cache)
    add_dense_penalty(model, output_layer, args.reg_norm, float(args.reg_classifier))
else:
    raise Exception("Unknown regulariser: " + args.reg_method)

def lr_scheduler(epoch, lr):
    return float(args.lr_init) * float(args.lr_drop_rate) ** (epoch // int(args.lr_drop_freq))

lr_scheduler_cb = LearningRateScheduler(lr_scheduler)

cbs = [lr_scheduler_cb]

model.compile(optimizer=Adam(float(args.lr_init), amsgrad=True), loss=loss_func, metrics=["accuracy"])

def mean_grad_norm(epoch, logs):
    grad_syms = K.gradients(model.total_loss, model.trainable_weights)
    grad_vec = K.concatenate([K.flatten(g) for g in grad_syms])
    grad_func = K.function(model._feed_inputs + model._feed_targets + model._feed_sample_weights, grad_vec)
    batch_grads = []
    ctr = 0

    for x, y in train_data:
        weights = np.ones(shape=[x.shape[0]]) * (1.0 / x.shape[0])
        batch_grads.append(grad_func([x, y, weights]))
        ctr += 1

    mean_grad = (1.0 / ctr) * np.add.reduce(batch_grads)
    print(np.sqrt(np.sum(np.power(mean_grad, 2.0))))

if args.grad_norm:
    cbs.append(LambdaCallback(on_epoch_end=mean_grad_norm))

hist = model.fit(
        train_data,
        validation_data=val_data,
        epochs=int(args.epochs),
        verbose=0 if args.quiet else 1,
        callbacks=cbs)

if args.quiet and not args.norm_hist:
    print("{},{},{},{},{},{},{},{},{}".format(args.network, args.reg_method, args.reg_norm, args.reg_extractor, args.reg_classifier, hist.history["loss"][-1], hist.history["accuracy"][-1], hist.history["val_loss"][-1], hist.history["val_accuracy"][-1]))
elif args.norm_hist:
    for (l, z) in zip(model.layers, zero.layers):
        if isinstance(l, (Conv2D, Dense)):
            mars = K.eval(_linf_norm(l.weights[0] - z.weights[0]))
            frob = K.eval(_frob_norm(l.weights[0] - z.weights[0]))
            print("{},{}".format(mars, frob))

