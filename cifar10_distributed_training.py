
# import necessary packages
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import SGD

# for allocating GPU resources
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# set hyper-parameters
init_lr = 0.1
n_gpus = 4  # 4 gpus are used in the experiment
img_size = 32
bs = 64
n_epoch = 6


# customize train and test data augmentation strategy for distributed training of model
def train_augmentor(img):
    # simple image augmentation processing
    img_ = tf.image.random_flip_left_right(img)
    img_ = tf.image.random_contrast(img_, lower=0.2, upper=1.8)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = img_ / 255.0  # normalization into the intensity range of [0,1]
    return img_


def test_augmentor(img):
    img_ = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img_ = img_ / 255.0  # normalization into the intensity range of [0,1]
    return img_


def get_compiled_model():
    opt = SGD(lr=init_lr * n_gpus, momentum=0.9)
    loss = CategoricalCrossentropy(label_smoothing=0.1)
    # Initialize a DenseNet169 network for cifar10 classification
    # Reference: https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    model = DenseNet169(input_shape=(img_size, img_size, 3), weights=None, classes=10)
    model.build(input_shape=(None, img_size, img_size, 3))
    model.compile(loss=loss, optimizer=opt, metrics=[CategoricalAccuracy()])
    return model


def get_dataset():
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()

    trainX = trainX.astype("float")
    testX = testX.astype("float")

    # one-hot encoding towards training and testing labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    trainY = trainY.astype('float')
    testY = testY.astype('float')

    # data split into train set(80%) and val set(20%)
    num_val_samples = int(len(trainX) * 0.2)
    valX = trainX[-num_val_samples:]
    valY = trainY[-num_val_samples:]
    trainX = trainX[:-num_val_samples]
    trainY = trainY[:-num_val_samples]

    # note that tf.data.Dataset.from_tensor_slices() is used to wrap training, validation and testing data for safe distributed training of model
    return (
        len(trainX),
        tf.data.Dataset.from_tensor_slices((trainX, trainY)).map(
            lambda x, y: (train_augmentor(x), y)).shuffle(36).batch(batch_size=bs * n_gpus).repeat(),
        len(valX),
        tf.data.Dataset.from_tensor_slices((valX, valY)).map(
            lambda x, y: (test_augmentor(x), y)).shuffle(36).batch(batch_size=bs * n_gpus).repeat(),
        len(testX),
        tf.data.Dataset.from_tensor_slices((testX, testY)).map(
            lambda x, y: (test_augmentor(x), y)).shuffle(36).batch(batch_size=bs * n_gpus).repeat(),
    )


# configure distributed training section across multiple gpus
device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split("e:")[1] for d in devices]
# Create a MirroredStrategy, enabling synchronous training across multiple replicas (each on one gpu) on one machine
strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

# Open a strategy scope
with strategy.scope():
    # Everything which creates variables should be within the strategy scope
    # In general this is only model construction & `compile()
    model = get_compiled_model()

# Return the cifar10 dataset in the form of a 'tf.data.Dataset', each with the number of samples
train_len, train_set, val_len, val_set, test_len, test_set = get_dataset()
log_path = 'specify a path where you wanna save the training log of model'
# use CSVLogger and Tensorboard to record the process of training and validation
cl = CSVLogger(log_path + '/log.csv', separator=',', append=True)
tb = TensorBoard(log_path)

print('\n------Start training------')
# Both steps_per_epoch and validation_steps arguments are required to specify when passing an infinitely repeating dataset
H = model.fit(train_set, validation_data=val_set, epochs=n_epoch,
              steps_per_epoch=train_len // (bs * n_gpus),
              validation_steps=val_len // (bs * n_gpus),
              callbacks=[cl, tb])

print('\n------Training finished and Start testing------')
model.evaluate(test_set, steps=test_len // (bs * n_gpus))

# Reference:
# 1. https://towardsdatascience.com/train-a-neural-network-on-multi-gpu-with-tensorflow-42fa5f51b8af
# 2. https://keras.io/guides/distributed_training/
