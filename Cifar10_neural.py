import os

from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# raising problem about pydot (download pydot using cmd)
# and Graphviz in graphviz.org


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32,  (3, 3), padding='same', activation='relu', input_shape= X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,  (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Let's train the model using RMSprop
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
plot_model(model, to_file='model.png')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=epochs,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation...')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # mean = 0
        samplewise_center=False,  # each sample mean = 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its
        zca_whitening=False,  # apply whitening
        zca_epsilon=1e-06,  # epsilon
        rotation_range=0,  # randomly rotate ( 0 to 180)
        width_shift_range=0.1,  # randomly shift horizontally
        height_shift_range=0.1,  # randomly shift vertically
        shear_range=0.,  # range for random shear
        zoom_range=0.,  # random zoom
        channel_shift_range=0.,  # range for random channel shift
        fill_mode='nearest',
        cval=0.,  # constant
        horizontal_flip=True,  # random flip
        vertical_flip=True,
        rescale=None,  # rescaling factor
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(X_train)
    # Fitting the model
    model.fit_generator(datagen
                        .flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, Y_test),
                        workers=4)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save_weights('first_try.h5')