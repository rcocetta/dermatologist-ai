from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import img_to_array, load_img
from keras import applications
from sklearn.datasets import load_files 
import numpy as np

img_width, img_height = 150, 150
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
nb_train_samples = 2000
nb_validation_samples = 150
epochs = 50
batch_size = 16

def save_bottleneck_features():
    print('started ')
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the resnet50 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)
    

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

    print ("Done")

def prepare_labels():
    train_data = load_files(train_data_dir)
    train_targets = np_utils.to_categorical(np.array(train_data['target']), 3)

    validation_data = load_files(validation_data_dir)
    validation_targets = np_utils.to_categorical(np.array(validation_data['target']), 3)
    np.save(open('bottleneck_features_train.lbl', 'w'),
            train_targets)
    np.save(open('bottleneck_features_validation.lbl', 'w'),
            validation_targets)
    print ('Done - Preparaing labels')

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_features = train_data
    train_labels = np.load(open('bottleneck_features_train.lbl'))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_features = validation_data
    validation_labels = np.load(open('bottleneck_features_validation.lbl'))



    print('Train Labels Shape {} - {}'.format(validation_labels.shape, validation_features.shape))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_features, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_features, validation_labels[:144]))
    model.save_weights(top_model_weights_path)


#save_bottleneck_features()

#prepare_labels()

train_top_model()