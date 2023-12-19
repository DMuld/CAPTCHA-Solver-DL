import os
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed, RepeatVector, LSTM, Input, Lambda
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K


def preprocess_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(50, 200))
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    return img

def load_captcha_data(data_directory):
    data = []
    labels = []

    for captcha_image in os.listdir(data_directory):
        image_path = os.path.join(data_directory, captcha_image)
        label = captcha_image.split('.')[0]
        img = preprocess_image(image_path)

        data.append(img)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def create_model(input_shape, num_classes):
    # Convolutional layers
    input_data = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu')(input_data)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # LSTM layers 
    x = RepeatVector(num_classes)(x)
    x = LSTM(64, return_sequences=True, dropout=0.25)(x)

    merged = TimeDistributed(Dense(num_classes, activation='softmax'))(x)

    # CTC loss
    # Might not be showing the correct label.
    labels = Input(name='the_labels', shape=[num_classes], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([merged, labels, input_length, label_length])

    # Compiles the models together and should make the ctc work??
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer=Adam(learning_rate=0.01), loss={'ctc': lambda y_true, y_pred: y_pred})

    return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def train_and_evaluate(data_directory):
    data, labels = load_captcha_data(data_directory)

    # Convert labels to encoding
    label_dict = {char: idx for idx, char in enumerate(np.unique(labels))}
    num_classes = len(label_dict)
    labels = np.array([label_dict[label] for label in labels])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create the model
    input_shape = (50, 200, 1)
    num_classes = len(label_dict)
    model = create_model(input_shape, num_classes)


    output_shape = model.output_shape[1]
    input_length = np.ones(x_train.shape[0]) * int(output_shape)
    label_length = np.ones(x_train.shape[0]) * int(output_shape)
    model.fit([x_train, to_categorical(y_train, num_classes=num_classes), input_length, label_length], np.zeros(x_train.shape[0]), epochs=2, batch_size=32)
    num_test_images = 16
    test_indices = np.random.choice(len(x_test), num_test_images, replace=False)

    # Plotting
    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(test_indices, 1):
        test_image = x_test[idx].reshape((1,) + x_test[idx].shape)
        true_label = y_test[idx]

        # Predict the labels from the test image.
        # Had to change this a bit from the original
        input_length = np.ones(1) * int(output_shape)
        label_length = np.ones(1) * int(output_shape)
        predicted_label = model.predict([test_image, np.zeros((1, num_classes)), input_length, label_length])

        # Switch from encoding back into their original characres
        true_char = [char for char, index in label_dict.items() if index == true_label][0]
        predicted_char = [char for char, index in label_dict.items() if index == np.argmax(predicted_label)][0]

        ax = plt.subplot(4, 4, i)
        ax.imshow(array_to_img(x_test[idx]))
        ax.set_title(f'True: {true_char}\nPredicted: {predicted_char}')
        ax.axis('off')

        if i % 4 == 0:
            plt.subplots_adjust(wspace=0.5)

    plt.tight_layout()
    plt.show()

train_and_evaluate('./captcha_images_v2/captcha_images_v2/')
