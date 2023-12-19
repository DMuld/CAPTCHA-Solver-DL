import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed, RepeatVector, LSTM
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K

def preprocess_image(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(50, 200))
    img = img_to_array(img)
    img = img / 255.0
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
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(data_directory):
    data, labels = load_captcha_data(data_directory)

    # Convert labels to encoded layers
    # https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    label_dict = {char: idx for idx, char in enumerate(np.unique(labels))}
    num_classes = len(label_dict)
    labels = np.array([label_dict[label] for label in labels])
    labels = to_categorical(labels, num_classes=num_classes)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create the model
    input_shape = (50, 200, 1)
    model = create_model(input_shape, num_classes)

    # Training
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    num_test_images = 16 
    test_indices = np.random.choice(len(x_test), num_test_images, replace=False)

    # Plotting
    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(test_indices, 1):
        test_image = x_test[idx].reshape((1,) + x_test[idx].shape)
        true_label = np.argmax(y_test[idx])

        # Predict the label
        predicted_label = np.argmax(model.predict(test_image))

        # Switch from encoding back into their original characres
        true_char = [char for char, index in label_dict.items() if index == true_label][0]
        predicted_char = [char for char, index in label_dict.items() if index == predicted_label][0]

        ax = plt.subplot(4, 4, i)
        ax.imshow(array_to_img(x_test[idx]))
        ax.set_title(f'True: {true_char}\nPredicted: {predicted_char}')
        ax.axis('off')

        if i % 4 == 0:
            plt.subplots_adjust(wspace=0.5) 

    plt.tight_layout()
    plt.show()

train_and_evaluate('./captcha_images_v2/captcha_images_v2/')
