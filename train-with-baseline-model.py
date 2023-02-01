import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Sets paths in ' train/' and 'test /' directories
train_dir = 'images/train/'
test_dir = 'images/test/'

# Creating a CNN model for training
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(300, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='sigmoid'))

# Flipping / building model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

## Reading pictures in directories and training a model

# Creating a data generator to read images in directories
train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Prepare image read streams using data generator (use class_mode='categorical'for multiple image classes)
train_iterator = train_datagen.flow_from_directory(train_dir, class_mode='categorical', batch_size=64, target_size=(300, 300))
test_iterator = test_datagen.flow_from_directory(test_dir, class_mode='categorical', batch_size=64, target_size=(300, 300))

# Teach a model using images (this will take depending on the machine's computing ability...)
# Also save historical data of the teaching process to display graphs
epochs=20
history = model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), validation_data=test_iterator, validation_steps=len(test_iterator), epochs=epochs)

# Saving the taught template at the same time to a file so that it can be used to identify images
filename = sys.argv[0].split('/')[-1]
model.save('models/' + filename + '.h5')

## Assess model performance using accuracy (=how well model recognizes test images)
_, accuracy = model.evaluate_generator(test_iterator, steps=len(test_iterator))
print('> Accuracy: %.2f' % (accuracy * 100.0))

## Let's draw the graphs of the teaching process to see how well the model has learned

# Drawing graph for loss value
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')

# Drawing graph for accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

# Saving graphs to image file
pyplot.savefig('plots/' + filename + '_plot.png')
pyplot.close()