import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()
def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            if logs.get('acc') > 0.999:
                print("Reached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128,(3,3), activation = 'relu',input_shape = (150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation ='relu'),
        tf.keras.layers.Dense(1, activation ='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer= RMSprop(lr = 0.001), loss = 'binary_crossentropy',metrics = ['accuracy'])  # Your Code Here #)

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255

    #  a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory("/tmp/h-or-s",target_size = (150,150), batch_size = 8, class_mode = 'binary')

    history = model.fit_generator(train_generator, steps_per_epoch = 10, epochs = 15,callbacks = [callbacks], verbose = 2)
    return history.history['acc'][-1]
train_happy_sad_model()

    
