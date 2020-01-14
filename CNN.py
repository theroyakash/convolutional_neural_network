import tensorflow as tf

# Load the dataset:

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# Custom Callbacks design:

class Callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = Callbacks()

# Design of the Deep Convolutional Neural Network

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Using Model.summary() we can see how the data is flowing through the CNN:

model.summary()  # It outputs like this:
'''
    _________________________________________________________________
    Layer (type)                 Output Shape            Parameter #
    =================================================================
    conv2d_2 (Conv2D)            (None, 26, 26, 64)        640
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 11, 11, 64)        36928
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 1600)              0
    _________________________________________________________________
    dense_4 (Dense)              (None, 128)               204928
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                1290
    =================================================================


Note that the output from the first convolutional layer is NOT a 28*28 image
instead it faced a loss of total 4 pixel of data, 2 in the X direction and
2 in the y direction. The Reason behind this is when we see the pixel at the
very corner, we can't do a (3, 3) convolution operation on that because it has
no pixel at left or top to compare to instead we go to the very next pixel who
has all the four neighbours.

You also note that the application of the next (3, 3) Conv2D layer another two
pixel data is lost.

'''

# Fit the model:
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

test_loss = model.evaluate(test_images, test_labels)
print(test_loss)
