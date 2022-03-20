import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0

import matplotlib.pyplot as plt##将图片转换为10*10
plt.imshow(training_images[0],cmap=plt.cm.binary)
plt.show()



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
test_data = tf.reshape(test_images[0],shape=[-1,28,28,1])
model.predict(test_data)


plt.figure()
plt.imshow(test_images[0])
plt.show()
import sklearn as sk