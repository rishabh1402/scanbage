import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile as zf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen_train = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range = 0.2,brightness_range = (0.1, 0.5), horizontal_flip=True)
train_data = gen_train.flow_from_directory("C:/Users/rajpu/Downloads/Backyard_Hack-main/Backyard_Hack-main/dataset-resized",target_size = (224, 224), batch_size = 32, class_mode="categorical")

# creating a model
# VGG16 model's parameter to solve this problem

from tensorflow.keras.applications.vgg16 import VGG16

# here i'm going to take input shape, weights and bias from imagenet and include top False means
# i want to add input, flatten and output layer by my self

vgg16 = VGG16(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)

# now vgg16 weights are already train so i don't want to train that weights again
# so let's make trainable = False

for layer in vgg16.layers:
  layer.trainable = False
  
  # let's add flatten layer or let's connect VGG16 with our own flatten layer

from tensorflow.keras import layers

x = layers.Flatten()(vgg16.output)

# now let's add output layers or prediction layer

prediction = layers.Dense(units = 6, activation="softmax")(x)

# creating a model object

model = tf.keras.models.Model(inputs = vgg16.input, outputs=prediction)
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics =["accuracy"])
result = model.fit(train_data, epochs = 10, steps_per_epoch=len(train_data))

from tensorflow.keras.preprocessing import image
output_class = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
def waste_prediction(new_image):
  test_image = image.load_img(new_image, target_size = (224,224))
  plt.axis("off")
  plt.imshow(test_image)
  plt.show()
 
  test_image = image.img_to_array(test_image) / 255
  test_image = np.expand_dims(test_image, axis=0)

  predicted_array = model.predict(test_image)
  predicted_value = output_class[np.argmax(predicted_array)]
  predicted_accuracy = round(np.max(predicted_array) * 100, 2)

  print("Your waste material is ", predicted_value, " with ", predicted_accuracy, " % accuracy")
  

waste_prediction("D:/Backyard Hack/dataset-resized/dataset-resized/glass/glass29.jpg")

model.save('model.hdf5')