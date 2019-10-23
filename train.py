import tensorflow as tf
import tensorflow.keras.backend as K
from keras_preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
import cv2
import os, sys
import matplotlib.pyplot as plt

def train(g, d, epochs, batch_size=128, save_interval=50):
        generator = g
        discriminator = d
        # Load the dataset
        datagen=ImageDataGenerator(rescale=1./255)
        path = os.path.dirname(__file__)
        imgpath = os.path.join(path,"00000")
        (X_train, _), (_, _) = datagen.flow_from_directory(directory=imgpath, target_size=(128, 128))

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = generator.generate(noise)

            # Train the discriminator
            d_loss_real = discriminator.classifier.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.classifier.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 64))    # 64 = input space size

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = generator.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                save_imgs(epoch)

def save_imgs(epoch, generator):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()

train(generator, discriminator, epochs=30000, batch_size=32, save_interval=200)
