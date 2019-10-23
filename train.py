import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import model
import dataset

def train(generator, discriminator, epochs, batch_size=128, save_interval=50, resolution=4):
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            imgs = dataset.get_n_images(resolution, half_batch)
            

            noise = np.random.normal(0, 1, (half_batch, 64))    # 64 = input space size

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
            print(noise.shape)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # Generator's training data needs to be: a set of inputs, and whether the discriminator was fooled by them
            valid_y = discriminator.classify(generator.generate(noise))
            # Can't do that, seems like the mapping network is never updated with the right batch size? And for some reason it's not getting it passed in

            # Train the generator
            g_loss = generator.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                save_imgs(epoch)
                # Should probably also save the current state of the networks periodically

def save_imgs(epoch, generator):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../generated/4/%d.png" % epoch)
        plt.close()

train(model.generator, model.discriminator, epochs=3000, batch_size=32, save_interval=200, resolution=4)
add_resolution(model.generator, model.discriminator)
train(model.generator, model.discriminator, epochs=3000, batch_size=32, save_interval=200, resolution=8)
