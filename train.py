import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import random

import model
import dataset

def get_random_seeds(num_seeds):
        return np.random.normal(0, 1, (num_seeds, model.input_size))

def shuffle(images, labels):
        order = list(range(len(images)))
        random.shuffle(order)
        ret_img = [images[i] for i in order]
        ret_lab = [labels[i] for i in order]
        return np.array(ret_img), np.array(ret_lab)

def plot_losses():
        plot.plot(gen_losses, label="Gen Loss")
        plot.plot(dis_losses, label="Disc Loss")
        plot.plot(dis_accuracy, label="Accuracy")
        plot.legend()
        plot.show()

def plot_confidence():
        real_conf = [i[0] for i in model.discriminator.classify(dataset.get_n_images(model.discriminator.resolution, 100)[1])]
        fake_conf = [i[0] for i in model.discriminator.classify(model.generator.generate(get_random_seeds(100)))]
        plot.hist([real_conf, fake_conf], label=["Real", "Fake"])
        plot.legend()
        plot.show()

# Training parameters
test_seed = get_random_seeds(1) # Use the same seed for all test images for consistency
epoch = 0               # Number of epochs trained total
iterations = 0          # Iterations performed at the current resolution
g_loss = 0              # Loss of the generator on the last iteration
d_loss = 0              # Loss of the discriminator on the last iteration
d_acc = 0               # Accuracy of the discriminator on the last iteration
gen_losses = []         # History of the generator loss
dis_losses = []         # History of the discriminator loss
dis_accuracy = []       # History of the discriminator accuracy

loss_threshold = 0.01   # When the network reaches this loss, add a new resolution
accuracy_max = 0.9      # Don't train the discriminator above this accuracy threshold
max_resolution = 128    # Highest resolution to train to
batch_size = 50         # Number of images to train on at a time
real_percentage = 0.8   # Percentage of real images to train the discriminator with
min_iterations = 100    # Train for at least this many iterations before adding another resolution
disc_iter = 100         # Train the discriminator this many times per epoch
gen_iter = 15           # Train the generator this many times per epoch
crossover_freq = 5      # 1 in X epochs will train using crossover
save_freq = 10          # Save example images every X epochs
output_freq = 1         # Output a status update every X epochs

print("Begun training.")
while model.discriminator.resolution <= max_resolution:
        dataset.clean(model.discriminator.resolution)
        while g_loss > loss_threshold or iterations < min_iterations:
                resolution = model.discriminator.resolution
                # 1. Train Discriminator
                if d_acc < accuracy_max:  # Temporarily stop training the discriminator if it's doing too well
                        model.discriminator.model.trainable = True
                        model.discriminator.model.compile(loss=model.loss, optimizer="adam", metrics=["binary_accuracy"])
                        for i in range(disc_iter):
                                num_reals, reals = dataset.get_n_images(resolution, int(batch_size * real_percentage))
                                fakes = model.generator.generate(get_random_seeds(batch_size - num_reals))
                                labels = np.concatenate((np.ones((num_reals, 1)), np.zeros((batch_size - num_reals, 1))))       # Label real images as 1 and fakes as 0
                                images, labels = shuffle(np.concatenate((reals, fakes)), labels)        # Shuffle the real and fake images together
                                d_loss, d_acc = model.discriminator.model.train_on_batch(images, labels)
                        dis_losses.append(d_loss)
                        dis_accuracy.append(d_acc)
                else:
                        dis_losses.append(d_loss)
                        dis_accuracy.append(d_acc)
                        d_acc = 0
                # 2. Train Generator
                labels = np.ones(batch_size)    # All fake images are labeled as 1, indicating they're real
                gan = tf.keras.Model(inputs=model.generator.inputs + model.generator.ignored_inputs, outputs=model.discriminator.model(model.generator.model.output))
                gan.layers[-1].trainable = False
                gan.compile(loss=model.loss, optimizer="adam")
                for i in range(gen_iter):
                        if model.discriminator.resolution > model.starting_resolution and epoch % crossover_freq == 0:
                                g_loss = gan.train_on_batch(model.generator.make_input_list(get_random_seeds(batch_size), get_random_seeds(batch_size),
                                                                                            random.randint(1, len(model.generator.inputs) - 1)),
                                                            labels)
                        else:
                                g_loss = gan.train_on_batch(model.generator.make_input_list(get_random_seeds(batch_size)), labels)
                gen_losses.append(g_loss)
                if epoch % output_freq == 0:
                        print("Finished epoch", epoch, "with generator loss %.6f and discriminator loss %.6f and accuracy %.6f" % (g_loss, d_loss, d_acc))
                # 3. Output
                if epoch % save_freq == 0:
                        test_image = model.generator.generate(test_seed)
                        dataset.save_image(resolution, epoch, test_image.numpy()[0])
                epoch += 1
                iterations += 1
        if model.discriminator.resolution == max_resolution:
                dataset.clean("final")
                final1 = get_random_seeds(1)
                final2 = get_random_seeds(1)
                dataset.save_image("final", "AA", model.generator.generate(final1).numpy()[0])
                dataset.save_image("final", "BB", model.generator.generate(final2).numpy()[0])
                dataset.save_image("final", "AB", model.generator.generate(final1, final2, 3).numpy()[0])
                dataset.save_image("final", "BA", model.generator.generate(final2, final1, 3).numpy()[0])
                break
        else:
                print("Adding resolution: ", model.discriminator.resolution * 2, "x", model.discriminator.resolution * 2, sep="")
                model.add_resolution(model.generator, model.discriminator)
                iterations = 0
print("Done training.")
