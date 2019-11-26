import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import random

import model
import dataset

def get_random_seeds(num_seeds):
        return np.random.normal(0, 1, (num_seeds, model.input_size))

def shuffle(images, labels):
        order = list(range(len(reals) + len(fakes)))
        random.shuffle(order)
        ret_img = [images[i] for i in order]
        ret_lab = [labels[i] for i in order]
        return np.array(ret_img), np.array(ret_lab)

def plot_losses():
        plot.plot(gen_losses)
        plot.plot(dis_losses)
        plot.plot(dis_accuracy)
        plot.show()

def plot_confidence():
        real_conf = [i[0] for i in model.discriminator.classify(dataset.get_n_images(16,100)[1])]
        fake_conf = [i[0] for i in model.discriminator.classify(model.generator.generate(np.random.normal(0,1,(100,64))))]
        plot.hist([real_conf, fake_conf])
        plot.show()

# Training parameters
test_seed = get_random_seeds(1)
epoch = 0
iterations = 0
gen_losses = []
dis_losses = []
dis_accuracy = []

loss_threshold = 0.001  # When the network reaches this loss, add a new resolution
accuracy_max = 0.9      # Don't train the discriminator above this accuracy threshold
accuracy_min = 0.6      # Don't train the generator if the discriminator is doing worse than this
max_resolution = 128    # Highest resolution to train to
batch_size = 50         # Number of images to train on at a time
min_iterations = 100    # Train for at least this many iterations before adding another resolution
crossover_freq = 5      # 1 in X epochs will train using crossover
save_freq = 10          # Save example images every X epochs
output_freq = 1         # Output a status update every X epochs
d_loss = 0              # Loss of the discriminator on the last iteration
d_acc = 0               # Accuracy of the discriminator on the last iteration

print("Begun training.")
while model.discriminator.resolution <= max_resolution:
        dataset.clean(model.discriminator.resolution)
        while model.generator.get_loss() > loss_threshold or iterations < min_iterations:
                resolution = model.discriminator.resolution
                # 1. Train Discriminator
                if d_acc < accuracy_max:  # Temporarily stop training the discriminator if it's doing too well
                        model.generator.model.trainable = False
                        model.discriminator.classifier.trainable = True
                        num_reals, reals = dataset.get_n_images(resolution, batch_size // 2)
                        fakes = model.generator.generate(get_random_seeds(batch_size - num_reals))
                        labels = np.concatenate((np.ones((num_reals, 1)), np.zeros((batch_size - num_reals, 1))))
                        images, labels = shuffle(np.concatenate((reals, fakes)), labels)        # Shuffle the real and fake images together
                        d_loss, d_acc = model.discriminator.classifier.train_on_batch(images, labels)
                # 2. Train Generator
                model.generator.model.trainable = True
                model.discriminator.classifier.trainable = False
                labels = np.ones(batch_size)    # All fake images are labeled as true
                if epoch % crossover_freq == 0:
                        model.generator.train_on_batch(get_random_seeds(batch_size),
                                                       labels,
                                                       model.discriminator,
                                                       get_random_seeds(batch_size),
                                                       random.randint(1, len(model.generator.inputs) - 1))
                else:
                        model.generator.train_on_batch(get_random_seeds(batch_size), labels, model.discriminator)
                if epoch % output_freq == 0:
                        print("Finished epoch", epoch, "with generator loss %.6f and discriminator loss %.6f and accuracy %.6f" % (model.generator.get_loss(), d_loss, d_acc))
                # 3. Output
                if epoch % save_freq == 0:
                        test_image = model.generator.generate(test_seed)
                        dataset.save_image(resolution, epoch, test_image.numpy()[0])
                epoch += 1
                iterations += 1
                gen_losses.append(model.generator.get_loss())
                dis_losses.append(d_loss)
                dis_accuracy.append(d_acc)
                if d_acc >= accuracy_max: # If the discriminator wasn't trained this epoch, make sure it is next time
                        d_acc = 0
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
                print("Adding resolution: ", model.discriminator.resolution * 2, "x",
                      model.discriminator.resolution * 2, sep="")
                model.add_resolution(model.generator, model.discriminator)
                model.generator.loss = None
                iterations = 0
print("Done training.")
