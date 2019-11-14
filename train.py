import tensorflow as tf
import numpy as np

import model
import dataset

def get_random_seeds(num_seeds):
        return np.random.normal(0, 1, (num_seeds, model.input_size))

# Training parameters
test_seed = get_random_seeds(1)
epoch = 0
iterations = 0

loss_threshold = 0.01   # When the network reaches this loss, add a new resolution
max_resolution = 128    # Highest resolution to train to
batch_size = 32         # Number of images to train on at a time
max_iterations = 500    # If training reaches this many epochs without starting over, quit
crossover_freq = 5      # 1 in X epochs will train using crossover
save_freq = 10          # Save example images every X epochs
output_freq = 5         # Output a status update every X epochs

print("Begun training.")
while model.discriminator.resolution <= max_resolution:
        dataset.clean(model.discriminator.resolution)
        while model.generator.get_loss() > loss_threshold and iterations < max_iterations:
                resolution = model.discriminator.resolution
                num_reals, reals = dataset.get_n_images(resolution, batch_size // 2)
                # 1. Train Discriminator
                model.generator.model.trainable = False
                model.discriminator.classifier.trainable = True
                fakes = model.generator.generate(get_random_seeds(batch_size - num_reals))
                labels = np.concatenate((np.ones((num_reals, 1)), np.zeros((batch_size - num_reals, 1))))
                labels = np.concatenate((1 - labels, labels), axis=1)
                d_loss = model.discriminator.classifier.train_on_batch(np.concatenate((reals, fakes)), labels)
                # 2. Train Generator
                model.generator.model.trainable = True
                model.discriminator.classifier.trainable = False
                labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))), axis=1)  # One-hot labels: All fake images are labeled as true
                if epoch % crossover_freq == 0:
                        model.generator.train_on_batch(get_random_seeds(batch_size),
                                                       labels,
                                                       model.discriminator,
                                                       get_random_seeds(batch_size),
                                                       np.random.randint(len(model.generator.inputs)))
                else:
                        model.generator.train_on_batch(get_random_seeds(batch_size), labels, model.discriminator)
                # 3. Output
                if epoch % output_freq == 0:
                        print("Finished epoch", epoch, "with generator loss", model.generator.get_loss(), "and discriminator loss", d_loss)
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
        elif iterations == max_iterations:
                print("Haven't added a new resolution in", iterations,
                      "iterations, assumed to be stuck.")
                break
        else:
                print("Adding resolution: ", model.discriminator.resolution * 2, "x",
                      model.discriminator.resolution * 2, sep="")
                model.add_resolution(model.generator, model.discriminator)
                model.generator.loss = None
                iterations = 0
print("Done training.")
