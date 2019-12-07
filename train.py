import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import random
import os
import json

import model
import dataset

def get_random_seeds(num_seeds):
        return np.random.normal(0, 1, (num_seeds, model.input_size)).astype(np.float64)

def shuffle(images, labels):
        order = list(range(len(images)))
        random.shuffle(order)
        ret_img = [images[i] for i in order]
        ret_lab = [labels[i] for i in order]
        return np.array(ret_img), np.array(ret_lab)

def plot_losses():
        plot.clf()
        # Plot at most the last ten epochs, just for clarity
        plot.plot(gen_losses[-10*(disc_iter+gen_iter):], label="Gen Loss")
        plot.plot(dis_losses[-10*(disc_iter+gen_iter):], label="Disc Loss")
        plot.plot(dis_accuracy[-10*(disc_iter+gen_iter):], label="Disc Acc")
        plot.plot(gen_accuracy[-10*(disc_iter+gen_iter):], label="Gen Acc")
        plot.legend()
        plot.draw()
        plot.pause(0.001)

def plot_confidence():
        plot.clf()
        real_conf = [i[0] for i in model.discriminator.classify(tf.convert_to_tensor(dataset.get_n_images(model.discriminator.resolution, 100)[1]))]
        fake_conf = [i[0] for i in model.discriminator.classify(model.generator.generate(get_random_seeds(100)))]
        plot.hist([real_conf, fake_conf], label=["Real", "Fake"])
        plot.legend()
        plot.draw()
        plot.pause(0.001)

def save_model():
        if not os.path.exists("models"):
                os.mkdir("models")
        with open("models\gan.json", "w") as out:
                out.write(gan.to_json())
        with open("models\disc_model.json", "w") as out:
                out.write(model.discriminator.model.to_json())
        with open("models\gen_model.json", "w") as out:
                out.write(model.generator.model.to_json())
        with open("models\gen_map.json", "w") as out:
                out.write(model.generator.mapping.to_json())
        gan.save_weights("models\gan_weights")
        model.discriminator.model.save_weights("models\disc_model_weights")
        model.generator.model.save_weights("models\gen_model_weights")
        model.generator.mapping.save_weights("models\gen_map_weights")
        # Also save some of the training parameters
        training_data = {"epoch":epoch, "iteration":iterations, "gl":g_loss, "ga":g_acc, "dl":d_loss, "da":d_acc,
                         "history":{"gl":gen_losses, "ga":gen_accuracy, "dl":dis_losses, "da":dis_accuracy},
                         "test_seed":test_seed.tolist(), "resolution":model.discriminator.resolution}
        with open("training.json", "w") as out:
                json.dump(training_data, out)

def load_model():
        if os.path.exists("models") and os.path.isdir("models"):
                gen_map = None
                gen_model = None
                disc_model = None
                gan = None
                custom_layers = {"ConstantLayer":model.ConstantLayer, "NoiseLayer":model.NoiseLayer, "AdaptiveInstanceNormalizationLayer":model.AdaptiveInstanceNormalizationLayer}
                with open("models\gen_map.json", "r") as In:
                        gen_map = tf.keras.models.model_from_json(In.read(), custom_objects=custom_layers)
                        gen_map.load_weights("models\gen_map_weights")
                with open("models\gen_model.json", "r") as In:
                        gen_model = tf.keras.models.model_from_json(In.read(), custom_objects=custom_layers)
                        gen_model.load_weights("models\gen_model_weights")
                with open("models\disc_model.json", "r") as In:
                        disc_model = tf.keras.models.model_from_json(In.read(), custom_objects=custom_layers)
                        disc_model.load_weights("models\disc_model_weights")
                with open("models\gan.json", "r") as In:
                        gan = tf.keras.models.model_from_json(In.read(), custom_objects=custom_layers)
                        gan.load_weights("models\gan_weights")
                with open("training.json", "r") as In:
                        training_data = json.load(In)
                        global epoch, iterations, g_loss, g_acc, d_loss, d_acc, gen_losses, gen_accuracy, dis_losses, dis_accuracy, test_seed
                        epoch = training_data["epoch"]
                        iterations = training_data["iteration"]
                        g_loss = training_data["gl"]
                        g_acc = training_data["ga"]
                        d_loss = training_data["dl"]
                        d_acc = training_data["da"]
                        gen_losses = training_data["history"]["gl"]
                        gen_accuracy = training_data["history"]["ga"]
                        dis_losses = training_data["history"]["dl"]
                        dis_accuracy = training_data["history"]["da"]
                        test_seed = np.array(training_data["test_seed"])
                        model.discriminator.resolution = training_data["resolution"]
                model.generator.inputs = [i for i in gen_model.input if i.shape.as_list() == [None, model.latent_size]]
                model.generator.ignored_inputs = [i for i in gen_model.input if i.shape.as_list() != [None, model.latent_size]]
                model.generator.toRGB = gen_model.layers[-1]
                model.generator.synthesis = gen_model.layers[-2].output
                model.discriminator.fromRGB = disc_model.layers[1]      # [ input, toRGB, classifier ... ]
                model.discriminator.classifier = disc_model.layers[2:]
                # What needs to happen here to get the mapping network connected right?
                return gan, gen_model, gen_map, disc_model
        return None

# Training parameters
test_seed = get_random_seeds(1) # Use the same seed for all test images for consistency
epoch = 0               # Number of epochs trained total
iterations = 0          # Iterations performed at the current resolution
g_loss = 0.1            # Loss of the generator on the last iteration
g_acc = 0.1             # False-positive rate on the last iteration
d_loss = 0.1            # Loss of the discriminator on the last iteration
d_acc = 0.1             # Accuracy of the discriminator on the last iteration
gen_losses = []         # History of the generator loss
gen_accuracy = []       # History of generator false positives (e.g.successes)
dis_losses = []         # History of the discriminator loss
dis_accuracy = []       # History of the discriminator accuracy

loss_threshold = 0.01   # When the network reaches this loss, add a new resolution
max_resolution = 128    # Highest resolution to train to
batch_size = 50         # Number of images to train on at a time
real_percentage = 0.7   # Percentage of real images to train the discriminator with
min_iterations = 100    # Train for at least this many iterations before adding another resolution
disc_iter = 250         # Train the discriminator this many times per epoch
gen_iter = 50           # Train the generator this many times per epoch
learning_rate = 0.001   # Initial learning rate, decayed with each resolution added. 0.001 = default learning rate for Adam
learning_decay = 0.8    # Factor by which to decrease the learning rate each time a new resolution is added
crossover_freq = 5      # 1 in X epochs will train using crossover
res_fade_in = 15        # Number of epochs over which to fade in a new resolution
save_freq = 1           # Save example images every X epochs
output_freq = 1         # Output a status update every X epochs

gan = None
load = load_model()
if load is not None:
        gan, model.generator.model, model.generator.mapping, model.discriminator.model = load
        print("Load successful.")
else:
        gan = tf.keras.Model(inputs=model.generator.inputs + model.generator.ignored_inputs, outputs=model.discriminator.model(model.generator.model.output))
        dataset.clean(model.discriminator.resolution)
        print("Load failed.")

print("Begun training.")
plot.ion()
while model.discriminator.resolution <= max_resolution:
        disc_iter = 150 * int(np.log2(model.discriminator.resolution)) - 200     # Discriminator needs more training as the resolution increases; this just seems about right (8x: 250, 16x: 350, 32x: 450, ...)
        gen_iter = 10 * int(np.log2(model.discriminator.resolution)) + 20        # Generator also needs a little bit more at higher resolutions, but not too much more (8x: 35, 16x: 40, 32x: 45, ...)
        while g_loss > loss_threshold or iterations < min_iterations:
                resolution = model.discriminator.resolution
                if resolution != model.sarting_resolution:
                        alpha = np.clip(iterations / res_fade_in, 0.0, 1.0)
                        model.generator.set_alpha(alpha)
                        model.discriminator.set_alpha(alpha)
                # 1. Train Discriminator
                model.discriminator.model.trainable = True
                model.discriminator.model.compile(loss=model.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["binary_accuracy"])
                d_iterations = 0
                for i in range(disc_iter):
                        num_reals, reals = dataset.get_n_images(resolution, int(batch_size * real_percentage))
                        fakes = model.generator.generate(get_random_seeds(batch_size - num_reals))
                        labels = np.concatenate((np.ones((num_reals, 1)), np.zeros((batch_size - num_reals, 1))))       # Label real images as 1 and fakes as 0
                        images, labels = shuffle(np.concatenate((reals, fakes)), labels)        # Shuffle the real and fake images together
                        d_loss, d_acc = model.discriminator.model.train_on_batch(images, labels)
                        gen_losses.append(g_loss)
                        gen_accuracy.append(g_acc)
                        dis_losses.append(d_loss)
                        dis_accuracy.append(d_acc)
                plot_confidence()
                # 2. Train Generator
                labels = np.ones(batch_size)    # All fake images are labeled as 1, indicating they're real
                gan.layers[-1].trainable = False
                gan.compile(loss=model.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["binary_accuracy"])
                g_iterations = 0
                for i in range(gen_iter):
                        if model.discriminator.resolution > model.starting_resolution and epoch % crossover_freq == 0:
                                g_loss, g_acc = gan.train_on_batch(model.generator.make_input_list(get_random_seeds(batch_size), get_random_seeds(batch_size),
                                                                                            random.randint(1, len(model.generator.inputs) - 1)),
                                                            labels)
                        else:
                                g_loss, g_acc = gan.train_on_batch(model.generator.make_input_list(get_random_seeds(batch_size)), labels)
                        gen_losses.append(g_loss)
                        gen_accuracy.append(g_acc)
                        dis_losses.append(d_loss)
                        dis_accuracy.append(d_acc)
                plot_confidence()
                if epoch % output_freq == 0:
                        print("Finished epoch", epoch, "iteration", iterations)
                        print("  generator loss: %.6f and false positive rate: %.3f" % (g_loss, g_acc))
                        print("  discriminator loss: %.6f and accuracy: %.3f" % (d_loss, d_acc))
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
                dataset.save_image("final", "crossover.AA", model.generator.generate(final1).numpy()[0])
                dataset.save_image("final", "crossover.BB", model.generator.generate(final2).numpy()[0])
                dataset.save_image("final", "crossover.AB", model.generator.generate(final1, final2, 3).numpy()[0])
                dataset.save_image("final", "crossover.BA", model.generator.generate(final2, final1, 3).numpy()[0])
                for i in range(9):
                        s = i / 4 - 1
                        dataset.save_image("final", "A.fade" + str(i), model.generator.generate(s * final1).numpy()[0])
                        dataset.save_image("final", "B.fade" + str(i), model.generator.generate(s * final2).numpy()[0])
                for i in range(10):
                        dataset.save_image("final", "A.noise" + str(i), model.generator.generate(final1).numpy()[0])
                        dataset.save_image("final", "B.noise" + str(i), model.generator.generate(final2).numpy()[0])
                break
        else:
                print("Adding resolution: ", model.discriminator.resolution * 2, "x", model.discriminator.resolution * 2, sep="")
                model.add_resolution(model.generator, model.discriminator)
                gan = tf.keras.Model(inputs=model.generator.inputs + model.generator.ignored_inputs, outputs=model.discriminator.model(model.generator.model.output))
                learning_rate *= learning_decay
                iterations = 0
print("Done training.")
