import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

K.set_floatx("float64")
# Model Parameters
input_size = 64                 # Dimensionality of input space
latent_size = 64                # Dimaensionality of latent space
initializer_shape = (4, 4, latent_size) # Shape of constant initialization layer
starting_resolution = 8         # Initial resolution of images
num_channels = 3                # Number of internal channels to use, to hopefully have more information to work with than just RGB
use_smaller_noise = False       # Use noise textures one resolution smaller than the current resolution, to significantly decrease the size of the network
use_mapping = False             # Use a mapping network to generate a latent code from the input code
use_second_block = False        # Use two Convolution - Noise - AdaIN blocks for each resolution

class NoiseLayer(tf.keras.layers.Layer):
        """Adds random zero-centered Gaussian noise to the input, with a learned
        standard deviation to the input, scaled by a learned factor.
        """
        def __init__(self, **kwargs):
                super(NoiseLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                self.scale = self.add_weight(shape=input_shape[1:],
                                             initializer=tf.keras.initializers.zeros,
                                             name="scaling_factor")
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                return inputs + K.random_normal(inputs.shape[1:3] + (1,), 0.0, 1.0) * K.expand_dims(self.scale, 0)

class ConstantLayer(tf.keras.layers.Layer):
        """Creates a set of learned constants to use as initial input.
        Ignores any input given to it and only returns the learned weights,
        but must be given an input layer to match the size of.
        """
        def __init__(self, **kwargs):
                super(ConstantLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                self.constant = self.add_weight(shape=input_shape[1:],
                                                initializer=tf.keras.initializers.ones,
                                                name="constant")
                super(ConstantLayer, self).build(input_shape)
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                return inputs - inputs + self.constant

class AdaptiveInstanceNormalizationLayer(tf.keras.layers.Layer):
        """Performs a learned AdaIN transformation.
        Matches the mean and standard deviation of the input to the learned
        mean and standard deviation of the dataset. Uses the input latent code
        as a "style" which controls factors in the generated image; the learned
        weights are the effect each factor of the latent code has on each channel
        of the image.
        """
        def __init__(self, **kwargs):
                super(AdaptiveInstanceNormalizationLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                # Learns a mapping from the latent space to per-channel Mean and StdDev
                assert len(input_shape) == 2
                latent_shape, layer_shape = input_shape         # Nonex64, NonexNxNxC
                self.mean = self.add_weight(shape=latent_shape[1:] + (layer_shape[3],),
                                            initializer=tf.keras.initializers.RandomNormal,
                                            name="mean")
                self.stddev = self.add_weight(shape=latent_shape[1:] + (layer_shape[3],),
                                              initializer=tf.keras.initializers.RandomNormal,
                                              name="stddev")
        def compute_output_shape(self, input_shape):
                assert len(input_shape) == 2
                return input_shape[1]
        def call(self, inputs):
                assert len(inputs) == 2
                latent_code, layer = inputs                                                     # Bx64, BxNxNxC
                old_mean = K.mean(layer, axis=[1,2], keepdims=True)                             # Bx1x1xC means
                old_stddev = K.std(layer, axis=[1,2], keepdims=True)                            # Bx1x1xC stddevs
                normalized = (layer - old_mean) / (old_stddev + K.epsilon())                    # BxNxNxC z-scores
                new_mean = K.expand_dims(K.expand_dims(K.dot(latent_code, self.mean), axis=1), axis=1)          # should be Bx1x1xC means, but dot produces BxC
                new_stddev = K.expand_dims(K.expand_dims(K.dot(latent_code, self.stddev), axis=1), axis=1)      # should be Bx1x1xC stddevs, but dot produce BxC
                return new_stddev * normalized + new_mean

class Generator:
        """Keeps all necessary information for the generator in one place.
        """
        def __init__(self, mapping, synthesis, inputs, ignored_inputs):
                self.mapping = mapping
                self.synthesis = synthesis
                self.toRGB = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding="same")
                self.inputs = inputs
                self.ignored_inputs = ignored_inputs
                self.model = tf.keras.Model(inputs=self.inputs + self.ignored_inputs, outputs=self.toRGB(self.synthesis))
                self.loss = None
        def generate(self, noise, crossover_noise=None, crossover_layer=0):
                # Inputs:
                #  - one input code for each resolution (Each layer could use a separate one, but it is currently only set up to use 2)
                #  - one ignored input fed to the constant layer
                #  - several zero inputs fed to the reduced-resolution noise layers
                # crossover_layer is the index of the first resolution to use the second input code
                if crossover_noise is None:
                        return self.model([noise for i in self.inputs] +
                                          [np.zeros((noise.shape[0],) + i.shape[1:]) for i in self.ignored_inputs])
                else:
                        assert noise.shape == crossover_noise.shape
                        return self.model([noise for i in self.inputs[:crossover_layer]] +
                                          [crossover_noise for i in self.inputs[crossover_layer:]] +
                                          [np.zeros((noise.shape[0],) + i.shape[1:]) for i in self.ignored_inputs])
        def train_on_batch(self, x, y, discriminator, crossover_x=None, crossover_layer=0):
                discriminator.model.trainable = False
                gan = tf.keras.Model(inputs=self.inputs + self.ignored_inputs, outputs=discriminator.model(self.model.output))
                gan.compile(loss="binary_crossentropy", optimizer="adam")
                if crossover_x is None:
                        self.loss = gan.train_on_batch([x for i in self.inputs] +
                                                       [np.zeros((x.shape[0],) + i.shape[1:]) for i in self.ignored_inputs], y)
                else:
                        assert x.shape == crossover_x.shape
                        self.loss = gan.train_on_batch([x for i in self.inputs[:crossover_layer]] +
                                                       [crossover_x for i in self.inputs[crossover_layer:]] +
                                                       [np.zeros((x.shape[0],) + i.shape[1:]) for i in self.ignored_inputs], y)
                discriminator.model.trainable = True
                return self.loss
        def get_loss(self):
                if self.loss:
                        return self.loss
                return float("inf")
        # Ideally, once training is finished, mapping network can be removed entirely
        # Input would be the latent code directly, for better feature separation, which means the inputs to the AdaIN layers will have to change

class Discriminator:
        """Keeps all necessary information for the discriminator in one place.
        """
        def __init__(self, classifier):
                self.classifier = classifier
                self.resolution = starting_resolution
                self.fromRGB = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1)
                self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(self.resolution, self.resolution, 3)), self.fromRGB, self.classifier])
                self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
        def classify(self, image):
                return self.classifier.predict(self.fromRGB(image))

# Mapping Network
# Takes in an input vector and outputs an intermediate latent code, intended to disentagle the input features
mapping = tf.keras.Sequential()
mapping.add(tf.keras.layers.Input(shape=(input_size,)))
mapping.add(tf.keras.layers.Dense(input_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(input_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(input_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(input_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(latent_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(latent_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(latent_size, activation="relu"))
mapping.add(tf.keras.layers.Dense(latent_size, activation="relu"))
mapping.compile(loss="mean_squared_error", optimizer="adam")

# Synthesis Network
# Has to be written using the Functional API so that the AdaIN layers can take multiple inputs.
latent_input = tf.keras.layers.Input(shape=(input_size,))
ignore_input = tf.keras.layers.Input(shape=initializer_shape)
synthesis = ConstantLayer()(ignore_input)
synthesis = tf.keras.layers.Flatten()(synthesis)
synthesis = tf.keras.layers.Dense(starting_resolution * starting_resolution * num_channels)(synthesis)
synthesis = tf.keras.layers.Reshape((starting_resolution, starting_resolution, num_channels))(synthesis)
synthesis = NoiseLayer()(synthesis)
if use_mapping:
        synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])
else:
        synthesis = AdaptiveInstanceNormalizationLayer()([latent_input, synthesis])
synthesis = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding="same")(synthesis)
synthesis = NoiseLayer()(synthesis)
if use_mapping:
        synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])
else:
        synthesis = AdaptiveInstanceNormalizationLayer()([latent_input, synthesis])

generator = Generator(mapping, synthesis, [latent_input,], [ignore_input,])

# Discriminator Network
classifier = tf.keras.Sequential()
classifier.add(tf.keras.layers.Input(shape=(starting_resolution, starting_resolution, num_channels)))
classifier.add(tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding="same"))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(8, activation="relu"))
classifier.add(tf.keras.layers.Dense(1, activation="sigmoid"))
classifier.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])

discriminator = Discriminator(classifier)

def add_resolution(generator, discriminaor):
        generator.synthesis = tf.keras.layers.UpSampling2D()(generator.synthesis)
        generator.synthesis = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding="same")(generator.synthesis)
        # Noise layers after the first are 1 resolution smaller than the layer they are being applied to, to dastically reduce the size of the network without sacrificing much quality
        shape = (discriminator.resolution, discriminator.resolution, num_channels)
        generator.inputs.append(tf.keras.layers.Input(shape=(input_size,)))     # Add a new input for each resolution, used for crossover training
        if use_smaller_noise:
                generator.ignored_inputs.append(tf.keras.layers.Input(shape=shape))     # Input = all zeros
                noise = NoiseLayer()(generator.ignored_inputs[-1])
                noise = tf.keras.layers.UpSampling2D()(noise)
                generator.synthesis = tf.keras.layers.Add()(inputs=[generator.synthesis, noise])
        else:
                generator.synthesis = NoiseLayer()(generator.synthesis)
        if use_mapping:
                generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.mapping(generator.inputs[-1]), generator.synthesis])
        else:
                generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.inputs[-1], generator.synthesis])
        if use_second_block:
                generator.synthesis = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding="same")(generator.synthesis)
                if use_smaller_noise:
                        noise = NoiseLayer()(generator.ignored_inputs[-1])      # Reuse the same input layer, since it's being ignored anyway
                        noise = tf.keras.layers.UpSampling2D()(noise)
                        generator.synthesis = tf.keras.layers.Add()(inputs=[generator.synthesis, noise])
                else:
                        generator.synthesis = NoiseLayer()(generator.synthesis)
                if use_mapping:
                        generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.mapping(generator.inputs[-1]), generator.synthesis])
                else:
                        generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.inputs[-1], generator.synthesis])
        generator.model = tf.keras.Model(inputs=generator.inputs + generator.ignored_inputs, outputs=generator.toRGB(generator.synthesis))
        generator.model.compile(loss="binary_crossentropy", optimizer="adam")
        # To-do: smooth fade-in of the new layer as decribed in [5]
        discriminator.resolution *= 2
        classifier = tf.keras.Sequential()
        classifier.add(tf.keras.layers.Input(shape=(discriminator.resolution, discriminator.resolution, num_channels)))
        classifier.add(tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, padding="same"))
        classifier.add(tf.keras.layers.AveragePooling2D())
        classifier.add(discriminator.classifier)
        discriminator.classifier = classifier
        discriminator.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(discriminator.resolution, discriminator.resolution, 3)), discriminator.fromRGB, discriminator.classifier])
        discriminator.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
