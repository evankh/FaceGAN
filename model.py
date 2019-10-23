import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

input_size = 64
latent_size = 64

class NoiseLayer(tf.keras.layers.Layer):
        """Adds random zero-centered Gaussian noise vector with the
        given standard deviation to the input, scaled by a learned factor.
        """
        def __init__(self, stddev, **kwargs):
                super(NoiseLayer, self).__init__(**kwargs)
                if not isinstance(stddev, float):
                        raise TypeError("Invalid argument for `stddev` - should be a float.")
                self.stddev = stddev
        def build(self, input_shape):
                self.scale = self.add_weight(shape=(1,),
                                              initializer=tf.keras.initializers.zeros,
                                              name="scaling_factor")
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                return inputs + np.random.normal(0.0, self.stddev, inputs.shape[1:]) * self.scale

class ConstantLayer(tf.keras.layers.Layer):
        """Creates a set of learned constants to use as initial input.
        """
        def __init__(self, **kwargs):
                super(ConstantLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                self.constant = self.add_weight(shape=input_shape[1:],
                                                initializer=tf.keras.initializers.zeros,
                                                name="constant")
                super(ConstantLayer, self).build(input_shape)
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                return inputs - inputs + self.constant

class AdaptiveInstanceNormalization(tf.keras.layers.Layer):
        """Does the magic AdaIN transformation.
        """
        def __init__(self, latent_code, **kwargs):
                super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
                self.latent_code = latent_code
        def build(self, input_shape):
                # Learns a mapping from the latent space to per-channel mean and Stddev
                self.stddev = self.add_weight(shape=self.latent_code.output.shape[1:] + input_shape[1:],
                                              initializer=tf.keras.initializers.ones,
                                              name="stddev")
                self.mean = self.add_weight(shape=self.latent_code.output.shape[1:] + input_shape[1:],
                                            initializer=tf.keras.initializers.zeros,
                                            name="mean")
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                # Aligns Mean and StdDev of input to learned Mean and StdDev of style
                normalized = (inputs - K.mean(inputs)) / K.std(inputs)
                # I wonder if this computation can / is cached, since much of it doesn't depend on the inputs
                return K.batch_dot(self.latent_code.output, K.expand_dims(self.stddev,0), axes=1) * normalized + K.batch_dot(self.latent_code.output, K.expand_dims(self.mean,0), axes=1)
        # This isn't ever getting any new latent codes. Once they're in, they're in for good.
        # Is there a way to make a layer take multiple inputs without using the functional model?        

class Generator:
        """Keeps all necessary information for the generator in one place.
        """
        def __init__(self, mapping, synthesis):
                self.mapping = mapping
                self.synthesis = synthesis
        # To-do: make the training from the synthesis propagate back into the mapping network
        # (not sure if that can be done here, or must be done in the training code)

class Discriminator:
        """Keeps all necessary information for the discriminator in one place.
        """
        def __init__(self, classifier):
                self.classifier = classifier
                self.resolution = 4

# Mapping Network
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
synthesis = tf.keras.Sequential()
synthesis.add(tf.keras.layers.Input(shape=(4, 4, latent_size))) # Anything put here will be ignored by the next layer, but this is the easiest way to give the model its needed input_shape
synthesis.add(ConstantLayer())
synthesis.add(tf.keras.layers.Flatten())
synthesis.add(tf.keras.layers.Dense(48))
synthesis.add(tf.keras.layers.Reshape((4, 4, 3)))
synthesis.add(NoiseLayer(1.0))
synthesis.add(AdaptiveInstanceNormalization(mapping))
synthesis.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
synthesis.add(NoiseLayer(1.0))
synthesis.add(AdaptiveInstanceNormalization(mapping))
synthesis.compile(loss="mean_squared_error", optimizer="adam")

generator = Generator(mapping, synthesis)
#generator.compile(loss="mean_squared_error", optimizer="adam")

# Discriminator Network
classifier = tf.keras.Sequential()
classifier.add(tf.keras.layers.Input((4, 4, 3)))
classifier.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(2, activation="relu"))
classifier.add(tf.keras.layers.Activation(tf.keras.activations.softmax))
classifier.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

discriminator = Discriminator(classifier)
#discriminator.compile(loss="mean_squared_error", optimizer="adam")

def add_resolution(generator, discriminaor):
        generator.synthesis.add(tf.keras.layers.UpSampling2D())
        generator.synthesis.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
        generator.synthesis.add(NoiseLayer(1.0))
        generator.synthesis.add(AdaptiveInstanceNormalization(generator.mapping))
        generator.synthesis.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
        generator.synthesis.add(NoiseLayer(1.0))
        generator.synthesis.add(AdaptiveInstanceNormalization(generator.mapping))
        # Note: could update generator.mapping with a new input vector at various resolutions
        generator.synthesis.compile(loss="mean_squared_error", optimizer="adam")
        # To-do: smooth fade-in of the new layer as decribed in [5]
        discriminator.resolution *= 2
        discriminator.classifier = tf.keras.Sequential([tf.keras.layers.Input(shape=(discriminator.resolution, discriminator.resolution, 3)),
                                                        tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"),
                                                        tf.keras.layers.AveragePooling2D()] + discriminator.classifier.layers)
