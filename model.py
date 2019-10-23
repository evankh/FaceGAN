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

class AdaptiveInstanceNormalizationLayer(tf.keras.layers.Layer):
        """Does the magic AdaIN transformation.
        """
        def __init__(self, **kwargs):
                super(AdaptiveInstanceNormalizationLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                # Learns a mapping from the latent space to per-channel Mean and StdDev
                assert len(input_shape) == 2
                latent_shape, layer_shape = input_shape
                self.stddev = self.add_weight(shape=latent_shape[1:] + layer_shape[1:],
                                              initializer=tf.keras.initializers.ones,
                                              name="stddev")
                self.mean = self.add_weight(shape=latent_shape[1:] + layer_shape[1:],
                                            initializer=tf.keras.initializers.zeros,
                                            name="mean")
        def compute_output_shape(self, input_shape):
                assert len(input_shape) == 2
                return input_shape[1]
        def call(self, inputs):
                assert len(inputs) == 2
                latent_code, layer = inputs
                # Aligns Mean and StdDev of input to learned Mean and StdDev of "style"
                normalized = (layer - K.mean(layer)) / K.std(layer)     # Is this doing it per-channel or per-layer?
                return K.batch_dot(latent_code, K.expand_dims(self.stddev,0), axes=1) * normalized + K.batch_dot(latent_code, K.expand_dims(self.mean,0), axes=1)

class Generator:
        """Keeps all necessary information for the generator in one place.
        """
        def __init__(self, mapping, synthesis, inputs):
                self.mapping = mapping
                self.synthesis = synthesis
                self.inputs = inputs
                self.model = tf.keras.Model(inputs=self.inputs, outputs=self.synthesis)
                self.model.compile(loss="mean_squared_error", optimizer="adam")
        def generate(self, noise):
                return self.model.predict([noise, np.zeros_like(noise)])
        def train_on_batch(self, x, y):
                return self.model.train_on_batch([x, np.zeros_like(x)], y)

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
# Has to be rewritten using the Functional API so that the AdaIN layers can take multiple inputs
# Although I'm quite certain there must be a way to make it work with Sequential.
# Benefit of this is that is should be training mapping at the same time
latent_input = tf.keras.layers.Input(shape=(input_size,))
ignore_input = tf.keras.layers.Input(shape=(4, 4, latent_size))
synthesis = ConstantLayer()(ignore_input)
synthesis = tf.keras.layers.Flatten()(synthesis)
synthesis = tf.keras.layers.Dense(48)(synthesis)
synthesis = tf.keras.layers.Reshape((4, 4, 3))(synthesis)
synthesis = NoiseLayer(1.0)(synthesis)
synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])
synthesis = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(synthesis)
synthesis = NoiseLayer(1.0)(synthesis)
synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])

generator = Generator(mapping, synthesis, [latent_input, ignore_input])

# Discriminator Network
classifier = tf.keras.Sequential()
classifier.add(tf.keras.layers.Input((4, 4, 3)))
classifier.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(2, activation="relu"))
classifier.add(tf.keras.layers.Activation(tf.keras.activations.softmax))
classifier.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

discriminator = Discriminator(classifier)

def add_resolution(generator, discriminaor):
        generator.synthesis = tf.keras.layers.UpSampling2D()(generator.synthesis)
        generator.synthesis = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(generator.synthesis)
        generator.synthesis = NoiseLayer(1.0)(generator.synthesis)
        generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.mapping(latent_input), generator.synthesis])
        generator.synthesis = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(generator.synthesis)
        generator.synthesis = NoiseLayer(1.0)(generator.synthesis)
        generator.synthesis = AdaptiveInstanceNormalizationLayer()([generator.mapping(latent_input), generator.synthesis])
        generator.model = tf.keras.Model(inputs=generator.inputs, outputs=generator.synthesis)
        generator.model.compile(loss="mean_squared_error", optimizer="adam")
        # Note: could update generator.mapping with a new input vector at various resolutions
        # To-do: smooth fade-in of the new layer as decribed in [5]
        discriminator.resolution *= 2
        discriminator.classifier = tf.keras.Sequential([tf.keras.layers.Input(shape=(discriminator.resolution, discriminator.resolution, 3)),
                                                        tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"),
                                                        tf.keras.layers.AveragePooling2D()] + discriminator.classifier.layers)
