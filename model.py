import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

input_size = 64
latent_size = 64
initializer_shape = (4, 4, latent_size)
starting_resolution = 4

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
                self.scale = self.add_weight(shape=input_shape[1:],
                                             initializer=tf.keras.initializers.zeros,
                                             name="scaling_factor")
                self.stddevs = self.add_weight(shape=input_shape[1:],
                                              initializer=tf.keras.initializers.zeros,
                                              name="stddev")
        def compute_output_shape(self, input_shape):
                return input_shape
        def call(self, inputs):
                return inputs + K.random_normal(inputs.shape[1:], 0.0, self.stddevs) * K.expand_dims(self.scale, 0)

class ConstantLayer(tf.keras.layers.Layer):
        """Creates a set of learned constants to use as initial input.
        Ignores any input given to it and only returns the learned weights, but must be given an input layer to match the size of.
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
        Matches the mean and standard deviation of the input to the learned mean and standard deviation of the dataset.
        Uses the input latent code as a "style" which controls factors in the generated image; the learned weights are
        the effect each factor of the latent code has on each feature of the image.
        """
        def __init__(self, **kwargs):
                super(AdaptiveInstanceNormalizationLayer, self).__init__(**kwargs)
        def build(self, input_shape):
                # Learns a mapping from the latent space to per-channel Mean and StdDev
                assert len(input_shape) == 2
                latent_shape, layer_shape = input_shape
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
                latent_code, layer = inputs
                normalized = (layer - K.mean(layer, axis=1)) / (K.std(layer, axis=1) + K.epsilon())     # NxNxC z-scores, but they do end up as nan if the inputs are 0
                new_mean = K.batch_dot(latent_code, K.expand_dims(self.mean, 0), axes=1)                # should be C means
                new_stddev = K.batch_dot(latent_code, K.expand_dims(self.stddev, 0), axes=1)            # should be C stddevs
                return new_stddev * normalized + new_mean

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
                return self.model.predict([noise, np.zeros((noise.shape[0],) + initializer_shape)])
        def train_on_batch(self, x, y):
                return self.model.train_on_batch([x, np.zeros((x.shape[0],) + initializer_shape)], y)
        # Ideally, once training is finished, mapping network can be removed entirely
        # Input would be the latent code directly, for better feature separation, which means the inputs to the AdaIN layers will have to change

class Discriminator:
        """Keeps all necessary information for the discriminator in one place.
        """
        def __init__(self, classifier):
                self.classifier = classifier
                self.resolution = starting_resolution
        def classify(self, image):
                return K.argmax(self.classifier.predict(image))

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
synthesis = tf.keras.layers.Dense(48)(synthesis)
synthesis = tf.keras.layers.Reshape((starting_resolution, starting_resolution, 3))(synthesis)   # 3 = R, G, B
synthesis = NoiseLayer(1.0)(synthesis)
synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])
synthesis = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")(synthesis)
synthesis = NoiseLayer(1.0)(synthesis)
synthesis = AdaptiveInstanceNormalizationLayer()([mapping(latent_input), synthesis])

generator = Generator(mapping, synthesis, [latent_input, ignore_input])

# Discriminator Network
classifier = tf.keras.Sequential()
classifier.add(tf.keras.layers.Input((starting_resolution, starting_resolution, 3)))    # 3 = R, G, B
classifier.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same"))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(8, activation="relu"))
classifier.add(tf.keras.layers.Dense(2, activation="relu"))
classifier.add(tf.keras.layers.Activation(tf.keras.activations.softmax))
classifier.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

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
