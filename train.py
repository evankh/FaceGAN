import tensorflow as tf
import PIL
import numpy as np

def images_to_vectors(images):
    return images.reshape(images.size(0), 49152)

def vectors_to_images(vectors):
    return vectors.reshape(vectors.size(0), 3, resolution, resolution)

DATA_FOLDER = './00000'
NOISE_SIZE = 100
BATCH_SIZE = 100

num_test_samples = 16
test_noise = noise(num_test_samples, NOISE_SIZE)

num_epochs = 200

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Iterate through epochs
for epoch in range(num_epochs):
    for n_batch, (batch,_) in enumerate(data_loader):
        
        # 1. Train Discriminator
        X_batch = images_to_vectors(batch.permute(0, 2, 3, 1).numpy())
        feed_dict = {X: X_batch, Z: noise(BATCH_SIZE, NOISE_SIZE)}
        _, d_error, d_pred_real, d_pred_fake = session.run(
            [D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict
        )

        # 2. Train Generator
        feed_dict = {Z: noise(BATCH_SIZE, NOISE_SIZE)}
        _, g_error = session.run(
            [G_opt, G_loss], feed_dict=feed_dict
        )

        if n_batch % 100 == 0:
            display.clear_output(True)
            # Generate images from test noise
            test_images = session.run(
                G_sample, feed_dict={Z: test_noise}
            )
            test_images = vectors_to_images(test_images)
            test_images = test_images.data
            print(test_images)
            npimages = test_images.numpy()
            np.squeeze(npimages)
            npimages = npimages[0,:,:]
            img = plt.imshow(npimages[0,:,:])
            plt.savefig("out.png")
