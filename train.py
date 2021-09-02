import netwoks
import loss
import tensorflow as tf

genetator = netwoks.LocalEnhancer
discriminator = netwoks.MultiscaleDiscriminator
optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_G, beta_1=arg.beta1, beta_2=0.999)
optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_D, beta_1=arg.beta1, beta_2=0.999)

# define the training loop
def train(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generator_loss, discriminator_loss, _ = loss.NetworkLoss.forward(input_image,
                                                                         target_image, genetator, discriminator)
    generator_gradients = gen_tape.gradient(generator_loss, genetator.trainable_variables)
    discriminator_gradients = gen_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    optimizer_G.apply_gradients(generator_gradients, genetator.trainable_variables)
    optimizer_D.apply_gradients(discriminator_gradients, discriminator.trainable_variables)