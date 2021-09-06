import netwoks
import loss
import tensorflow as tf
import utils
import os
import time
import datetime

# instantiate generator and discriminator
Generator = netwoks.LocalEnhancer()
Discriminator = netwoks.MultiscaleDiscriminator()

# define scheduler for scheduling learning rate
lr_scheduler_g = utils.lr_decay(args.initial_learning_rate, args.epoch, args.epoch_decay)
lr_scheduler_d = utils.lr_decay(args.initial_learning_rate, args.epoch, args.epoch_decay)

# define optimizer of generator and discriminator
optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_g, beta_1=arg.beta1, beta_2=0.999)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_d, beta_1=arg.beta1, beta_2=0.999)

# define a checkpoint-saver
checkpoint_dir = './training_checkpoints_L2R_V2'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint = tf.train.Checkpoint(generator=Generator, discriminator=Discriminator,
                                 generator_optimizer=optimizer_g,
                                 discriminator_optimizer=optimizer_d)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, checkpoint_name='L2R_ckpt')

# create s summary writer for logging the losses
log_dir = 'logs_v2/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
current_time = datetime.datetime.now().strftime('%Y%M%D-%H%M%S')
writer_path = log_dir + 'train/' + current_time
summary_writer = tf.summary.create_file_writer(writer_path)

# define the training loop
def train_step(input_image, target_image):

    # get the loss of generator and discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generator_loss, discriminator_loss, fake_img = loss.network_loss(input_image,
                                                                         target_image, Generator, Discriminator)
    # calculate the gradient of generator and discriminator
    generator_gradients = gen_tape.gradient(generator_loss, Generator.trainable_variables)
    discriminator_gradients = dis_tape.gradient(discriminator_loss, Discriminator.trainable_variables)

    # optimize the parameters of generator and discriminator
    optimizer_g.apply_gradients(zip(generator_gradients, Generator.trainable_variables))
    optimizer_d.apply_gradients(zip(discriminator_gradients, Discriminator.trainable_variables))

def train(train_data, test_data, epochs):
    test_input, target_image = next(iter(test_data.take(1)))

    for epoch in range(epochs):
        start = time.time()

        for input_image, target in train_data:
            train_step(input_image, target,)






