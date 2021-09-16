import netwoks
import loss
import tensorflow as tf
import utils
import os
import time
import datetime
from dataset import dataloader


# define the training loop
@tf.function
def train_step(input_image, target_image):

    # get the loss of generator and discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generator_loss, discriminator_loss, fm_loss, gan_loss_g = loss.network_loss(input_image,
                                                                         target_image, Generator, Discriminator)
    # calculate the gradient of generator and discriminator
    generator_gradients = gen_tape.gradient(generator_loss, Generator.trainable_variables)
    discriminator_gradients = dis_tape.gradient(discriminator_loss, Discriminator.trainable_variables)

    # optimize the parameters of generator and discriminator
    optimizer_g.apply_gradients(zip(generator_gradients, Generator.trainable_variables))
    optimizer_d.apply_gradients(zip(discriminator_gradients, Discriminator.trainable_variables))

    # accumulate loss across a epoch
    train_loss_generator(generator_loss)
    train_loss_feature_matching(fm_loss)
    train_loss_gan_g(gan_loss_g)
    train_loss_discriminator(discriminator_loss)


def train(train_data, test_data, epochs):
    test_input, target_image = next(iter(test_data.take(1)))

    for epoch in range(epochs):
        start = time.time()

        for input_image, target in train_data:
            train_step(input_image, target)

            # log loss
            with summary_writer.as_default():
                tf.summary.scalar('generator_loss', train_loss_generator.result(), step=epoch)
                tf.summary.scalar('feature_matching_loss', train_loss_feature_matching.result(), step=epoch)
                tf.summary.scalar('gan_loss_g', train_loss_gan_g.result(), step=epoch)
                tf.summary.scalar('discriminator_loss', train_loss_discriminator.result(), step=epoch)

        template = 'epoch: {}, generator_loss: {}, feature_matching_loss: {}, gan_loss_g: {}, discriminator_loss: {}'
        print(template.format(epoch+1, train_loss_generator.result(), train_loss_feature_matching.result(),
                              train_loss_gan_g.result(), train_loss_discriminator.result()))

        # reset metrics every epoch
        train_loss_generator.reset_states()
        train_loss_feature_matching.reset_states()
        train_loss_gan_g.reset_states()
        train_loss_discriminator.reset_states()

        # save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_manager.save()
            print('saving checkpoint for epoch {} at {}'.format(epoch+1, checkpoint_prefix))

        print('time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
        print('- * 40')

        # evaluate the trained generator
        if epoch == 199:
            utils.generate_image(Generator, test_input, target_image, epoch)

if __name__ == '__main__':
    args_parser = utils.get_args_parser()
    args = args_parser.parse_args()

    # ensure reproducibility
    #tf.random.set_seed(666)

    # get the dictionary of configuration parameters
    config = utils.get_config(args.config)

    IMAGE_PATH_LIDAR = config['image_path_lidar']
    IMAGE_PATH_RADAR = config['image_path_radar']
    BATCH_SIZE = config['batch_size']
    BUFFER_SIZE = config['buffer_size']
    IMG_WIDTH = config['img_width']
    IMG_HEIGHT = config['img_height']
    EPOCHS = config['epoch']
    EPOCH_DECAY = config['epoch_decay']
    INITIAL_LEARNING_RATE = config['initial_learning_rate']
    BETA_1 = config['beta_1']

    # instantiate generator and discriminator
    Generator = netwoks.LocalEnhancer()
    Discriminator = netwoks.MultiscaleDiscriminator()

    # define scheduler for scheduling learning rate
    lr_scheduler_g = utils.lr_decay(INITIAL_LEARNING_RATE, EPOCHS, EPOCH_DECAY)
    lr_scheduler_d = utils.lr_decay(INITIAL_LEARNING_RATE, EPOCHS, EPOCH_DECAY)

    # define optimizer of generator and discriminator
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_g, beta_1= BETA_1, beta_2=0.999)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_d, beta_1= BETA_1, beta_2=0.999)

    # define a checkpoint-saver
    checkpoint_dir = './training_checkpoints_L2R_V2'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(generator=Generator, discriminator=Discriminator,
                                     generator_optimizer=optimizer_g,
                                     discriminator_optimizer=optimizer_d)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, checkpoint_name='L2R_ckpt', max_to_keep=5)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'L2R_ckpt')

    # create s summary writer for logging the losses
    log_dir = 'logs_v2/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.datetime.now().strftime('%Y%M%D-%H%M%S')
    writer_path = log_dir + 'train/' + current_time
    summary_writer = tf.summary.create_file_writer(writer_path)

    # define metrics to accumulate loss and calculate mean value
    train_loss_generator = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
    train_loss_feature_matching = tf.keras.metrics.Mean('feature_matching_loss', dtype=tf.float32)
    train_loss_gan_g = tf.keras.metrics.Mean('gan_loss_g', dtype=tf.float32)
    train_loss_discriminator = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)


    # start training
    DS_TRAIN, DS_TEST = dataloader(IMAGE_PATH_LIDAR, IMAGE_PATH_RADAR, BATCH_SIZE, BUFFER_SIZE)
    train(DS_TRAIN, DS_TEST, EPOCHS)



