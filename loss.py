import tensorflow as tf


def network_loss(input_img, target_img, generator, discriminator):

    lambda_fm = 10.0
    loss_object = tf.losses.BinaryCrossentropy(from_logits=True)

    # define basic gan loss
    def generative_loss(discriminator_preds, real_image):   # real_image = True ???
        gan_loss = 0.0

        for preds in discriminator_preds:
            pred = preds[-1]

            if real_image:
                gan_loss += loss_object(pred, tf.ones_like(pred))
            else:
                gan_loss += loss_object(pred, tf.zeros_like(pred))

        return gan_loss

    # compute the feature matching loss for different layers of discriminators with 3 different scales
    def feature_matching_loss(dis_real_outputs, dis_fake_outputs):

        fm_loss = 0.0
        for real_features, fake_features in zip(dis_real_outputs, dis_fake_outputs):
            for real_feature, fake_feature in zip(real_features, fake_features):
                real_feature_stop = tf.stop_gradient(real_feature)
                fm_loss = tf.keras.losses.MeanAbsoluteError(real_feature_stop, fake_feature)   # real_feature.stop_gradient ????
        fm_loss = fm_loss / 3
        return fm_loss

    # compute generator loss and discriminator loss

    fake_img = generator(input_img, training=True)               # training=True???
    fake_img_stop = tf.stop_gradient(fake_img)                   #???
    # target_img_stop = tf.stop_gradient(target_img)               ???
    fake_outputs_g = discriminator(tf.concat([input_img, fake_img], -1))     # do we need dtype=tf.float32 here?
    dis_fake_outputs = discriminator(tf.concat([input_img, fake_img_stop], -1))    # fake_img.tf.stop_gradient??????
    dis_real_outputs = discriminator(tf.concat([input_img, target_img], -1))  # target_img.tf.stop_gradient??????

    loss_fm = feature_matching_loss(dis_real_outputs, dis_fake_outputs)
    loss_gan_g = generative_loss(fake_outputs_g, False)

    generator_loss = loss_gan_g + lambda_fm * loss_fm    #0.1 or 10???
    discriminator_loss = generative_loss(dis_real_outputs, True) * 0.5 + generative_loss(dis_fake_outputs, False) * 0.5

    return generator_loss, discriminator_loss, loss_fm, loss_gan_g