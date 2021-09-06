import tensorflow as tf


def network_loss(input_img, target_img, generator, discriminator):

    lambda_fm = 10
    loss_object = tf.losses.BinaryCrossentropy(from_logits=True)

    # define basic gan loss
    def generative_loss( discriminator_preds, real_image):   # real_image = True ???
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
                fm_loss = tf.keras.losses.MeanAbsoluteError(real_feature.tf.stop_gradient(), fake_feature)   # real_feature.stop_gradient ????
        fm_loss = fm_loss / 3
        return fm_loss

    # compute generator loss and discriminator loss

    fake_img = generator(input_img)
    fake_img_stop = tf.stop_gradient(fake_img)
    target_img_stop = tf.stop_gradient(target_img)
    fake_outputs_g = discriminator(tf.concat(input_img, fake_img))
    dis_fake_outputs = discriminator(tf.concat(input_img, fake_img_stop, concat_dim=-1)) # fake_img.tf.stop_gradient??????
    dis_real_outputs = discriminator(tf.concat(input_img, target_img_stop, concat_dim=-1))  # target_img.tf.stop_gradient??????

    generator_loss = generative_loss(fake_outputs_g, True) + lambda_fm * feature_matching_loss(dis_real_outputs, dis_fake_outputs) # 0.1 or 10???
    discriminator_loss = generative_loss(dis_real_outputs, True) + generative_loss(dis_fake_outputs, False)

    return generator_loss, discriminator_loss, fake_img