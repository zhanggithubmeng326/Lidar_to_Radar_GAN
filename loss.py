import tensorflow as tf

class NetworkLoss(tf.keras.Model):

    def __init__(self, lambda_fm=10):
        super(NetworkLoss, self).__init__()
        self.loss_object = tf.losses.BinaryCrossentropy(from_logits=True)  # or use MSE loss ???
        self.lambda_fm = lambda_fm

    # compute basic gan loss
    def gan_loss(self, discriminator_preds, real_image):   # real_image = True ???
        gan_loss = 0.0

        for preds in discriminator_preds:
            pred = preds[-1]

            if real_image:
                gan_loss += self.loss_object(pred, tf.ones_like(pred))
            else:
                gan_loss += self.loss_object(pred, tf.zeros_like(pred))

        return gan_loss

    # compute the feature matching loss for different layers of discriminators with 3 different scales
    def feature_matching_loss(self, dis_real_outputs, dis_fake_outputs):

        fm_loss = 0.0
        for real_features, fake_features in zip(dis_real_outputs, dis_fake_outputs):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss = tf.keras.losses.MeanAbsoluteError(real_feature.tf.stop_gradient(), fake_feature)   # real_feature.stop_gradient ????
        fm_loss = fm_loss / 3
        return fm_loss

    # compute generator loss and discriminator loss
    # def call(self, input_img, training=None, mask=None):     use call() or forward() ??????
    def forward(self, input_img, target_img, generator, discriminator):

        fake_img = generator(input_img)
        fake_img_stop = tf.stop_gradient(fake_img)
        target_img_stop = tf.stop_gradient(target_img)
        fake_outputs_g = discriminator(tf.concat(input_img, fake_img))
        dis_fake_outputs = discriminator(tf.concat(input_img, fake_img_stop, concat_dim=-1)) # fake_img.tf.stop_gradient??????
        dis_real_outputs = discriminator(tf.concat(input_img, target_img_stop, concat_dim=-1))  # target_img.tf.stop_gradient??????

        generator_loss = self.gan_loss(fake_outputs_g, True) + \
                         self.lambda_fm * self.feature_matching_loss(dis_real_outputs, dis_fake_outputs) # 0.1 or 10???

        discriminator_loss = self.gan_loss(dis_real_outputs, True) + self.gan_loss(dis_fake_outputs, False)

        return generator_loss, discriminator_loss, fake_img_stop