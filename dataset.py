import glob
import tensorflow as tf
import os


# given the image path and return the paired images (lidar,radar)
def dataloader(image_path_lidar, image_path_radar, batch_size, buffer_size):

    def load_and_preprocess(image_path):
        image = tf.io.read_file(image_path)                  # read image as uint(0~255) tensor
        image = tf.image.decode_png(image)                   # decode image to [h,w,c]
        image = tf.cast(image, tf.float32)                   # convert images to float32 tensor
        if tf.random.uniform(()) > 0.5:                      # random flip
            image = tf.image.flip_left_right(image)
        image = (image / 127.5) - 1                           # normalize image to [-1,1]

        return image

    # get the list which contains path of each image
    all_lidar_path = glob.glob(os.path.join(image_path_lidar, '*.png'))
    all_radar_path = glob.glob(os.path.join(image_path_radar, '*.png'))

    # split data into training and test set
    # num_data = len(all_lidar_path)
    train_lidar_path = all_lidar_path[:1608]
    test_lidar_path = all_lidar_path[1608:]

    train_radar_path = all_radar_path[:1608]
    test_radar_path = all_radar_path[1608:]

    # build dataset
    path_ds_lidar_train = tf.data.Dataset.from_tensor_slices(train_lidar_path)
    path_ds_lidar_test = tf.data.Dataset.from_tensor_slices(test_lidar_path)
    path_ds_radar_train = tf.data.Dataset.from_tensor_slices(train_radar_path)
    path_ds_radar_test = tf.data.Dataset.from_tensor_slices(test_radar_path)

    auto_tune = tf.data.experimental.AUTOTUNE
    ds_lidar_train = path_ds_lidar_train.map(load_and_preprocess, num_parallel_calls=auto_tune)
    ds_lidar_test = path_ds_lidar_test.map(load_and_preprocess, num_parallel_calls=auto_tune)
    ds_radar_train = path_ds_radar_train.map(load_and_preprocess, num_parallel_calls=auto_tune)
    ds_radar_test = path_ds_radar_test.map(load_and_preprocess, num_parallel_calls=auto_tune)

    # build paired data of lidar and radar
    ds_lidar_radar_train = tf.data.Dataset.zip((ds_lidar_train, ds_radar_train))
    ds_lidar_radar_test = tf.data.Dataset.zip((ds_lidar_test, ds_radar_test))

    ds_train = ds_lidar_radar_train.shuffle(buffer_size).batch(batch_size)
    ds_test = ds_lidar_radar_test.shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test