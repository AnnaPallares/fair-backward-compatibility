import tensorflow as tf

def _parse_and_augment(img, augment):
    """Internal helper for consistent image augmentation."""
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

def make_baseline_dataset(df, batch_size, seed, preprocess_fn, image_size, augment=False):
    # Small dataset (for baseline model) of same size as batch size, used for the proper old-new model comparison and NF metric aquisition
    ds = tf.data.Dataset.from_tensor_slices((df['file'].values, df['target'].values))
    
    def _map(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = preprocess_fn(_parse_and_augment(img, augment))
        return img, label

    return (ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(buffer_size=min(len(df), 1000), seed=seed)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))

def make_fair_dataset(df, bs, seed, pre_fn, img_size, augment=False):
    # Small dataset (for new model) of same size as batch size, used for the proper old-new model comparison and NF metric aquisition
    # Pass 4 inputs: image path, true label, old prediction, and sensitive attribute
    ds = tf.data.Dataset.from_tensor_slices((
        df['file'].values, 
        df['target'].values, 
        df['y_old_pred'].values, 
        df['sensitive'].values
    ))

    def _map(p, y, y_old, s):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = pre_fn(_parse_and_augment(img, augment))
        return (img, y_old, s), y

    return (ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(buffer_size=len(df), seed=seed)
              .batch(bs)
              .prefetch(tf.data.AUTOTUNE))

def make_image_ds(files, batch_size, preprocess_fn, image_size):
    """Pure inference dataset—no shuffling, no augmentation."""
    ds = tf.data.Dataset.from_tensor_slices(files)
    def _map(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return preprocess_fn(img)
    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)