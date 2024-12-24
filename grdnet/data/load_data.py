import os
import time
import multiprocessing
import random
from pathlib import Path

import tensorflow as tf

import numpy as np

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

_RESIZE_METHODS = {
    'bilinear'     : tf.image.ResizeMethod.BILINEAR,
    'nearest'      : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic'      : tf.image.ResizeMethod.BICUBIC,
    'area'         : tf.image.ResizeMethod.AREA,
    'lanczos3'     : tf.image.ResizeMethod.LANCZOS3,
    'lanczos5'     : tf.image.ResizeMethod.LANCZOS5,
    'gaussian'     : tf.image.ResizeMethod.GAUSSIAN,
    'mitchellcubic': tf.image.ResizeMethod.MITCHELLCUBIC
}

def get_interpolation(interpolation):
    interpolation = interpolation.lower()
    if interpolation not in _RESIZE_METHODS:
        raise NotImplementedError(
                'Value not recognized for `interpolation`: {}. Supported values '
                'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
    return _RESIZE_METHODS[interpolation]

def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname

def index_subdirectory(directory, class_indices, follow_links, formats):
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
                dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)
    return filenames, labels

def index_directory(directory,
            labels,
            formats=ALLOWLIST_FORMATS,
            class_names=None,
            shuffle=True,
            seed=None,
            follow_links=False):

    if labels is None:
        # in the no-label case, index from the parent directory down.
        subdirs = ['']
        class_names = subdirs
    else:
        subdirs = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                subdirs.append(subdir)
        if not class_names:
            class_names = subdirs
        else:
            if set(class_names) != set(subdirs):
                raise ValueError(
                        'The `class_names` passed did not match the '
                        'names of the subdirectories of the target directory. '
                        'Expected: %s, but received: %s' %
                        (subdirs, class_names))
    class_indices = dict(zip(class_names, range(len(class_names))))

    # Build an index of the files
    # in the different class subfolders.
    pool = multiprocessing.pool.ThreadPool()
    results = []
    filenames = []

    for dirpath in (os.path.join(directory, subdir) for subdir in subdirs):
        results.append(
                pool.apply_async(index_subdirectory,
                                                 (dirpath, class_indices, follow_links, formats)))
    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res.get()
        labels_list.append(partial_labels)
        filenames += partial_filenames
    if labels not in ('inferred', None):
        if len(labels) != len(filenames):
            raise ValueError('Expected the lengths of `labels` to match the number '
                                             'of files in the target directory. len(labels) is %s '
                                             'while we found %s files in %s.' % (
                                                     len(labels), len(filenames), directory))
    else:
        i = 0
        labels = np.zeros((len(filenames),), dtype='int32')
        for partial_labels in labels_list:
            labels[i:i + len(partial_labels)] = partial_labels
            i += len(partial_labels)

    if labels is None:
        print('Found %d files.' % (len(filenames),))
    else:
        print('Found %d files belonging to %d classes.' %
                    (len(filenames), len(class_names)))
    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, fname) for fname in filenames]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = int(time.time())
            seed = np.random.randint(seed)
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)
    return file_paths, labels, class_names

def check_validation_split_arg(validation_split, subset, shuffle, seed):
    """Raise errors in case of invalid argument values.
    Args:
        validation_split: float between 0 and 1, fraction of data to reserve for
            validation.
        subset: One of "training" or "validation". Only used if `validation_split`
            is set.
        shuffle: Whether to shuffle the data. Either True or False.
        seed: random seed for shuffling and transformations.
    """
    if validation_split and not 0 < validation_split < 1:
        raise ValueError(
                '`validation_split` must be between 0 and 1, received: %s' %
                (validation_split,))
    if (validation_split or subset) and not (validation_split and subset):
        raise ValueError(
                'If `subset` is set, `validation_split` must be set, and inversely.')
    if subset not in ('training', 'validation', 'both', None):
        raise ValueError('`subset` must be either "training" '
                                         'or "validation", received: %s' % (subset,))
    if validation_split and shuffle and seed is None:
        raise ValueError(
                'If using `validation_split` and shuffling the data, you must provide '
                'a `seed` argument, to make sure that there is no overlap between the '
                'training and validation subset.')

def get_training_and_validation_split(samples, labels, validation_split):
    if not validation_split:
        return samples, labels

    num_val_samples = int(validation_split * len(samples))

    print('Using %d files for training.' % (len(samples) - num_val_samples,))
    samples_t = samples[:-num_val_samples]
    labels_t = labels[:-num_val_samples]

    print('Using %d files for validation.' % (num_val_samples,))
    samples_v = samples[-num_val_samples:]
    labels_v = labels[-num_val_samples:]

    return samples_t, labels_t, samples_v, labels_v

def get_training_or_validation_split(samples, labels, validation_split, subset):
    """Potentially restict samples & labels to a training or validation split.
    Args:
        samples: List of elements.
        labels: List of corresponding labels.
        validation_split: Float, fraction of data to reserve for validation.
        subset: Subset of the data to return.
            Either "training", "validation", or None. If None, we return all of the
            data.
    Returns:
        tuple (samples, labels), potentially restricted to the specified subset.
    """
    if not validation_split:
        return samples, labels

    num_val_samples = int(validation_split * len(samples))
    if subset == 'training':
        print('Using %d files for training.' % (len(samples) - num_val_samples,))
        samples = samples[:-num_val_samples]
        labels = labels[:-num_val_samples]
    elif subset == 'validation':
        print('Using %d files for validation.' % (num_val_samples,))
        samples = samples[-num_val_samples:]
        labels = labels[-num_val_samples:]
    else:
        raise ValueError('`subset` must be either "training" '
                                         'or "validation", received: %s' % (subset,))
    return samples, labels

def labels_to_dataset(labels, label_mode, num_classes):
    """Create a tf.data.Dataset from the list/tuple of labels.
    Args:
        labels: list/tuple of labels to be converted into a tf.data.Dataset.
        label_mode:
        - 'binary' indicates that the labels (there can be only 2) are encoded as
            `float32` scalars with values 0 or 1 (e.g. for `binary_crossentropy`).
        - 'categorical' means that the labels are mapped into a categorical vector.
            (e.g. for `categorical_crossentropy` loss).
        num_classes: number of classes of labels.
    Returns:
        A `Dataset` instance.
    """
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    if label_mode == 'binary':
        label_ds = label_ds.map(
                lambda x: tf.expand_dims(tf.cast(x, 'float32'), axis=-1),
                num_parallel_calls=tf.data.AUTOTUNE)
    elif label_mode == 'categorical':
        label_ds = label_ds.map(lambda x: tf.one_hot(x, num_classes),
                                                        num_parallel_calls=tf.data.AUTOTUNE)
    return label_ds

def load_image(path, image_size, num_channels, interpolation, mask=False, mask_type='mask'):
    """Load an image from a path and resize it."""

    gray = False
    if (num_channels == 1):
            gray = True
            num_channels = 3

    if (mask):
        if (path == ''):
            img = tf.zeros([image_size[0], image_size[1], 1]) if mask_type == 'mask' else tf.ones([image_size[0], image_size[1], 1]) * 255.0
            img = tf.cast(img, dtype=tf.int32)
        else:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
            img.set_shape((image_size[0], image_size[1], 1))
            img = tf.cast(img, dtype=tf.int32)
    else:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)

        if (gray == True):
                img = tf.image.rgb_to_grayscale(img)
                num_channels = 1

        img = tf.image.resize(img, image_size, method=interpolation, antialias=True)
        img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def paths_and_labels_to_dataset(image_paths,
        image_size,
        num_channels,
        labels,
        label_mode,
        num_classes,
        interpolation,
        class_names,
        load_masks,
        mask_type,
        mask_dir,
        mask_ext):
    """Constructs a dataset of images and labels."""
    # TODO(fchollet): consider making num_parallel_calls settable
    class_names = tf.convert_to_tensor(class_names)
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    if (load_masks):
        def gen_mask_path(path):
            if os.path.exists(path):
                # Convert the original path to a Path object
                path_obj = Path(path)
                canonical_path = path_obj.resolve()

                # Get the relative path after 'test' directory (adjust this depending on the base directory)
                relative_path = canonical_path.relative_to(canonical_path.parents[1])

                # Extract filename without extension and the extension itself
                filename_without_ext = canonical_path.stem
                extension = canonical_path.suffix

                # Reconstruct the new mask path, keeping the subdirectory structure after 'test'
                if mask_ext is None or mask_ext == '':
                    new_mask_path = Path(mask_dir) / relative_path.parent / f"{filename_without_ext}{extension}"
                else:
                    new_mask_path = Path(mask_dir) / relative_path.parent / f"{filename_without_ext}_{mask_ext}{extension}"

                # Return the new mask path if it exists, else return an empty string
                if os.path.exists(new_mask_path):
                    return str(new_mask_path)
                else:
                    return '' 
            else:
                return ''
        mask_paths = list(map(gen_mask_path, image_paths))
        path_masks_ds = tf.data.Dataset.from_tensor_slices(mask_paths)

    args = (image_size, num_channels, interpolation)
    img_ds_tmp = path_ds.map(
            lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE)

    args = (image_size, num_channels, interpolation, True, mask_type)
    if (load_masks):
        mask_ds_tmp = path_masks_ds.map(
                lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE)

    if label_mode:
        label_ds = labels_to_dataset(labels, label_mode, num_classes)
        img_ds = tf.data.Dataset.zip((img_ds_tmp, label_ds))
    indices = tf.data.Dataset.from_tensor_slices(tf.range(img_ds_tmp.cardinality()))
    classes_ds = label_ds.map(lambda y: class_names[y])

    if (load_masks):
        img_path_ds = tf.data.Dataset.zip((img_ds_tmp, label_ds, classes_ds, indices, path_ds, mask_ds_tmp))
    else:
        img_path_ds = tf.data.Dataset.zip((img_ds_tmp, label_ds, classes_ds, indices, path_ds))
    return img_ds, img_path_ds

def image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        reshuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        load_masks=False,
        mask_type='mask',
        mask_dir='MASK',
        mask_ext='png',
        samples=None):
    """Generates a `tf.data.Dataset` from image files in a directory.
    If your directory structure is:
    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```
    Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of images from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
    Supported image formats: jpeg, png, bmp, gif.
    Animated gifs are truncated to the first frame.
    Args:
        directory: Directory where the data is located.
                If `labels` is "inferred", it should contain
                subdirectories, each containing images for a class.
                Otherwise, the directory structure is ignored.
        labels: Either "inferred"
                (labels are generated from the directory structure),
                None (no labels),
                or a list/tuple of integer labels of the same size as the number of
                image files found in the directory. Labels should be sorted according
                to the alphanumeric order of the image file paths
                (obtained via `os.walk(directory)` in Python).
        label_mode:
                - 'int': means that the labels are encoded as integers
                        (e.g. for `sparse_categorical_crossentropy` loss).
                - 'categorical' means that the labels are
                        encoded as a categorical vector
                        (e.g. for `categorical_crossentropy` loss).
                - 'binary' means that the labels (there can be only 2)
                        are encoded as `float32` scalars with values 0 or 1
                        (e.g. for `binary_crossentropy`).
                - None (no labels).
        class_names: Only valid if "labels" is "inferred". This is the explicit
                list of class names (must match names of subdirectories). Used
                to control the order of the classes
                (otherwise alphanumerical order is used).
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
        batch_size: Size of the batches of data. Default: 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        image_size: Size to resize images to after they are read from disk.
                Defaults to `(256, 256)`.
                Since the pipeline processes batches of images that must all have
                the same size, this must be provided.
        shuffle: Whether to shuffle the data. Default: True.
                If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
                fraction of data to reserve for validation.
        subset: One of "training" or "validation".
                Only used if `validation_split` is set.
        interpolation: String, the interpolation method used when resizing images.
            Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
            `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
                Defaults to False.
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the target
            aspect ratio, the output image will be cropped so as to return the largest
            possible window in the image (of size `image_size`) that matches
            the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
            aspect ratio may not be preserved.
        **kwargs: Legacy keyword arguments.
    Returns:
        A `tf.data.Dataset` object.
            - If `label_mode` is None, it yields `float32` tensors of shape
                `(batch_size, image_size[0], image_size[1], num_channels)`,
                encoding images (see below for rules regarding `num_channels`).
            - Otherwise, it yields a tuple `(images, labels)`, where `images`
                has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
                and `labels` follows the format described below.
    Rules regarding labels format:
        - if `label_mode` is `int`, the labels are an `int32` tensor of shape
            `(batch_size,)`.
        - if `label_mode` is `binary`, the labels are a `float32` tensor of
            1s and 0s of shape `(batch_size, 1)`.
        - if `label_mode` is `categorial`, the labels are a `float32` tensor
            of shape `(batch_size, num_classes)`, representing a one-hot
            encoding of the class index.
    Rules regarding number of channels in the yielded images:
        - if `color_mode` is `grayscale`,
            there's 1 channel in the image tensors.
        - if `color_mode` is `rgb`,
            there are 3 channel in the image tensors.
        - if `color_mode` is `rgba`,
            there are 4 channel in the image tensors.
    """
    if labels not in ('inferred', None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                    '`labels` argument should be a list/tuple of integer labels, of '
                    'the same size as the number of image files in the target '
                    'directory. If you wish to infer the labels from the subdirectory '
                    'names in the target directory, pass `labels="inferred"`. '
                    'If you wish to get a dataset that only contains images '
                    f'(no labels), pass `labels=None`. Received: labels={labels}')
        if class_names:
            raise ValueError('You can only pass `class_names` if '
                                             f'`labels="inferred"`. Received: labels={labels}, and '
                                             f'class_names={class_names}')
    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(
                '`label_mode` argument must be one of "int", "categorical", "binary", '
                f'or None. Received: label_mode={label_mode}')
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
                '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
                f'Received: color_mode={color_mode}')
    interpolation = get_interpolation(interpolation)
    check_validation_split_arg(validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = index_directory(
            directory,
            labels,
            formats=ALLOWLIST_FORMATS,
            class_names=class_names,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links)

    if (not (samples is None)):
        indexes = list(range(len(image_paths)))
        random.seed(seed)
        indexes = random.sample(indexes, k=samples)
        indexes = list(sorted(indexes))
        image_paths = [image_paths[index] for index in indexes]
        labels = [labels[index] for index in indexes]

    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(
                f'When passing `label_mode="binary"`, there must be exactly 2 '
                f'class_names. Received: class_names={class_names}')

    image_paths_v = None
    labels_v      = None
    if (subset == 'both'):
        image_paths, labels, image_paths_v, labels_v = get_training_and_validation_split(image_paths, labels, validation_split)
    else:
        image_paths, labels = get_training_or_validation_split(image_paths, labels, validation_split, subset)
    if not image_paths:
        raise ValueError(f'No images found in directory {directory}. '
                                         f'Allowed formats: {ALLOWLIST_FORMATS}')

    dataset, dataset_path = paths_and_labels_to_dataset(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=len(class_names),
            interpolation=interpolation,
            class_names=class_names,
            load_masks=load_masks,
            mask_type=mask_type,
            mask_dir=mask_dir,
            mask_ext=mask_ext)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset_path = dataset_path.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=reshuffle)
            dataset_path = dataset_path.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=reshuffle)
        dataset = dataset.batch(batch_size)
        dataset_path = dataset_path.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=reshuffle)
            dataset_path = dataset_path.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=reshuffle)

    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    dataset_path.class_names = class_names
    # Include file paths for images as attribute.
    dataset.file_paths = image_paths
    dataset_path.file_paths = image_paths

    if (subset == 'both'):
        dataset_v, dataset_path_v = paths_and_labels_to_dataset(
                image_paths=image_paths_v,
                image_size=image_size,
                num_channels=num_channels,
                labels=labels_v,
                label_mode=label_mode,
                num_classes=len(class_names),
                interpolation=interpolation,
                class_names=class_names,
                load_masks=load_masks,
                mask_type=mask_type,
                mask_dir=mask_dir,
                mask_ext=mask_ext)
        dataset_v = dataset_v.prefetch(tf.data.AUTOTUNE)
        dataset_path_v = dataset_path_v.prefetch(tf.data.AUTOTUNE)
        if batch_size is not None:
            if shuffle:
                # Shuffle locally at each iteration
                dataset_v = dataset_v.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=reshuffle)
                dataset_path_v = dataset_path_v.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=reshuffle)
            dataset_v = dataset_v.batch(batch_size)
            dataset_path_v = dataset_path_v.batch(batch_size)
        else:
            if shuffle:
                dataset_v = dataset_v.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=reshuffle)
                dataset_path_v = dataset_path_v.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=reshuffle)

        # Users may need to reference `class_names`.
        dataset_v.class_names = class_names
        dataset_path_v.class_names = class_names
        # Include file paths for images as attribute.
        dataset_v.file_paths = image_paths
        dataset_path_v.file_paths = image_paths
        return dataset, dataset_path, dataset_v, dataset_path_v
    return dataset, dataset_path
