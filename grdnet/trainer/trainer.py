import os
import datetime
import random
import json
import gc
from typing import Tuple, Optional, Union, Dict, List

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

from ..util import set_seed
from ..grd_nets import build_generator, build_discriminator, build_segmentator
from ..perlin import Perlin
from ..augment import AugmentPipe
from ..data import image_dataset_from_directory
from ..aggregator import Aggregator, GroupTypeEnum, MetricEnum, ScoreEnum

class Trainer:
    def __init__(
        self,
        directory: str,
        roi_directory: str,
        test_directory: str,
        mask_directory: str,
        real_dfct_directory: str,
        real_mask_directory: str,
        patch_size: int,
        patches: Tuple[int, int],
        strides: Optional[Tuple[int, int]],
        channels: int,
        batch_size: int,
        con_loss: str ='mae',
        adv_loss: str ='cos',
        use_bn: bool = True,
        log_path: str = '/tmp/grdnet_logs',
        experiment_name: str = 'grdnet_experiment',
        generator_model_w_path: Optional[str] = None,
        discriminator_model_w_path: Optional[str] = None,
        segmentator_model_w_path: Optional[str] = None,
    ):
        self.patch_size = patch_size
        self.patches = patches
        self.strides = strides

        image_size = (
            self.patch_size + (self.patches[0] - 1) * self.strides[0],
            self.patch_size + (self.patches[1] - 1) * self.strides[1]
        )

        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.log_path = log_path
        self.experiment_name = experiment_name
        self.experiment_name_with_datetime = str(self.experiment_name) + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.logdir = self.log_path + '/' + self.experiment_name_with_datetime
        self.writer = tf.summary.create_file_writer(self.logdir)

        self.color_mode = 'rgb'
        if channels == 1:
            self.color_mode = 'grayscale'
        elif channels == 3:
            self.color_mode = 'rgb'
        elif channels == 4:
            self.color_mode = 'rgba'
            raise NotImplementedError(f'Value not recognized for `channels`: {channels}. Supported values are: 1, 3, 4')
        else:
            raise ValueError(f'Value not recognized for `channels`: {channels}. Supported values are: 1, 3, 4')

        self.seed = set_seed()

        _, train_dataset, _, validation_dataset = image_dataset_from_directory(
            directory=os.path.realpath(directory),
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            reshuffle=True,
            seed=self.seed,
            validation_split=0.05,
            subset='both',
            load_masks=True,
            mask_type='roi',
            mask_dir=os.path.realpath(roi_directory),
            mask_ext='mask',
            samples=None
        )

        size = 256
        reference_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=os.path.realpath(directory),
            labels='inferred',
            label_mode='int',
            color_mode=self.color_mode,
            batch_size=1,
            image_size=(size, size),
            shuffle=True,
            seed=self.seed,
            interpolation='bilinear',
            follow_links=True,
        )

        self.train_dataset_len = len(train_dataset)
        self.validation_dataset_len = len(validation_dataset)

        train_normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        train_dataset = train_dataset.map(lambda x, y, l, i, p, m: (train_normalization_layer(x), tf.where(train_normalization_layer(m) > 0.5, 1.0, 0.0)))

        validation_normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        validation_dataset = validation_dataset.map(lambda x, y, l, i, p, m: (validation_normalization_layer(x), tf.where(validation_normalization_layer(m) > 0.5, 1.0, 0.0)))

        reference_normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        reference_dataset = reference_dataset.map(lambda x, _: (reference_normalization_layer(x)))

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.reference_dataset = reference_dataset

        _, real_defect_dataset = image_dataset_from_directory(
            directory=os.path.realpath(real_dfct_directory),
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            reshuffle=True,
            load_masks=True,
            mask_type='mask',
            mask_dir=os.path.realpath(real_mask_directory),
            mask_ext='mask'
        )

        def extract(x, m):
            x, m = self.extract_patches(x, m)
            return x, tf.where(m > 0.5, 1.0, 0.0)

        real_dfct_normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.real_defect_dataset = real_defect_dataset.map(lambda x, y, l, i, p, m: extract(real_dfct_normalization_layer(x), real_dfct_normalization_layer(m)))
        self.real_defect_dataset_len = len(self.real_defect_dataset)

        # The mask of good product is a black image (filled by float32 0.0s)
        # We must generate a black image with the same size as the original image in order to apply the mask and to load it from the disk
        # We need to create a `mask` directory in the same directory of `train`. Both train and mask have to be in the same directory and must have two subdirectories: `good` and `bad`.
        # From the `good` subdirectory of mask, we must run the following command:
        # for i in $(find ../../test/good -iname '*.png' -type f); do echo $i; size=$(identify -format "%wx%h" "${i}"); echo ${size}; filename_with_extension=$(basename "$i"); filename="${filename_with_extension%.*}"; extension="${filename_with_extension##*.}"; convert -size ${size} xc:black "${filename}_mask.${extension}"; echo; done

        _, test_dataset = image_dataset_from_directory(
            directory=os.path.realpath(test_directory),
            color_mode=self.color_mode,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=False,  # No need to shuffle test data
            reshuffle=False,  # Obviously no need to reshuffle test data
            load_masks=True,
            mask_type='mask',
            mask_dir=os.path.realpath(mask_directory),
            mask_ext='mask'
        )

        test_normalization_layer = tf.keras.layers.Rescaling(1. / 255.)
        self.test_dataset = test_dataset.map(lambda x, y, l, i, p, m: (test_normalization_layer(x), tf.where(test_normalization_layer(m) > 0.5, 1.0, 0.0), p))
        self.test_dataset_len = len(self.test_dataset)

        self.generator_lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=self.train_dataset_len * 2,
            t_mul=2.00,
            m_mul=0.75,
            alpha=0.01
        )

        self.discriminator_lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=self.train_dataset_len * 2,
            t_mul=2.00,
            m_mul=0.75,
            alpha=0.01
        )

        self.segmentator_lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=self.train_dataset_len * 2,
            t_mul=2.00,
            m_mul=0.75,
            alpha=0.01
        )

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.generator_lr_scheduler
        )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.discriminator_lr_scheduler
        )

        self.segmentator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.segmentator_lr_scheduler
        )

        # Define the selection of the loss function
        # Losses for CONTEXTUAL LOSS
        con_loss = con_loss.strip().lower()
        if con_loss == 'mae':
            self.contextual_loss_fn = tf.keras.losses.MeanAbsoluteError()  # mean(|x_original - x_rebuilt|)
        elif con_loss == 'mse':
            self.contextual_loss_fn = tf.keras.losses.MeanSquaredError()  # mean((x_original - x_rebuilt)^2)
        elif con_loss == 'huber':
            self.contextual_loss_fn = tf.keras.losses.Huber()  # mean(0.5 * (x_original - x_rebuilt)^2) for |x_original - x_rebuilt| <= delta, mean(delta * (|x_original - x_rebuilt| - 0.5 * delta)) otherwise
        elif con_loss == 'logcosh':
            self.contextual_loss_fn = tf.keras.losses.LogCosh()  # mean(log(cosh(x_original - x_rebuilt)))
        elif con_loss == 'bce':
            self.contextual_loss_fn = tf.keras.losses.BinaryCrossentropy()  # mean(-[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)])
        elif con_loss == 'focal':
            self.contextual_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)  # mean(-[alpha * (1 - y_pred)^gamma * y_true * log(y_pred) + (1 - alpha) * y_pred^gamma * (1 - y_true) * log(1 - y_pred)])
        elif con_loss == 'ssim':
            if self.channels == 1:
                # Structural Similarity Index (SSIM) for grayscale images
                self.contextual_loss_fn = lambda y_true, y_pred: 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0), axis=None)  # mean(1 - SSIM(x_original, x_rebuilt))
            else:
                # SSIM for RGB images with three channels, averaged across channels
                self.contextual_loss_fn = lambda y_true, y_pred: 1.0 * (1 - tf.reduce_mean(
                    (
                        tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1], max_val=1.0) +
                        tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2], max_val=1.0) +
                        tf.image.ssim(y_true[..., 2:3], y_pred[..., 2:3], max_val=1.0)
                    ) / 3.0,
                    axis=None))  # mean(1 - SSIM(x_original, x_rebuilt)) across RGB channels
        elif con_loss == 'huber_ssim':
            huber = tf.keras.losses.Huber()
            if self.channels == 1:
                # Structural Similarity Index (SSIM) for grayscale images
                self.contextual_loss_fn = lambda y_true, y_pred: 1.0 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0), axis=None)) + 10.0 * huber(y_true, y_pred)  # mean(1 - SSIM(x_original, x_rebuilt))
            else:
                # SSIM for RGB images with three channels, averaged across channels
                self.contextual_loss_fn = lambda y_true, y_pred: 1.0 * (1 - tf.reduce_mean(
                    (
                        tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1], max_val=1.0) +
                        tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2], max_val=1.0) +
                        tf.image.ssim(y_true[..., 2:3], y_pred[..., 2:3], max_val=1.0)
                    ) / 3.0,
                    axis=None)) + 10.0 * huber(y_true, y_pred)  # mean(1 - SSIM(x_original, x_rebuilt)) across RGB channels
        else:
            raise ValueError(f'Unknown contextual loss function: {con_loss}')

        # Loss for LATENT (or ENCODER) LOSS
        self.latent_loss = tf.keras.losses.MeanSquaredError()

        # Loss for NOISE LOSS
        self.noise_loss = tf.keras.losses.MeanSquaredError()

        # Loss for ADVERSARIAL LOSS
        adv_loss = adv_loss.strip().lower()
        if adv_loss == 'mse':
            self.adversarial_loss = tf.keras.losses.MeanSquaredError()
        elif adv_loss == 'cos':
            self.cos_sim = tf.keras.losses.CosineSimilarity()
            self.adversarial_loss = lambda y_true, y_pred: (1.0 + self.cos_sim(y_true, y_pred))
        else:
            raise ValueError(f'Unknown adversarial loss function: {adv_loss}')

        # Loss for DISCRIMINATOR LOSS
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy()

        # Loss for SEGMENTATOR LOSS
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.focal = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False, alpha=0.25, gamma=2.0)

        self.segmentator_loss = lambda y_true, y_pred: self.bce(y_true, y_pred) + 2.0 * self.focal(y_true, y_pred)

        self.encoder_model, self.autoencoder_model, self.generator_model = build_generator(
            image_size=self.patch_size,
            channels=self.channels,
            initial_filters=64,
            latent_size=256,
            dense_bottleneck=False,
            name='generator',
            net_shape=(2, 2, 2, 2, 2),
            flbn=False,
            bias=False,
            enc_act='lrelu',
            dec_act='lrelu',
            iks=4,
            batch_norm=use_bn,
            kernel_size=3,
            initial_padding=-1,
            initial_padding_filters=-1,
            verbose=False,
        )

        self.discriminator_model = build_discriminator(
            image_size=self.patch_size,
            channels=self.channels,
            initial_filters=64,
            name='generator',
            net_shape=(2, 2, 2, 2, 2),
            flbn=False,
            bias=False,
            enc_act='lrelu',
            iks=4,
            batch_norm=use_bn,
            kernel_size=3,
            initial_padding=-1,
            initial_padding_filters=-1,
            verbose=False,
        )

        self.segmentator_model = build_segmentator(
            image_size=self.patch_size,
            channels=int(self.channels * 2),
            init_filters=128,
            verbose=False,
        )

        if generator_model_w_path is not None:
            self.generator_model.load_weights(generator_model_w_path)
        if discriminator_model_w_path is not None:
            self.discriminator_model.load_weights(discriminator_model_w_path)
        if segmentator_model_w_path is not None:
            self.segmentator_model.load_weights(segmentator_model_w_path)

        self.perlin_generator = Perlin(
            size=self.patch_size,
            target_size=(self.patch_size, self.patch_size),
            reference_dataset=self.reference_dataset,
            real_defect_dataset=self.real_defect_dataset,
            fraction=0.75,
            choice=0.50,
            def_choice=0.20,
            use_perlin_noise=True,
            use_gaussian_noise=True,
            generate_perlin_each_time=False,
            perlin_queue_max=1000,
            perlin_queue_min=1000,
            perlin_generation_every_n_epochs=1,
            perlin_generate_m_perturbations=150,
        )

        self.aug_pipe = AugmentPipe(
            random_90_rotation=3,
            rotation_angle=np.pi,
            flip_mode='both',
            translation_range=20,
            zoom_range=0.15,
        )

        self.aggregator = Aggregator(
            self.test_dataset,
            self.patch_size,
            self.patches,
            self.strides,
            self.autoencoder_model,
            self.segmentator_model,
            self.contextual_loss_fn,
            self.segmentator_loss,
            regular_expression=r'^(.+)\.[A-z]+$',
            group_descriptor={
                GroupTypeEnum.BATCH: 'BATCH',
                GroupTypeEnum.RECIPE: 'RECIPE',
                GroupTypeEnum.STATION: 'STATION',
                GroupTypeEnum.ID: 1,
                GroupTypeEnum.RUN: '00',
                GroupTypeEnum.FRAME: '00',
                GroupTypeEnum.VIAL: '00',
                GroupTypeEnum.REGION: '00',
            },
            db_aggregate_levels=[GroupTypeEnum.ID, GroupTypeEnum.PATCH],
            metric_aggregate_level=GroupTypeEnum.RUN,
            score=ScoreEnum.HEATMAP,
            metric=MetricEnum.ACCURACY,
            blur_kernel_size=5,
            database_name='GRDNetDB',
            run_table_name=self.experiment_name_with_datetime,
            database_host='localhost',
            database_user='root',
            database_pwd='BE',
            writer=self.writer
        )

        self.total_train_steps = 0
        self.total_valid_steps = 0
        self.total_test_steps = 0


    def apply_gaussian_noise_to_image(self, image: tf.Tensor):
        """
        Applies a random Gaussian noise to the input image.

        The function first generates two random numbers: a standard deviation `sigma` in the range [0, 0.5],
        and a coefficient `beta` in the range [0.3, 8.0]. Then, it generates a normal random noise of the
        same shape as the input image, with the generated standard deviation and mean 0. The noise is then
        multiplied by the generated coefficient and added to the input image. The result is then clipped to
        the range [0, 1].

        Parameters:
        - image: The input image to add noise to.

        Returns:
        - An image tensor with the same shape as the input image, but with a random Gaussian noise added.
        """
        sigma = tf.random.uniform(shape=(), minval=0, maxval=0.5, dtype=tf.float32)
        beta = tf.random.uniform(shape=(), minval=0.1, maxval=1.0, dtype=tf.float32)
        noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=sigma, dtype=image.dtype)
        image = tf.clip_by_value(t=(image + (beta * noise)), clip_value_min=0.0, clip_value_max=1.0)
        return image


    @tf.function(autograph=True, reduce_retracing=True)
    def apply_gaussian_noise_to_batch(self, batch: tf.Tensor):
        """
        Applies random Gaussian noise to all the images in a batch.

        The function takes a batch of images, applies the `apply_gaussian_noise_to_image` function to each image
        using `tf.map_fn`, and returns the resulting batch of noisy images.

        Parameters:
        - batch: A tensor of shape (batch_size, height, width, channels) representing the input batch of images.

        Returns:
        - A tensor of shape (batch_size, height, width, channels) representing the resulting batch of noisy images.
        """
        batch = tf.map_fn(fn=self.apply_gaussian_noise_to_image, elems=batch, parallel_iterations=16, swap_memory=True)
        return batch


    @tf.function(autograph=True, reduce_retracing=True)
    def augment(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Applies random augmentations to the input batch of images and masks.

        The function takes a batch of images and masks, applies the `apply` method of the `AugmentPipe` instance
        to each image and mask using `tf.map_fn`, and returns the resulting batch of augmented images and masks.

        Parameters:
        - x: A tensor of shape (batch_size, height, width, channels) representing the input batch of images.
        - mask: An optional tensor of shape (batch_size, height, width, 1) representing the input batch of masks.

        Returns:
        - A tuple of two tensors: the first is a tensor of shape (batch_size, height, width, channels) representing
            the resulting batch of augmented images, and the second is a tensor of shape (batch_size, height, width, 1)
            representing the resulting batch of augmented masks.
        """
        x, roi = tf.map_fn(
            fn=lambda x_: self.aug_pipe.apply(image=x_[0], mask=x_[1]),
            elems=(x, mask),
            parallel_iterations=12,
            swap_memory=True,
            fn_output_signature=(
                tf.TensorSpec(shape=None, dtype=x.dtype),
                tf.TensorSpec(shape=None, dtype=mask.dtype),
            )
        )
        roi = tf.where(roi > 0.5, 1.0, 0.0)
        return x, roi


    @tf.function(autograph=True, reduce_retracing=True)
    def extract_patches(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # Extract patches
        # For Images
        """
        Extracts patches from the input batch of images and masks.

        The function takes a batch of images and masks, applies `tf.image.extract_patches` to each image and mask,
        and returns the resulting batch of extracted patches.

        Parameters:
        - x: A tensor of shape (batch_size, height, width, channels) representing the input batch of images.
        - mask: An optional tensor of shape (batch_size, height, width, 1) representing the input batch of masks.

        Returns:
        - A tuple of two tensors: the first is a tensor of shape (batch_size * num_patches, patch_size, patch_size, channels)
            representing the resulting batch of extracted patches from the input images, and the second is a tensor of shape
            (batch_size * num_patches, patch_size, patch_size, 1) representing the resulting batch of extracted patches from
            the input masks.
        """
        sizes = [1, self.patch_size, self.patch_size, 1]
        strides = [
            1,
            self.patch_size if self.strides is None else self.strides[0],
            self.patch_size if self.strides is None else self.strides[1],
            1
        ]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        x = tf.image.extract_patches(
            images=x,
            sizes=sizes,
            strides=strides,
            rates=rates,
            padding=padding
        )
        x = tf.reshape(x, [-1, self.patch_size, self.patch_size, self.channels])
        # For ROIs
        if mask is not None:
            mask = tf.image.extract_patches(
                images=mask,
                sizes=sizes,
                strides=strides,
                rates=rates,
                padding=padding
            )
            mask = tf.reshape(mask, [-1, self.patch_size, self.patch_size, 1])
        return x, mask


    @tf.function(autograph=True, reduce_retracing=True)
    def train_step(
        self,
        x: tf.Tensor,
        x_noise: tf.Tensor,
        roi: tf.Tensor,
        m: tf.Tensor,
        n: tf.Tensor,
        betas: tf.Tensor
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        # Ensure that the map and the roi are binary (float32 but only 0.0 and 1.0)
        roi = tf.where(roi > 0.5, 1.0, 0.0)
        m = tf.where(m > 0.5, 1.0, 0.0)
        m_ = tf.math.multiply_no_nan(m, roi)

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as segmentator_tape:
            # Generator step: Z, Xf, Zf = Generator(X_Noise)
            latent_space, x_rebuilt, latent_space_rebuilt = self.generator_model(x_noise, training=True)

            # Discriminator step: f(X), ClassX = Discriminator(X) ; f(Xf), ClassXf = Discriminator(Xf)
            feat_real, x_real = self.discriminator_model(x, training=True)
            feat_fake, x_fake = self.discriminator_model(tf.stop_gradient(x_rebuilt), training=True)

            # Segmentator step: Map = Segmentator(X concat X_Rebuilt)
            m_rebuilt = self.segmentator_model(tf.concat((x_noise, tf.stop_gradient(x_rebuilt)), axis=-1), training=True)

            # Calculate loss between X_Rebuilt = Model(X_Noise) and X (Original images)
            contextual_loss_value = self.contextual_loss_fn(x, x_rebuilt)
            adversarial_loss_value = self.adversarial_loss(feat_real, feat_fake)
            latent_loss_value = self.latent_loss(latent_space, latent_space_rebuilt)
            noise_loss_value = self.noise_loss(tf.math.abs(tf.math.multiply_no_nan(tf.math.multiply_no_nan((1.0 - betas), m_rebuilt), x_rebuilt) - tf.math.multiply_no_nan(m_, x_noise)), tf.math.multiply_no_nan(betas, n))
            generator_loss_value = 10.0 * contextual_loss_value + 1.0 * adversarial_loss_value + 1.0 * latent_loss_value + 1.0 * noise_loss_value

            discriminator_loss_value = ((self.discriminator_loss(tf.ones_like(x_real), x_real)) + (self.discriminator_loss(tf.zeros_like(x_fake), x_fake))) / 2.0

            segmentator_loss_value = self.segmentator_loss(m_, m_rebuilt)
        
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        generator_grads = generator_tape.gradient(generator_loss_value, self.generator_model.trainable_weights)  # L'(x, x_rebuilt) = L'(x, model(x_noise)) = L'(x, model(x + noise))
        discriminator_grads = discriminator_tape.gradient(discriminator_loss_value, self.discriminator_model.trainable_weights)
        segmentator_grads = segmentator_tape.gradient(segmentator_loss_value, self.segmentator_model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator_model.trainable_weights))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator_model.trainable_weights))
        self.segmentator_optimizer.apply_gradients(zip(segmentator_grads, self.segmentator_model.trainable_weights))

        return {
            'images' : {
                'X' : x,
                'Xn' : x_noise,
                'Xf' : x_rebuilt,
                'M': m_,
                'Mf': m_rebuilt,
            },
            'latents' : {
                'Z' : latent_space,
                'Zf' : latent_space_rebuilt,
                'feat_real' : feat_real,
                'feat_fake' : feat_fake,
            },
            'losses' : {
                'contextual_loss' : contextual_loss_value,
                'adversarial_loss' : adversarial_loss_value,
                'latent_loss' : latent_loss_value,
                'noise_loss' : noise_loss_value,
                'generator_loss' : generator_loss_value,
                'discriminator_loss' : discriminator_loss_value,
                'segmentator_loss' : segmentator_loss_value,
            }
        }


    @tf.function(autograph=True, reduce_retracing=True)
    def test_step(
        self,
        x: tf.Tensor,
        x_noise: tf.Tensor,
        roi: tf.Tensor,
        m: tf.Tensor,
        n: tf.Tensor,
        betas: tf.Tensor
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        # Generator step: Z, Xf, Zf = Generator(X_Noise)
        latent_space, x_rebuilt, latent_space_rebuilt = self.generator_model(x_noise, training=False)

        # Discriminator step: f(X), ClassX = Discriminator(X) ; f(Xf), ClassXf = Discriminator(Xf)
        feat_real, x_real = self.discriminator_model(x, training=False)
        feat_fake, x_fake = self.discriminator_model(x_rebuilt, training=False)

        # Segmentator step: Map = Segmentator(X concat X_Rebuilt)
        m_rebuilt = self.segmentator_model(tf.concat((x_noise, x_rebuilt), axis=-1), training=False)

        # Calculate loss between X_Rebuilt = Model(X_Noise) and X (Original images)
        contextual_loss_value = self.contextual_loss_fn(x, x_rebuilt)
        adversarial_loss_value = self.adversarial_loss(feat_real, feat_fake)
        latent_loss_value = self.latent_loss(latent_space, latent_space_rebuilt)
        noise_loss_value = self.noise_loss(tf.math.abs(tf.math.multiply_no_nan(tf.math.multiply_no_nan((1.0 - betas), m_rebuilt), x_rebuilt) - tf.math.multiply_no_nan(m, x_noise)), tf.math.multiply_no_nan(betas, n))
        generator_loss_value = 10.0 * contextual_loss_value + 1.0 * adversarial_loss_value + 1.0 * latent_loss_value + 1.0 * noise_loss_value

        discriminator_loss_value = ((self.discriminator_loss(tf.ones_like(x_real), x_real)) + (self.discriminator_loss(tf.zeros_like(x_fake), x_fake))) / 2.0

        segmentator_loss_value = self.segmentator_loss(tf.math.multiply_no_nan(roi, m), m_rebuilt)

        return {
            'images' : {
                'X' : x,
                'Xn' : x_noise,
                'Xf' : x_rebuilt,
                'M': m,
                'Mf': m_rebuilt,
            },
            'latents' : {
                'Z' : latent_space,
                'Zf' : latent_space_rebuilt,
                'feat_real' : feat_real,
                'feat_fake' : feat_fake,
            },
            'losses' : {
                'contextual_loss' : contextual_loss_value,
                'adversarial_loss' : adversarial_loss_value,
                'latent_loss' : latent_loss_value,
                'noise_loss' : noise_loss_value,
                'generator_loss' : generator_loss_value,
                'discriminator_loss' : discriminator_loss_value,
                'segmentator_loss' : segmentator_loss_value,
            }
        }


    def train_loop(self, epoch: int, force_shuffle: bool = False) -> Dict[str, float]:
        train_images = self.train_dataset

        loss_acc: Dict[str, float] = {}

        min_area: int = 25
        self.perlin_generator.pre_generate_noise(epoch=epoch, min_area=min_area)

        images_num: int = 0

        x: tf.Tensor = None
        roi: tf.Tensor = None
        x_perlin: tf.Tensor = None
        perlin_noise: tf.Tensor = None
        perlin_noise_mask: tf.Tensor = None
        x_rebuilt: tf.Tensor = None
        m_rebuilt: tf.Tensor = None

        with tqdm(train_images, desc='Train', leave=True, unit='batch') as pbar:
            for step, (x, roi) in enumerate(pbar):

                # Extract patches
                x, roi = self.extract_patches(x, roi)

                # Augment the batch not patches
                x, roi = self.augment(x, roi)

                actual_batch_size = int(x.shape[0])
                images_num += actual_batch_size

                if force_shuffle:
                    batch_images = list(zip(x, roi))
                    random.shuffle(batch_images)
                    batch_images = tuple(batch_images)
                    x, roi = zip(*batch_images)
                    x = tf.stack(x, axis=0)
                    roi = tf.stack(roi, axis=0)
                    del batch_images

                x, x_perlin, perlin_noise, perlin_noise_mask, betas = self.perlin_generator.perlin_noise_batch(
                    X=x,
                    channels=self.channels,
                    p=0.75,
                    area_min=min_area
                )

                x_noise = self.apply_gaussian_noise_to_batch(x_perlin)

                gc.collect()

                dict_values = self.train_step(
                    x=x,
                    x_noise=x_noise,
                    roi=roi,
                    m=perlin_noise_mask,
                    n=perlin_noise,
                    betas=betas
                )

                x_rebuilt = dict_values['images']['Xf']
                m_rebuilt = dict_values['images']['Mf']
                m_effect = dict_values['images']['M']

                losses_dict = dict_values['losses']
                for key, value in losses_dict.items():
                    if key in loss_acc:
                        loss_acc[key] += value.numpy().item() * float(actual_batch_size)
                    else:
                        loss_acc[key] = value.numpy().item() * float(actual_batch_size)
                
                losses_postfix_dict = {key: f'{value.numpy().item():.4f}' for key, value in losses_dict.items()}

                self.total_train_steps += 1

                with self.writer.as_default():
                    for key, value in losses_dict.items():
                        tf.summary.scalar(f'TRAIN: {key}', value, step=self.total_train_steps)
                    tf.summary.scalar('LEARNING RATE', float(self.generator_optimizer.learning_rate.numpy()), step=self.total_train_steps)

                pbar.set_postfix(losses_postfix_dict)

                if step % 100 == 0:
                    with self.writer.as_default():
                        if x is not None:
                            roi_ = tf.where(roi > 0.5, 1.0, 0.0)
                            m_ = tf.where(perlin_noise_mask > 0.5, 1.0, 0.0)
                            mul_ = tf.math.multiply(m_, roi_)
                            tf.summary.image(f"1.a TRAIN [{step}]: x", x, max_outputs=int(len(x)), step=epoch)
                            tf.summary.image(f"1.b TRAIN [{step}]: x_perlin", x_perlin, max_outputs=int(len(x_perlin)), step=epoch)
                            tf.summary.image(f"1.c TRAIN [{step}]: roi", roi_, max_outputs=int(len(roi_)), step=epoch)
                            tf.summary.image(f"1.d TRAIN [{step}]: perlin_noise", perlin_noise, max_outputs=int(len(perlin_noise)), step=epoch)
                            tf.summary.image(f"1.e TRAIN [{step}]: perlin_noise_mask", m_, max_outputs=int(len(m_)), step=epoch)
                            tf.summary.image(f"1.f TRAIN [{step}]: mask * roi", mul_, max_outputs=int(len(mul_)), step=epoch)
                            tf.summary.image(f"1.g TRAIN [{step}]: x_rebuilt", x_rebuilt, max_outputs=int(len(x_rebuilt)), step=epoch)
                            tf.summary.image(f"1.h TRAIN [{step}]: m_rebuilt", m_rebuilt, max_outputs=int(len(m_rebuilt)), step=epoch)
                            tf.summary.image(f"1.i TRAIN [{step}]: m_effect", m_effect, max_outputs=int(len(m_effect)), step=epoch)
        
        losses_mean_dict = {}
        for key, value in loss_acc.items():
            losses_mean_dict[key] = value / float(images_num)

        return losses_mean_dict


    def test_loop(self, epoch: int) -> Dict[str, float]:
        valid_images = self.validation_dataset

        loss_acc: Dict[str, float] = {}

        min_area: int = 25

        images_num: int = 0

        x: tf.Tensor = None
        roi: tf.Tensor = None
        x_perlin: tf.Tensor = None
        perlin_noise: tf.Tensor = None
        perlin_noise_mask: tf.Tensor = None
        x_rebuilt: tf.Tensor = None
        m_rebuilt: tf.Tensor = None

        with tqdm(valid_images, desc='Validation', leave=True, unit='batch') as pbar:
            for step, (x, roi) in enumerate(pbar):
                # Extract patches
                x, roi = self.extract_patches(x, roi)

                actual_batch_size = int(x.shape[0])
                images_num += actual_batch_size
                roi = tf.where(roi > 0.5, 1.0, 0.0)

                x, x_perlin, perlin_noise, perlin_noise_mask, betas = self.perlin_generator.perlin_noise_batch(
                    X=x,
                    channels=self.channels,
                    p=0.75,
                    area_min=min_area
                )

                x_noise = self.apply_gaussian_noise_to_batch(x_perlin)

                dict_values = self.test_step(
                    x=x,
                    x_noise=x_noise,
                    roi=roi,
                    m=perlin_noise_mask,
                    n=perlin_noise_mask,
                    betas=betas
                )

                x_rebuilt = dict_values['images']['Xf']
                m_rebuilt = dict_values['images']['Mf']

                losses_dict = dict_values['losses']
                for key, value in losses_dict.items():
                    if key in loss_acc:
                        loss_acc[key] += value.numpy().item() * float(actual_batch_size)
                    else:
                        loss_acc[key] = value.numpy().item() * float(actual_batch_size)
                
                losses_postfix_dict = {key: f'{value.numpy().item():.4f}' for key, value in losses_dict.items()}

                self.total_valid_steps += 1

                with self.writer.as_default():
                    for key, value in losses_dict.items():
                        tf.summary.scalar(f'VALID: {key}', value, step=self.total_valid_steps)

                pbar.set_postfix(losses_postfix_dict)

        with self.writer.as_default():
            if x is not None:
                tf.summary.image("2.a VALIDATION: x", x, max_outputs=int(len(x)), step=epoch)
                tf.summary.image("2.b VALIDATION: x_perlin", x_perlin, max_outputs=int(len(x_perlin)), step=epoch)
                tf.summary.image("2.c VALIDATION: roi", roi, max_outputs=int(len(roi)), step=epoch)
                tf.summary.image("2.d VALIDATION: perlin_noise", perlin_noise, max_outputs=int(len(perlin_noise)), step=epoch)
                tf.summary.image("2.e VALIDATION: perlin_noise_mask", perlin_noise_mask, max_outputs=int(len(perlin_noise_mask)), step=epoch)
                tf.summary.image("2.f VALIDATION: x_rebuilt", x_rebuilt, max_outputs=int(len(x_rebuilt)), step=epoch)
                tf.summary.image("2.g VALIDATION: m_rebuilt", m_rebuilt, max_outputs=int(len(m_rebuilt)), step=epoch)
        
        losses_mean_dict = {}
        for key, value in loss_acc.items():
            losses_mean_dict[key] = value / float(images_num)

        return losses_mean_dict


    def train(self, epochs: int) -> Tuple[dict, dict, dict, dict]:
        history: Dict[str, Dict[str, List[float]]] = {}
        best_metric_value: Optional[float] = None
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1} / {epochs}')

            # Train for one epoch over batches
            train_losses_mean_over_batches = self.train_loop(
                epoch=epoch,
                force_shuffle=True,
            )

            for key, value in train_losses_mean_over_batches.items():
                print(f'Train {key} over batches: {value}')

            # Valid for one epoch over batches
            test_losses_mean_over_batches = self.test_loop(
                epoch=epoch
            )

            for key, value in test_losses_mean_over_batches.items():
                print(f'Validation {key} over batches: {value}')

            _, best_value = self.aggregator.aggregate(
                epoch=epoch
            )
            to_be_saved: bool = False
            if best_metric_value is None:
                best_metric_value = best_value
                to_be_saved = True
            if best_value > best_metric_value:
                best_metric_value = best_value
                to_be_saved = True
            if to_be_saved:
                self.save_models(
                    methods=['keras', 'weight'],
                    epoch=epoch,
                    metric=best_metric_value
                )

            for key, value in train_losses_mean_over_batches.items():
                if key not in history:
                    history[key] = {}

                if 'loss' not in history[key]:
                    history[key]['loss'] = []
                    history[key]['val_loss'] = []

                history[key]['loss'].append(train_losses_mean_over_batches[key])
                history[key]['val_loss'].append(test_losses_mean_over_batches[key])
        
            gc.collect()
            print()
            print('#' * 10)
            print()

        return history, best_metric_value


    def save_models(
        self,
        methods: Union[str, List[str], Tuple[str, ...]],
        epoch: int,
        metric: float
    ) -> None:
        """
        Saves the models in the specified formats.

        Parameters:
            methods (Union[str, List[str], Tuple[str, ...]]): The methods to save the models with. Can be a string for a single method, a list or tuple of strings for multiple methods, or 'all' to save with all methods.
            epoch (int): The epoch to save the models at.
            metric (float): The metric to save the models at.

        Returns:
            None
        """
        path = f'{self.logdir}/models/epoch{epoch:03d}_metric{metric:.4f}'

        if methods == 'all':
            methods = ['kh5', 'ktf', 'keras', 'weight']
        elif isinstance(methods, str):
            methods = [methods]
    
        # Save shared metadata
        metadata_path = f'{path}/metadata.json'
        os.makedirs(path, exist_ok=True)
        with open(metadata_path, 'w') as metadata_file:
            json.dump({'epoch': epoch, 'metric': metric}, metadata_file)
        print('\tShared metadata saved.')

        print ('Saving models...')
        if 'kh5' in methods:
            print ('\tkh5:')
            path_dir = f'{path}/kh5'
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)
            tf.keras.models.save_model(self.generator_model,        f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.h5')
            tf.keras.models.save_model(self.autoencoder_model,      f'{path_dir}/autoencoder_model_e{epoch:03d}_m{metric:.4f}.h5')
            tf.keras.models.save_model(self.discriminator_model,    f'{path_dir}/discriminator_model_e{epoch:03d}_m{metric:.4f}.h5')
            tf.keras.models.save_model(self.segmentator_model,      f'{path_dir}/segmentator_model_e{epoch:03d}_m{metric:.4f}.h5')
            print ('DONE.')

        if 'ktf' in methods:
            print ('\tktf:')
            path_dir = f'{path}/ktf'
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)
            self.generator_model.export                         (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}', format='tf_saved_model')
            self.autoencoder_model.export                       (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}', format='tf_saved_model')
            self.discriminator_model.export                     (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}', format='tf_saved_model')
            self.segmentator_model.export                       (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}', format='tf_saved_model')
            print ('DONE.')

        
        if 'keras' in methods:
            print ('\tkeras:')
            path_dir = f'{path}/keras'
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)
            self.generator_model.save                           (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.keras')
            self.autoencoder_model.save                         (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.keras')
            self.discriminator_model.save                       (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.keras')
            self.segmentator_model.save                         (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.keras')
            print ('DONE.')

        if 'weight' in methods:
            print ('\tweight:')
            path_dir = f'{path}/weight'
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, exist_ok=True)
            self.generator_model.save_weights                   (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.weights.h5')
            self.autoencoder_model.save_weights                 (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.weights.h5')
            self.discriminator_model.save_weights               (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.weights.h5')
            self.segmentator_model.save_weights                 (   f'{path_dir}/generator_model_e{epoch:03d}_m{metric:.4f}.weights.h5')
            print ('DONE.')
