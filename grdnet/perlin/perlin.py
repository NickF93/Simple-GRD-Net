"""
Module for generating and managing Perlin noise perturbations. 

This module provides functionalities for generating Perlin noise, applying it to image tensors, and managing
a queue of generated noise for reuse in machine learning tasks. The Perlin class handles all the core noise
generation, augmentation, and perturbation functionalities.

Dependencies:
- TensorFlow (tf): Used for tensor manipulations and operations.
- Numpy (np): Used for array manipulations and mathematical operations.
- Tqdm (tqdm.auto): Provides progress bars for longer-running noise generation processes.
- Imgaug (imgaug.augmenters): Augmentation library used for applying image transformations to the noise.
"""

import math
import random
import gc

from typing import Tuple, Union, List, Optional, Callable

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
import imgaug.augmenters as iaa

from ..augment import AugmentPipe
from ..util import set_seed

class Perlin:
    """
    Class for generating and managing Perlin noise perturbations.

    The Perlin class provides functionalities to create noise perturbations, apply augmentations, and maintain
    a queue of generated Perlin noise. This noise can be applied to image tensors for tasks like data augmentation
    in machine learning models.

    Attributes:
        size (int): Size of the image or tensor to generate noise for.
        target_size (Tuple[int, int]): The target size for resizing the noise or image.
        reference_dataset (Optional[tf.data.Dataset]): Reference dataset for pulling real images.
        real_defect_dataset (Optional[tf.data.Dataset]): Dataset containing real defects.
        fraction (float): Fraction of the dataset to use for noise generation.
        choice (float): Probability of selecting reference data for noise application.
        def_choice (float): Probability of selecting real defect data for noise application.
        use_perlin_noise (bool): Flag indicating if Perlin noise should be used.
        use_gaussian_noise (bool): Flag indicating if Gaussian noise should be used.
        generate_perlin_each_time (bool): Flag for generating new Perlin noise each time or reusing.
        perlin_queue_max (int): Maximum number of Perlin noise tensors to store.
        perlin_queue_min (int): Minimum number of Perlin noise tensors to generate initially.
        perlin_generation_every_n_epochs (int): Interval of epochs to generate new Perlin noise.
        perlin_generate_m_perturbations (int): Number of perturbations to generate when refreshing noise.
        perlin_noise_array (List[tf.Tensor]): Queue to store generated Perlin noise.
        perlin_mask_array (List[tf.Tensor]): Queue to store generated Perlin noise masks.

    Methods:
        generate_perlin_noise: Generates or retrieves Perlin noise based on the specified probability.
        perlin_noise_batch: Applies Perlin noise perturbation to a batch of image tensors.
        perlin_perturbation: Generates Perlin noise perturbations with specified augmentations.
        generate_perlin_perturbation_greater_than: Ensures noise area exceeds a minimum size.
        pre_generate_noise: Pre-generates noise for efficient retrieval in training loops.
    """
    def __init__(self,
                 size: int,
                 target_size: tuple[int, int],
                 reference_dataset: Optional[tf.data.Dataset] = None,
                 real_defect_dataset: Optional[tf.data.Dataset] = None,
                 fraction: float = 0.75,
                 choice: float = 0.25,
                 def_choice: float = 0.25,
                 use_perlin_noise: bool = True,
                 use_gaussian_noise: bool = True,
                 generate_perlin_each_time: bool = False,
                 perlin_queue_max: int = 1000,
                 perlin_queue_min: int = 500,
                 perlin_generation_every_n_epochs: int = 1,
                 perlin_generate_m_perturbations: int = 10):
        """
        Initializes the class with various parameters for image augmentation, 
        noise generation, and dataset iteration.

        Args:
            size (int): The size of the images to be processed.
            target_size (tuple[int, int]): The target size for image resizing.
            reference_dataset (Optional[tf.data.Dataset]): A TensorFlow dataset for reference data. Defaults to None.
            real_defect_dataset (Optional[tf.data.Dataset]): A TensorFlow dataset for real defect data. Defaults to None.
            fraction (float): Fraction of how much the reference dataset will be used. Defaults to 0.75.
            choice (float): Fraction for random choice when picking from reference dataset. Defaults to 0.25.
            def_choice (float): Fraction for random choice when picking from the defect dataset. Defaults to 0.25.
            use_perlin_noise (bool): Whether to use Perlin noise augmentation. Defaults to True.
            use_gaussian_noise (bool): Whether to use Gaussian noise augmentation. Defaults to True.
            generate_perlin_each_time (bool): Flag to generate Perlin noise for every batch. Defaults to False.
            perlin_queue_max (int): Maximum size of the Perlin noise queue. Defaults to 1000.
            perlin_queue_min (int): Minimum size of the Perlin noise queue. Defaults to 500.
            perlin_generation_every_n_epochs (int): Number of epochs between Perlin noise regeneration. Defaults to 1.
            perlin_generate_m_perturbations (int): Number of Perlin noise perturbations to generate each time. Defaults to 10.
        """
        
        # Image size and target size for resizing
        self.size = size
        self.target_size = target_size

        # Reference and defect datasets, if provided
        self.reference_dataset = reference_dataset
        self.real_defect_dataset = real_defect_dataset

        # Fraction settings for reference and defect dataset usage
        self.fraction = fraction
        self.choice = choice
        self.def_choice = def_choice

        # Iterator setup for reference and defect datasets
        self.r_iter = iter(self.reference_dataset) if self.reference_dataset else None
        self.d_iter = iter(self.real_defect_dataset) if self.real_defect_dataset else None

        # Image augmentation pipeline configuration
        self.pipe = AugmentPipe(
            random_90_rotation=3,
            rotation_angle=np.pi,
            flip_mode='both',
            translation_range=round(max(target_size) / 200),
            zoom_range=0.1
        )

        # Noise augmentation configurations
        self.use_perlin_noise = use_perlin_noise
        self.use_gaussian_noise = use_gaussian_noise
        self.generate_perlin_each_time = generate_perlin_each_time
        self.perlin_queue_max = perlin_queue_max
        self.perlin_queue_min = perlin_queue_min
        self.perlin_generation_every_n_epochs = perlin_generation_every_n_epochs
        self.perlin_generate_m_perturbations = perlin_generate_m_perturbations

        # Initializing empty arrays for Perlin noise and masks
        self.perlin_noise_array: List[np.ndarray] = []
        self.perlin_mask_array: List[np.ndarray] = []

    def get_image(self, regularize: bool = True) -> tf.Tensor:
        """
        Retrieves and augments an image from the reference dataset, applying a series of transformations
        such as random rotations, flips, and color adjustments. Optionally regularizes the image.

        Args:
            regularize (bool): Whether to regularize the image to [0, 1] range by normalizing based on max and min values. Defaults to True.

        Returns:
            tf.Tensor: The augmented and potentially regularized image tensor.
        """
        try:
            # Try to get the next batch from the reference dataset iterator
            batch = next(self.r_iter)
        except StopIteration:
            # Reset the iterator if it reaches the end of the dataset
            self.r_iter = iter(self.reference_dataset)
            batch = next(self.r_iter)

        # Unpack the batch, only considering the first element (image)
        x, *_ = batch

        # Handle cases where x is a tuple or a list, and select a random element from it
        if isinstance(x, (tuple, list)):
            x = x[random.randint(0, len(x) - 1)]
        # Handle cases where x is a tensor and has 4 dimensions (batch of images)
        elif isinstance(x, tf.Tensor):
            if len(x.shape) == 4:
                # Randomly select one image from the batch
                x = x[random.randint(0, x.shape[0] - 1)]

        # Apply image augmentations: random rotation, flips, hue, saturation, and brightness adjustments
        x = tf.image.rot90(x, random.randint(-3, 3))
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        if x.shape[-1] == 3:
            x = tf.image.random_hue(x, 0.05)
            x = tf.image.random_saturation(x, 0.95, 1.05)
        x = tf.image.random_brightness(x, 0.02)

        # Optionally regularize the image to the range [0, 1]
        if regularize:
            xmax = tf.math.reduce_max(x)
            xmin = tf.math.reduce_min(x)
            a = tf.math.maximum(xmax, 1.0)  # Ensure the maximum value is at least 1.0
            b = tf.math.minimum(xmin, 0.0)  # Ensure the minimum value is at most 0.0
            # Normalize x to the range [0, 1]
            x = tf.math.divide_no_nan(tf.math.subtract(x, b), tf.math.subtract(a, b))

        # Clip values to ensure they are in the range [0, 1]
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)

        # Apply central cropping and random cropping based on target size
        x = tf.image.central_crop(x, central_fraction=self.fraction)
        x = tf.image.random_crop(x, size=(self.size, self.size, x.shape[-1]))

        return x

    def lerp_np(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float], w: float) -> Union[np.ndarray, float]:
        """
        Linearly interpolates between two values or arrays `x` and `y` based on weight `w`.

        Args:
            x (Union[np.ndarray, float]): The starting value or array.
            y (Union[np.ndarray, float]): The target value or array.
            w (float): The interpolation factor. Typically in the range [0, 1] where:
                       - 0 returns `x`
                       - 1 returns `y`
                       - Values in between return a weighted mix of `x` and `y`.

        Returns:
            Union[np.ndarray, float]: The interpolated result between `x` and `y` based on the weight `w`.
        """
        # Perform the linear interpolation
        fin_out = (y - x) * w + x
        return fin_out

    def _rand_perlin_2d_np(self, 
                           shape: Tuple[int, int], 
                           res: Tuple[int, int], 
                           fade: Callable[[np.ndarray], np.ndarray] = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
                           ) -> np.ndarray:
        """
        Generates a 2D Perlin noise pattern using numpy.

        Args:
            shape (Tuple[int, int]): The shape of the output noise image (height, width).
            res (Tuple[int, int]): The resolution of the grid (controls the scale of the noise).
            fade (Callable[[np.ndarray], np.ndarray]): The fade function used for smoothing the transition between 
                                                       gradients. Defaults to the Perlin fade function.

        Returns:
            np.ndarray: A 2D array representing the generated Perlin noise.
        """
        
        # Delta represents the ratio between grid resolution and the desired shape
        delta = (res[0] / shape[0], res[1] / shape[1])
        
        # 'd' is the size of each grid cell in pixels
        d = (shape[0] // res[0], shape[1] // res[1])
        
        # Grid of coordinates modulo 1 (i.e., the fractional part of the grid)
        grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

        # Generate random angles for gradients
        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        # Stack the gradient vectors (cos and sin of angles) for each grid point
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        # Helper function to tile gradient vectors across the grid
        def tile_grads(slice1: Tuple[int, int], slice2: Tuple[int, int]) -> np.ndarray:
            g = gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]]
            a = np.repeat(g, d[0], axis=0)
            b = np.repeat(a, d[1], axis=1)
            return b

        # Helper function to compute dot product between gradient vectors and displacement vectors
        def dot(grad: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
            return (
                np.stack((
                    grid[:shape[0], :shape[1], 0] + shift[0], 
                    grid[:shape[0], :shape[1], 1] + shift[1]
                ), axis=-1) * grad[:shape[0], :shape[1]]
            ).sum(axis=-1)

        # Compute dot products for the four corners of the grid cells
        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        
        # Fade the grid to smooth transitions between gradients
        t = fade(grid[:shape[0], :shape[1]])

        # Interpolate between dot products along x and then along y
        return math.sqrt(2) * self.lerp_np(
            self.lerp_np(n00, n10, t[..., 0]), 
            self.lerp_np(n01, n11, t[..., 0]), 
            t[..., 1]
        )

    def random_2d_perlin(self, 
                         shape: Tuple[int, int], 
                         res: Tuple[Union[int, tf.Tensor], Union[int, tf.Tensor]], 
                         fade: Callable[[np.ndarray], np.ndarray] = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
                         ) -> Union[np.ndarray, tf.Tensor]:
        """
        Generates a random 2D Perlin noise array or tensor.

        Args:
            shape (Tuple[int, int]): The shape of the 2D map (height, width).
            res (Tuple[Union[int, tf.Tensor], Union[int, tf.Tensor]]): Tuple specifying the scale of Perlin noise 
                                                                      for both height and width dimensions.
            fade (Callable[[np.ndarray], np.ndarray], optional): Function used for smoothing transitions in the 
                                                                 generated Perlin noise. Defaults to 6*t**5 - 15*t**4 + 10*t**3.

        Returns:
            Union[np.ndarray, tf.Tensor]: A random 2D array or tensor generated using Perlin noise.
        
        Raises:
            TypeError: If `res[0]` is not of type `int` (only supports numpy-based Perlin noise generation).
        """
        # Check if resolution (res) is an integer (Numpy-based Perlin noise generation)
        if isinstance(res[0], int):
            # Generate Perlin noise using the Numpy version
            result = self._rand_perlin_2d_np(shape, res, fade)
        else:
            # Raise a TypeError if res[0] is not an integer
            raise TypeError(f"Expected integer resolution for Perlin noise, but got {type(res[0])}")

        return result

    def perlin_perturbation(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates a perturbation using Perlin noise and applies a series of random augmentations
        to create an anomaly source image. It optionally selects real defect images from datasets.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The generated perturbation and corresponding mask.
        """
        
        # Ensure the size is a power of 2 for Perlin noise generation
        size = int(self.size)
        if size & (size - 1) == 0:
            new_size = size
        else:
            # Adjust the size to the next power of 2
            new_size = int(2 ** int(math.ceil(math.log(size) / math.log(2))))

        # Define a list of augmentations
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]

        # Rotation augmentations
        rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        # Randomly selects 3 augmenters from the list
        def rand_augmenter() -> iaa.Sequential:
            """Selects 3 random transforms to be applied to the anomaly source images.
            
            Returns:
                iaa.Sequential: A sequence of 3 randomly selected augmentations.
            """
            aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
            aug = iaa.Sequential([augmenters[aug_ind[0]], augmenters[aug_ind[1]], augmenters[aug_ind[2]]])
            return aug

        # Perlin noise settings
        perlin_scale = 6
        min_perlin_scale = 0

        # Randomly select scales for Perlin noise generation
        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale)

        # Generate Perlin noise
        perlin_noise = self.random_2d_perlin((new_size, new_size), (perlin_scalex, perlin_scaley))
        perlin_noise = rot(image=perlin_noise)

        # Create mask from Perlin noise
        mask = np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        # Create anomaly source image (3-channel grayscale) from Perlin noise
        anomaly_source_img = np.expand_dims(perlin_noise, 2).repeat(3, 2)
        anomaly_source_img = (anomaly_source_img * 255).astype(np.uint8)

        # Apply random augmentations to the anomaly source image
        aug = rand_augmenter()
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Create anomalous perturbation to apply to the image
        perturbation = anomaly_img_augmented.astype(np.float32) * mask / 255.0

        # Resize perturbation and mask to the target size
        perturbation = tf.image.resize(tf.convert_to_tensor(perturbation), self.target_size, method='bilinear', antialias=True)
        mask = tf.image.resize(tf.convert_to_tensor(mask), self.target_size, method='nearest', antialias=True)

        # Apply perturbation or real defect from datasets, based on random chance
        if self.reference_dataset is not None and random.random() > self.choice:
            # Use reference image to apply the perturbation
            x = self.get_image()
            perturbation = tf.math.multiply_no_nan(x, mask)
            del x
        elif self.real_defect_dataset is not None and random.random() > self.def_choice:
            # Use real defect data from the defect dataset
            mask = tf.zeros(shape=(1,))
            indices: List[int] = []
            
            # Ensure valid mask and defect indices
            while tf.reduce_sum(mask) < tf.keras.backend.epsilon() or len(indices) <= 0:
                try:
                    perturbation, mask = next(self.d_iter)
                except StopIteration:
                    # Reset the iterator if it reaches the end of the dataset
                    self.d_iter = iter(self.real_defect_dataset)
                    perturbation, mask = next(self.d_iter)
                finally:
                    indices = [idx for idx in range(int(mask.shape[0])) if tf.reduce_sum(mask[idx]) > tf.keras.backend.epsilon()]

            # Randomly select an image from valid defect data
            if len(perturbation.shape) == 4:
                r = indices[random.randint(0, len(indices) - 1)]
                perturbation = perturbation[r]
                mask = mask[r]
                perturbation = tf.math.multiply_no_nan(perturbation, mask)
                perturbation, mask = self.pipe(perturbation, mask)
            else:
                perturbation = tf.math.multiply_no_nan(perturbation, mask)
                perturbation, mask = self.pipe(perturbation, mask)
            del indices

        if self.is_to_be_resized(perturbation):
            perturbation = tf.image.resize(perturbation, self.target_size, method='bilinear', antialias=True)

        if self.is_to_be_resized(mask):
            mask = tf.image.resize(mask, self.target_size, method='nearest', antialias=True)

        return perturbation, mask

    def is_to_be_resized(self, x):
        int_shapes = tf.keras.backend.int_shape(x)
        if len(int_shapes) == 3:
            hp, wp, _ = int_shapes
        else:
            _, hp, wp, _ = int_shapes

        resize = False
        if self.target_size[0] != hp:
            resize = True
        if self.target_size[1] != wp:
            resize = True
        return resize

    #region [perlin_noise_batch]
    def generate_perlin_perturbation_greater_than(self, area_min: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate a Perlin noise perturbation where the area of the noise mask is greater than a specified minimum.

        Args:
            area_min (float): The minimum area (in pixels) for the Perlin noise perturbation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the generated noise tensor and the corresponding noise mask.
        """
        # Ensure that the minimum area is valid (greater than 0)
        if area_min > 0:
            noise, noise_mask = None, None
            redm = 0.0  # To track the area of the generated noise mask
            redo_count = 0  # To track how many times the perturbation has been regenerated

            # Repeat until a noise mask with an area larger than `area_min` is generated
            while (noise is None or noise_mask is None) or redm < area_min:
                # Generate Perlin noise perturbation
                noise, noise_mask = self.perlin_perturbation()

                # Calculate the sum of the noise mask to estimate its area
                redm = tf.math.reduce_sum(noise_mask)

                # Increment the redo count
                redo_count += 1

                # Reset the Perlin noise seed if the perturbation has been regenerated too many times
                if redo_count > 3:
                    set_seed()  # Assuming `set_seed()` is defined elsewhere
                    print('Reset seed...')

            # Cleanup to free memory
            del redm, redo_count
        else:
            # If no area constraint is required, generate a simple Perlin perturbation
            noise, noise_mask = self.perlin_perturbation()

        return noise, noise_mask

    def generate_perlin_noise(self, area_min: int = 25) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate Perlin noise perturbation, either by generating new noise or by retrieving it from stored arrays.

        Args:
            area_min (int, optional): The minimum area for the Perlin noise perturbation. Defaults to 25.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the generated noise tensor and the corresponding noise mask.
        """
        # Check if new Perlin noise should be generated each time
        if self.generate_perlin_each_time:
            # Generate Perlin noise perturbation greater than the specified minimum area
            noise, noise_mask = self.generate_perlin_perturbation_greater_than(area_min=area_min)
        else:
            # Length of the Perlin noise array
            l = len(self.perlin_noise_array)
            
            # Ensure the noise and mask arrays are of the same length
            assert l == len(self.perlin_mask_array), "Perlin noise arrays size mismatch"
            
            # Select a random index from the available noise arrays
            p = random.randint(0, l - 1)
            
            # Retrieve the noise and corresponding mask from the stored arrays
            noise = tf.convert_to_tensor(self.perlin_noise_array[p])
            noise_mask = tf.convert_to_tensor(self.perlin_mask_array[p])

        return noise, noise_mask

    def perlin_noise_tensors(self, 
                             n: int, 
                             tsize: int = 224, 
                             channels: int = 3, 
                             p: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate multiple Perlin noise tensors with an option to control the probability of Perlin noise generation.

        Args:
            n (int): Number of noise tensors to generate.
            tsize (int, optional): Size of each generated tensor (height and width). Defaults to 224.
            channels (int, optional): Number of channels in the generated tensor. Defaults to 3.
            p (float, optional): Probability of generating Perlin noise instead of a zero tensor. Defaults to 1.0.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple of the noise mask tensor and the noise tensor.
        """
        # Initialize tensors for noise (N) and noise mask (M)
        N: tf.Tensor = None
        M: tf.Tensor = None

        # Loop n times to generate the requested number of noise tensors
        for _ in range(n):
            # Generate Perlin noise with probability p, otherwise generate zero tensors
            if random.random() <= p:
                noise, noise_mask = self.generate_perlin_noise(area_min=100)
            else:
                noise = tf.zeros((tsize, tsize, channels))
                noise_mask = tf.zeros((tsize, tsize, 1))
            
            # If the number of channels is 1, convert noise to grayscale
            if channels == 1:
                noise = tf.image.rgb_to_grayscale(noise)
                noise_mask_tmp = noise_mask
            else:
                # Convert noise mask to RGB if multiple channels
                noise_mask_tmp = tf.image.grayscale_to_rgb(noise_mask)

            # If M and N are not initialized, initialize them with the first noise and mask
            if M is None:
                M = noise_mask_tmp[tf.newaxis, ...]  # Add batch dimension
                N = noise[tf.newaxis, ...]           # Add batch dimension
            else:
                # Concatenate subsequent noise and noise masks along the batch axis
                N = tf.concat([N, noise[tf.newaxis, ...]], axis=0)
                M = tf.concat([M, noise_mask_tmp[tf.newaxis, ...]], axis=0)

            # Delete temporary variables to free memory (although in Python, this is not strictly necessary)
            del noise, noise_mask, noise_mask_tmp

        # Convert final mask tensor to grayscale
        M = tf.image.rgb_to_grayscale(M)

        return M, N

    def perlin_noise_batch(self, 
                           X: tf.Tensor, 
                           channels: int = 3, 
                           p: float = 1.0, 
                           area_min: int = 50) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate a batch of Perlin noise and apply it to the input tensor `X`.

        Args:
            X (tf.Tensor): Input tensor representing the batch of images.
            channels (int, optional): Number of channels in the images. Defaults to 3.
            p (float, optional): Probability of applying Perlin noise to each image. Defaults to 1.0.
            area_min (int, optional): Minimum area for the Perlin noise mask. Defaults to 50.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
                - X: Original input tensor.
                - Xn: Noisy version of the input tensor.
                - N: Generated noise tensor.
                - M: Noise mask tensor.
        """
        # Initialize tensors for noisy image (Xn), noise (N), and noise mask (M)
        Xn: tf.Tensor = None
        N: tf.Tensor = None
        M: tf.Tensor = None

        # Loop over the batch of images in X
        for _ in range(len(X)):
            # If random number is less than p, generate Perlin noise, otherwise generate zero tensors
            if random.random() < p:
                noise, noise_mask = self.generate_perlin_noise(area_min=area_min)
            else:
                _s = list(X.shape[1:-1])  # Shape for noise mask
                _s.append(1)
                noise, noise_mask = tf.zeros(X.shape[1:]), tf.zeros(_s)

            # Convert noise and mask based on the number of channels
            if channels == 1 and noise.shape[-1] != 1:
                noise = tf.image.rgb_to_grayscale(noise)

            if channels == 1:
                noise_mask_tmp = noise_mask
            else:
                noise_mask_tmp = tf.concat([noise_mask, noise_mask, noise_mask], axis=-1)

            # Initialize M and N tensors with the first generated noise and mask
            if M is None:
                M = noise_mask_tmp[tf.newaxis, ...]  # Add batch dimension
                N = noise[tf.newaxis, ...]           # Add batch dimension
            else:
                # Concatenate subsequent noise and noise masks along the batch axis
                N = tf.concat([N, noise[tf.newaxis, ...]], axis=0)
                M = tf.concat([M, noise_mask_tmp[tf.newaxis, ...]], axis=0)

            # Delete temporary variables to free memory
            del noise, noise_mask, noise_mask_tmp

        # Get the batch size from the input image X (assuming the first axis is the batch size)
        batch_size = tf.shape(X)[0]

        # Generate a random beta for each image in the batch, with values between 0.2 and 1.0
        betas = tf.random.uniform(
            shape=[batch_size, 1, 1, 1],  # Shape to broadcast over height, width, and channels
            minval=0.5,
            maxval=1.0,
            dtype=tf.float32
        )

        M = tf.where(M > 0.5, 1.0, 0.0)

        # Threshold the image based on the mask M (i.e., keep only the masked parts of X)
        X_i = tf.math.multiply_no_nan(X, M)
        X_o = tf.math.multiply_no_nan(X, (1.0 - M))

        # Compute the part of the image where the mask is not applied (1 - M) to get the unmasked part
        Xn = (
            X_o +  # Unmasked part of X
            tf.math.multiply_no_nan((1.0 - betas), X_i) +  # Weighted by (1 - beta) over the masked X
            tf.math.multiply_no_nan((betas), N)
        )

        # Clip the values of Xn to be within the range [0, 1] to ensure valid pixel values
        Xn = tf.clip_by_value(Xn, 0., 1.)

        # Convert M to grayscale
        if channels == 3:
            M = tf.image.rgb_to_grayscale(M)
            M = tf.where(M > 0.5, 1.0, 0.0)

        # Replace any NaN values with 0
        X = tf.where(tf.math.is_nan(X), 0., X)
        Xn = tf.where(tf.math.is_nan(Xn), 0., Xn)
        N = tf.where(tf.math.is_nan(N), 0., N)
        M = tf.where(tf.math.is_nan(M), 0., M)

        # Clip values between 0 and 1 for all tensors
        X = tf.clip_by_value(X, clip_value_min=0., clip_value_max=1.)
        Xn = tf.clip_by_value(Xn, clip_value_min=0., clip_value_max=1.)
        N = tf.clip_by_value(N, clip_value_min=0., clip_value_max=1.)
        M = tf.clip_by_value(M, clip_value_min=0., clip_value_max=1.)

        return X, Xn, N, M, betas

    def pre_generate_noise(self, epoch: int, min_area: int = 10) -> None:
        """
        Pre-generates Perlin noise perturbations at specified epochs and manages the Perlin noise array.

        Args:
            epoch (int): Current training epoch.
            min_area (int, optional): Minimum area for the Perlin noise perturbation. Defaults to 10.
        """
        # Check if Perlin noise generation is enabled and not generated each time
        if self.use_perlin_noise and not self.generate_perlin_each_time:
            # Determine if noise should be generated in the current epoch
            if (epoch == 0) or (epoch % self.perlin_generation_every_n_epochs == 0):
                # First epoch requires more noise generation
                if epoch == 0:
                    print('Generate Perlin perturbations for the first time. This could take a while...')
                    # Initialize progress bar for the first noise generation
                    with tqdm(total=self.perlin_queue_min, leave=True) as pbar:
                        for i in range(self.perlin_queue_min):
                            # Generate Perlin noise and mask until the mask area exceeds the minimum
                            m_sum: float = 0.0
                            while m_sum < float(min_area) + tf.keras.backend.epsilon():
                                noise, mask = self.perlin_perturbation()
                                m_sum = tf.reduce_sum(noise)
                            # Append noise and mask to arrays
                            self.perlin_noise_array.append(noise.numpy())
                            self.perlin_mask_array.append(mask.numpy())
                            # Free memory
                            del noise, mask
                            # Update progress bar
                            pbar.update(1)
                else:
                    print('Generate additional Perlin perturbations...')
                    # Initialize progress bar for subsequent noise generations
                    with tqdm(total=self.perlin_generate_m_perturbations, leave=True) as pbar:
                        for i in range(self.perlin_generate_m_perturbations):
                            # Generate Perlin noise and mask until the mask area exceeds the minimum
                            m_sum: float = 0.0
                            while m_sum < float(min_area) + tf.keras.backend.epsilon():
                                noise, mask = self.perlin_perturbation()
                                m_sum = tf.reduce_sum(noise)
                            # Append noise and mask to arrays
                            self.perlin_noise_array.append(noise.numpy())
                            self.perlin_mask_array.append(mask.numpy())
                            # Free memory
                            del noise, mask
                            # Update progress bar
                            pbar.update(1)

                # Remove old noise data if the noise array exceeds the maximum allowed size
                while len(self.perlin_noise_array) > self.perlin_queue_max:
                    self.perlin_noise_array.pop(0)
                    self.perlin_mask_array.pop(0)
                gc.collect()

    #endregion [perlin_noise_batch]

