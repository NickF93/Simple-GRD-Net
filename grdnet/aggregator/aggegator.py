import re
from typing import Optional, Callable, Union, Tuple, Any, Dict, List
from enum import Enum

import tensorflow as tf
from tqdm.auto import tqdm
import mysql.connector
import numpy as np
from sklearn.metrics import auc


LEVEL_IT: str = 'level'
CLASS_IT: str = 'class'
SCORES_IT: str = 'scores'
DICT_IT: str = 'dict'


class ScoreEnum(Enum):
    """
    Enum class for the different scores used in the aggregation process.

    The scores are used to calculate the quality of the generated images and masks.
    The available scores are:

    - `HETAMAP`: The score from the heatmap, which is the output of the segmentator.
    - `DIFF_HEATMAP`: The score from the difference between the heatmap and the mask.
    - `SSIM`: The structural similarity index between the generated image and the input image.

    The scores are used to calculate the F1-score, accuracy, precision and recall of the generated images and masks.
    """

    HEATMAP = 'heatmap'
    DIFF_HEATMAP = 'diff_heatmap'
    SSIM = 'ssim'


class MetricEnum(Enum):
    """
    Enum class for the different metrics used to evaluate the quality of the generated images and masks.

    The metrics are used to calculate the quality of the generated images and masks.
    The available metrics are:

    - `F1`: The F1-score metric, which is the harmonic mean of precision and recall.
    - `ACCURACY`: The accuracy metric, which is the proportion of correctly classified samples.
    - `PRECISION`: The precision metric, which is the proportion of true positive samples among all positive samples.
    - `RECALL`: The recall metric, which is the proportion of true positive samples among all actual positive samples.

    The metrics are used to calculate the quality of the generated images and masks.
    """

    F1 = 'f1'  # The F1-score metric
    ACCURACY = 'accuracy'  # The accuracy metric
    PRECISION = 'precision'  # The precision metric
    RECALL = 'recall'  # The recall metric


class GroupDataTypeEnum(Enum):
    """
    Enum class for the data types of the groups in the aggregation database.

    The data types are used to store the values of the groups in the database.
    The available data types are:

    - `STRING`: The string data type, used for storing strings.
    - `INT`: The integer data type, used for storing integers.

    """

    STRING = 0  # The string data type
    INT = 1    # The integer data type


class GroupTypeEnum(Enum):
    """
    Enum class representing various group types used in aggregation.

    Each group type is associated with a unique integer identifier and a data type.
    The available group types are:

    - `BATCH`: Represents a batch group type, associated with a string data type.
    - `RECIPE`: Represents a recipe group type, associated with a string data type.
    - `STATION`: Represents a station group type, associated with a string data type.
    - `ID`: Represents an ID group type, associated with a string data type.
    - `RUN`: Represents a run group type, associated with an integer data type.
    - `FRAME`: Represents a frame group type, associated with an integer data type.
    - `VIAL`: Represents a vial group type, associated with an integer data type.
    - `REGION`: Represents a region group type, associated with an integer data type.

    Attributes:
        dtype (GroupDataTypeEnum): The data type associated with the group type.
    """

    BATCH = (0, GroupDataTypeEnum.STRING)
    RECIPE = (1, GroupDataTypeEnum.STRING)
    STATION = (2, GroupDataTypeEnum.STRING)
    ID = (3, GroupDataTypeEnum.STRING)
    RUN = (4, GroupDataTypeEnum.INT)
    FRAME = (5, GroupDataTypeEnum.INT)
    VIAL = (6, GroupDataTypeEnum.INT)
    REGION = (7, GroupDataTypeEnum.INT)
    PATCH = (8, GroupDataTypeEnum.INT)

    def __new__(cls, value: int, dtype: GroupDataTypeEnum):
        """
        Create a new instance of the GroupTypeEnum.

        Parameters:
            value (int): The unique integer identifier for the group type.
            dtype (GroupDataTypeEnum): The data type associated with the group type.

        Returns:
            GroupTypeEnum: An instance of the GroupTypeEnum with the specified value and data type.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj.dtype = dtype
        return obj


class Aggregator:
    def __init__(
        self,
        test_dataset: tf.data.Dataset,
        patch_size: int,
        patches: Tuple[int, int],
        strides: Optional[Tuple[int, int]],
        autoencoder_model: tf.keras.models.Model,
        segmentator_model: tf.keras.models.Model,
        contextual_loss_fn: Callable,
        segmentator_loss_fn: Callable,
        regular_expression: str,
        group_descriptor: Dict[GroupTypeEnum, Union[int, str]],
        db_aggregate_levels: Union[List[GroupTypeEnum], Tuple[GroupTypeEnum, ...]],
        metric_aggregate_level: GroupTypeEnum,
        score: ScoreEnum,
        metric: MetricEnum,
        blur_kernel_size: int,
        database_name: str,
        run_table_name: str,
        database_host: str,
        database_user: str,
        database_pwd: str,
        writer: Optional[Any] = None,
    ):
        """
        Initialize an Aggregator object.

        Parameters
        ----------
        test_dataset : tf.data.Dataset
            The test dataset.
        autoencoder_model : tf.keras.models.Model
            The autoencoder model.
        segmentator_model : tf.keras.models.Model
            The segmentator model.
        contextual_loss_fn : Callable
            The contextual loss function.
        segmentator_loss_fn : Callable
            The segmentator loss function.
        regular_expression : str
            The regular expression used to extract the file name from the image path.
        group_descriptor : Dict[GroupTypeEnum, Union[int, str]]
            A dictionary describing the group level for each of the aggregate levels.
        db_aggregate_levels : Union[List[GroupTypeEnum], Tuple[GroupTypeEnum, ...]]
            The aggregate levels for the database.
        metric_aggregate_levels : GroupTypeEnum
            The aggregate level for the metric values.
        metric : str
            The name of the metric.
        blur_kernel_size : int
            The size of the blur kernel for the contextual loss.
        database_name : str
            The name of the database.
        run_table_name : str
            The name of the table containing the run information.
        database_host : str
            The host name of the database.
        database_user : str
            The username for the database.
        database_pwd : str
            The password for the database.
        writer : Optional[Any], optional
            The writer to use for logging. Defaults to None.
        """
        self.test_dataset = test_dataset
        self.autoencoder_model = autoencoder_model
        self.segmentator_model = segmentator_model
        self.contextual_loss_fn = contextual_loss_fn
        self.segmentator_loss_fn = segmentator_loss_fn
        self.regular_expression = regular_expression
        self.group_descriptor = group_descriptor
        self.db_aggregate_levels = db_aggregate_levels
        self.metric_aggregate_level = metric_aggregate_level
        self.score = score
        self.metric = metric
        self.blur_kernel_size = blur_kernel_size
        self.database_name = database_name
        self.run_table_name = str(re.sub(r'\W+', '_', str(run_table_name).strip()))
        self.writer = writer

        self.patch_size = patch_size
        self.patches = patches
        self.strides = strides

        image_size = (
            self.patch_size + (self.patches[0] - 1) * self.strides[0],
            self.patch_size + (self.patches[1] - 1) * self.strides[1]
        )
        self.image_size = image_size

        self.best_metric_value: Optional[float] = None

        self.regex = re.compile(self.regular_expression)

        self.mydb = mysql.connector.connect(
            host=database_host,
            user=database_user,
            password=database_pwd,
            database=self.database_name
        )
        self.mycursor = self.mydb.cursor()

        self.database_table_built: bool = False

        self.total_test_steps = 0


    def build_database(
        self,
        verbose: bool = False
    ):
        # Map enum data types to SQL types
        type_mapping = {
            GroupDataTypeEnum.STRING: "VARCHAR(256)",
            GroupDataTypeEnum.INT: "BIGINT UNSIGNED"
        }

        create_table_sql = f"""
CREATE TABLE `{self.database_name}`.`{self.run_table_name}` (
    ID                              BIGINT UNSIGNED AUTO_INCREMENT              NOT NULL,
    epoch                           BIGINT UNSIGNED                             NOT NULL,
    aggregation_level               VARCHAR(256)                                NOT NULL,"""

        # Add each field from the GroupTypeEnum
        for item in GroupTypeEnum:
            sql_type = type_mapping[item.dtype]
            create_table_sql += f"\n    product_{item.name.lower()}           {sql_type} {'NOT' if item.value == 0 else 'DEFAULT'} NULL,"
            create_table_sql += f"\n    product_{item.name.lower()}_class     {sql_type} DEFAULT NULL,"

        create_table_sql += f"""
    heatmap_score                   REAL                                        NOT NULL,
    diff_heatmap_score              REAL                                        NOT NULL,
    ssim_score                      REAL                                        NOT NULL,
    file_name                       VARCHAR(1024)                  DEFAULT ''   NOT NULL,
    result_file_name                VARCHAR(1024)                  DEFAULT ''   NOT NULL,

    CONSTRAINT {self.run_table_name}_PK_ID PRIMARY KEY  (ID)
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8
COLLATE=utf8_general_ci
"""
        index_id            = "CREATE INDEX " + self.run_table_name + "_ID_IDX                   USING BTREE ON `" + self.database_name + "`.`" + self.run_table_name + "` (ID)"
        index_epoch         = "CREATE INDEX " + self.run_table_name + "_product_epoch_IDX        USING BTREE ON `" + self.database_name + "`.`" + self.run_table_name + "` (epoch)"
        index_product_id    = "CREATE INDEX " + self.run_table_name + "_product_id_IDX           USING BTREE ON `" + self.database_name + "`.`" + self.run_table_name + "` (product_id)"
        
        if verbose:
            print(create_table_sql)
            print(index_id)
            print(index_epoch)
            print(index_product_id)

        self.mycursor.execute(create_table_sql)
        self.mycursor.execute(index_id)
        self.mycursor.execute(index_epoch)
        self.mycursor.execute(index_product_id)

        self.mydb.commit()
    
    
    @tf.function(autograph=True, reduce_retracing=True)
    def test_step(
        self,
        x: tf.Tensor,
        m: tf.Tensor,
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        # Generator step: Z, Xf, Zf = Generator(X_Noise)
        x_rebuilt = self.autoencoder_model(x, training=False)

        # Segmentator step: Map = Segmentator(X concat X_Rebuilt)
        m_rebuilt = self.segmentator_model(tf.concat((x, x_rebuilt), axis=-1), training=False)

        # Calculate loss between X_Rebuilt = Model(X_Noise) and X (Original images)
        contextual_loss_value = self.contextual_loss_fn(x, x_rebuilt)
        autoencoder_loss_value = 10.0 * contextual_loss_value

        segmentator_loss_value = self.segmentator_loss_fn(m, m_rebuilt)

        return {
            'images' : {
                'X' : x,
                'Xf' : x_rebuilt,
                'M': m,
                'Mf': m_rebuilt,
            },
            'losses' : {
                'contextual_loss' : contextual_loss_value,
                'autoencoder_loss_value' : autoencoder_loss_value,
                'segmentator_loss' : segmentator_loss_value,
            }
        }


    def test_loop(
        self,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict]:
        if not self.database_table_built:
            self.build_database(verbose=True)
            self.database_table_built = True

        test_images = self.test_dataset

        loss_acc: Dict[str, float] = {}

        images_num: int = 0

        # ----- -> ------ -> ------- -> C  -> --- -> ----- -> C    -> C
        # BATCH -> RECIPE -> STATION -> ID -> RUN -> FRAME -> VIAL -> REGION
        definition_dict: Dict = {}

        with tqdm(test_images, desc='Test', leave=True, unit='batch') as pbar:
            for step, (x, m, path) in enumerate(pbar):
                x, m, path, patch_index = self.extract_patches(x, m, path)

                actual_batch_size = int(x.shape[0])
                images_num += actual_batch_size
                m = tf.where(m > 0.5, 1.0, 0.0)

                # The class of the image referring to ground truth (0 => BAD, 1 => GOOD)
                x_class = tf.cast(
                    tf.where(
                        tf.reduce_max(m, axis=list(range(1, len(m.shape)))) > 0.5,
                        0, 1
                    ), dtype=tf.uint8
                )

                dict_values = self.test_step(
                    x=x,
                    m=m,
                )

                x_rebuilt = dict_values['images']['Xf']  # Shape: [BATCH, HEIGHT, WIDTH, CHANNEL]
                m_rebuilt = dict_values['images']['Mf']  # Shape: [BATCH, HEIGHT, WIDTH, 1 = # CHANNEL OF A HEATMAP]

                # CALCULATING SCORE USING THE SEGMENTATOR HETATMAP RESULT
                if self.blur_kernel_size > 0:
                    heatmap_blur = tf.nn.avg_pool(m_rebuilt, ksize=self.blur_kernel_size, strides=1, padding='SAME')
                else:
                    heatmap_blur = m_rebuilt
                heatmap_score = tf.reduce_max(heatmap_blur, axis=list(range(1, len(heatmap_blur.shape))))

                # CALCULATING SCORE USING THE DIFFERENCE BETWEEN X AND X' (x_rebuilt)
                diff_heatmap = tf.math.abs(x - x_rebuilt)
                if diff_heatmap.shape[-1] == 3:
                    diff_heatmap = tf.image.rgb_to_grayscale(diff_heatmap)

                if self.blur_kernel_size > 0:
                    diff_heatmap_blur = tf.nn.avg_pool(diff_heatmap, ksize=self.blur_kernel_size, strides=1, padding='SAME')
                else:
                    diff_heatmap_blur = diff_heatmap
                diff_heatmap_score = tf.reduce_max(diff_heatmap_blur, axis=list(range(1, len(diff_heatmap_blur.shape))))
                
                # CALCULATING SCORE USING THE SSIM SCORE BETWEEN X AND X'
                channels = int(x.shape[-1])
                if channels == 1:
                    # Structural Similarity Index (SSIM) for grayscale images
                    score_fn = lambda y_true, y_pred: (1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0))  # mean(1 - SSIM(x_original, x_rebuilt))
                else:
                    # SSIM for RGB images with three channels, averaged across channels
                    score_fn = lambda y_true, y_pred: (1.0 - ((
                            tf.image.ssim(y_true[..., 0:1], y_pred[..., 0:1], max_val=1.0) +
                            tf.image.ssim(y_true[..., 1:2], y_pred[..., 1:2], max_val=1.0) +
                            tf.image.ssim(y_true[..., 2:3], y_pred[..., 2:3], max_val=1.0)
                        ) / 3.0))
                ssim_score = score_fn(x, x_rebuilt)

                path_batch_numpy = path.numpy()
                x_batch_class_numpy = x_class.numpy()
                heatmap_score_batch_numpy = heatmap_score.numpy()
                diff_heatmap_score_batch_numpy = diff_heatmap_score.numpy()
                ssim_score_batch_numpy = ssim_score.numpy()

                # Test that paths, gt classes and scores have the same length
                assert len(path_batch_numpy) == len(x_batch_class_numpy)
                assert len(path_batch_numpy) == len(heatmap_score_batch_numpy)
                assert len(path_batch_numpy) == len(diff_heatmap_score_batch_numpy)
                assert len(path_batch_numpy) == len(ssim_score_batch_numpy)

                for i, path_enc in enumerate(path_batch_numpy):
                    path_str = path_enc.decode('utf-8')
                    patch_index_np = int(patch_index.numpy()[i])
                    x_class_np = x_batch_class_numpy[i]
                    heatmap_score_np = heatmap_score_batch_numpy[i]
                    diff_heatmap_score_np = diff_heatmap_score_batch_numpy[i]
                    ssim_score_np = ssim_score_batch_numpy[i]

                    match = re.search(self.regex, path_str)

                    # For each Group Type ordered by its value
                    val: Union[int, str]
                    act_dict = definition_dict
                    types_len = len(GroupTypeEnum)
                    for idx, item in enumerate(sorted(GroupTypeEnum, key=lambda x: x.value)):
                        if item != GroupTypeEnum.PATCH:
                            group_val = self.group_descriptor[item]
                            if isinstance(group_val, int):
                                extracted_val = str(match.group(group_val))
                            elif isinstance(group_val, str):
                                extracted_val = group_val
                            else:
                                raise ValueError('Group val must be int (group position) or str (default value)')
                        else:
                            extracted_val = int(patch_index_np)

                        if item.dtype == GroupDataTypeEnum.STRING:
                            val = str(extracted_val)
                        elif item.dtype == GroupDataTypeEnum.INT:
                            val = int(extracted_val)
                        else:
                            raise ValueError('Group val must be int (group position) or str (default value)')
                        
                        if val not in act_dict:
                            if idx < types_len - 1:
                                act_dict[val] = {
                                    LEVEL_IT : item,
                                    CLASS_IT : 1,
                                    SCORES_IT: [0., 0., 0.],
                                    DICT_IT: {},
                                }
                            else:
                                act_dict[val] = {
                                    LEVEL_IT : item,
                                    CLASS_IT : 1,
                                    SCORES_IT: [0., 0., 0.],
                                }
                        act_dict[val][CLASS_IT] = int(min(act_dict[val][CLASS_IT], int(x_class_np)))
                        act_dict[val][SCORES_IT][0] = float(max(act_dict[val][SCORES_IT][0], float(heatmap_score_np)))
                        act_dict[val][SCORES_IT][1] = float(max(act_dict[val][SCORES_IT][1], float(diff_heatmap_score_np)))
                        act_dict[val][SCORES_IT][2] = float(max(act_dict[val][SCORES_IT][2], float(ssim_score_np)))

                        if idx < types_len - 1:
                            act_dict = act_dict[val][DICT_IT]

                losses_dict = dict_values['losses']
                for key, value in losses_dict.items():
                    if key in loss_acc:
                        loss_acc[key] += value.numpy().item() * float(actual_batch_size)
                    else:
                        loss_acc[key] = value.numpy().item() * float(actual_batch_size)
                
                losses_postfix_dict = {key: f'{value.numpy().item():.4f}' for key, value in losses_dict.items()}

                self.total_test_steps += 1

                if self.writer is not None:
                    with self.writer.as_default():
                        for key, value in losses_dict.items():
                            tf.summary.scalar(f'TEST: {key}', value, step=self.total_test_steps)

                if self.writer is not None:
                    with self.writer.as_default():
                            tf.summary.image(f"T.a TEST: x BATCH: {step}", x, max_outputs=int(len(x)), step=epoch)
                            tf.summary.image(f"T.b TEST: m BATCH: {step}", m, max_outputs=int(len(m)), step=epoch)
                            tf.summary.image(f"T.f TEST: x_rebuilt BATCH: {step}", x_rebuilt, max_outputs=int(len(x_rebuilt)), step=epoch)
                            tf.summary.image(f"T.g TEST: m_rebuilt BATCH: {step}", m_rebuilt, max_outputs=int(len(m_rebuilt)), step=epoch)
                            tf.summary.image(f"T.g TEST: heatmap_blur BATCH: {step}", heatmap_blur, max_outputs=int(len(heatmap_blur)), step=epoch)
                            tf.summary.image(f"T.g TEST: diff_heatmap_blur BATCH: {step}", diff_heatmap_blur, max_outputs=int(len(diff_heatmap_blur)), step=epoch)

                pbar.set_postfix(losses_postfix_dict)
        
        losses_mean_dict = {}
        for key, value in loss_acc.items():
            losses_mean_dict[key] = value / float(images_num)

        return losses_mean_dict, definition_dict


    @tf.function(autograph=True, reduce_retracing=True)
    def extract_patches(
        self,
        x: tf.Tensor,
        mask: tf.Tensor,
        paths: tf.Tensor
    ) -> tf.Tensor:
        # Extract patches
        # For Images
        """
        Extracts patches from the input tensor.

        Args:
        x (tf.Tensor): The input tensor to extract patches from.
        mask (tf.Tensor, optional): The mask tensor to extract patches from. Defaults to None.

        Returns:
        tf.Tensor: The extracted patches.
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
        px = tf.image.extract_patches(
            images=x,
            sizes=sizes,
            strides=strides,
            rates=rates,
            padding=padding
        )
        px = tf.reshape(px, [-1, self.patch_size, self.patch_size, tf.shape(x)[-1]])

        pmask = tf.image.extract_patches(
            images=mask,
            sizes=sizes,
            strides=strides,
            rates=rates,
            padding=padding
        )
        pmask = tf.reshape(pmask, [-1, self.patch_size, self.patch_size, 1])

        num_patches_per_image = tf.shape(px)[0] // tf.shape(x)[0]  # Number of patches per image
        ppaths = tf.repeat(paths, num_patches_per_image)

        # Generate patch indices
        indices = tf.range(num_patches_per_image, dtype=tf.int32)
        patch_indices = tf.tile(indices, [tf.shape(x)[0]])

        return px, pmask, ppaths, patch_indices


    def aggregate(
        self,
        epoch: int
    ):
        losses_mean_dict, definition_dict = self.test_loop(
            epoch=epoch
        )

        for key, value in losses_mean_dict.items():
            print(f'Test {key} over batches: {value}')

        level_list: List[GroupTypeEnum] = []
        def insert_query(
            _definition_dict: Dict,
            _level_list: List[GroupTypeEnum],
            _query: str,
            _val_p: str,
            _val_s: str,
            _cursor: mysql.connector.cursor.MySQLCursor,
            _ground_truth_list: List[int],
            _scores_list: List[List[float]],
        ):
            for key, value in _definition_dict.items():
                level: GroupTypeEnum = value[LEVEL_IT]
                class_num = value[CLASS_IT]
                heatmap_score = value[SCORES_IT][0]
                diff_heatmap_score = value[SCORES_IT][1]
                ssim_score = value[SCORES_IT][2]
                
                definition_dict: Optional[Dict] = None
                if DICT_IT in value:
                    definition_dict = value[DICT_IT]

                _new_level_list = _level_list + [level]

                _query_1 = _query + f'`product_{level.name.lower()}`, `product_{level.name.lower()}_class`, '
                _val_s_1 = _val_s + f"'{key}', '{class_num}', "

                _full_query = f'{_query_1}`heatmap_score`, `diff_heatmap_score`, `ssim_score`, `file_name`, `result_file_name`)'
                _full_value = f"{_val_p}{level.name}', {_val_s_1}'{heatmap_score}', '{diff_heatmap_score}', '{ssim_score}', '', '')"

                full_sql = f'{_full_query} {_full_value}'

                if level in self.db_aggregate_levels:
                    _cursor.execute(full_sql)
                
                if level == self.metric_aggregate_level:
                    _ground_truth_list.append(class_num)
                    _scores_list.append([heatmap_score, diff_heatmap_score, ssim_score])

                if definition_dict is not None:
                    insert_query(
                        _definition_dict=definition_dict,
                        _level_list=_new_level_list,
                        _query=_query_1,
                        _val_p=_val_p,
                        _val_s=_val_s_1,
                        _cursor=_cursor,
                        _ground_truth_list=_ground_truth_list,
                        _scores_list=_scores_list,
                    )

        ground_truth_list: List[int] = []
        scores_list: List[float] = []

        query = f'INSERT INTO   `{self.run_table_name}` (`epoch`, `aggregation_level`, '
        val_p = f"VALUES                                ({epoch}, '"
        val_s = ''
        insert_query(
            _definition_dict=definition_dict,
            _level_list=level_list,
            _query=query,
            _val_p=val_p,
            _val_s=val_s,
            _cursor=self.mycursor,
            _ground_truth_list=ground_truth_list,
            _scores_list=scores_list,
        )
        # Check that every sub-list in scores_list is of length 3 (heatmap score, diff heatmap score, ssim score)
        self.mydb.commit()
    
        assert any([len(l) == 3 for l in scores_list])
        heatmap_scores = [l[0] for l in scores_list]
        diff_heatmap_scores = [l[1] for l in scores_list]
        ssim_scores = [l[2] for l in scores_list]
        assert len(ground_truth_list) == len(heatmap_scores), f'len(ground_truth_list) = {len(ground_truth_list)}, len(heatmap_scores) = {len(heatmap_scores)}'
        assert len(ground_truth_list) == len(diff_heatmap_scores), f'len(ground_truth_list) = {len(ground_truth_list)}, len(diff_heatmap_scores) = {len(diff_heatmap_scores)}'
        assert len(ground_truth_list) == len(ssim_scores), f'len(ground_truth_list) = {len(ground_truth_list)}, len(ssim_scores) = {len(ssim_scores)}'

        ground_truth_list = np.asarray(ground_truth_list)
        heatmap_scores = np.asarray(heatmap_scores)
        diff_heatmap_scores = np.asarray(diff_heatmap_scores)
        ssim_scores = np.asarray(ssim_scores)

        def ths_curve(y_true, y_score):
            """
            Computes precision, recall, F1 scores, accuracy, AP, AUROC, TP, TN, FP, FN, and thresholds.

            Parameters:
            - y_true: array-like of shape (n_samples,)
                    True binary labels (0 or 1).
            - y_score: array-like of shape (n_samples,)
                    Scores assigned by the classifier.

            Returns:
            - precision: Precision values at different thresholds.
            - recall: Recall values at different thresholds.
            - f1_scores: F1 scores at different thresholds.
            - accuracy: Accuracy at different thresholds.
            - ap: Average Precision
            - auroc: Area Under the ROC Curve
            - tp: True Positives at different thresholds
            - tn: True Negatives at different thresholds
            - fp: False Positives at different thresholds
            - fn: False Negatives at different thresholds
            - thresholds: Thresholds at which metrics were calculated.
            """
            # Sort scores and corresponding true labels by score in descending order
            desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
            y_score = y_score[desc_score_indices]
            y_true = y_true[desc_score_indices]

            # Identify thresholds where score changes
            distinct_value_indices = np.where(np.diff(y_score))[0]
            threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
            thresholds = y_score[threshold_idxs]

            # True Positives and False Positives at each threshold
            tps = np.cumsum(y_true)[threshold_idxs]
            fps = np.cumsum(1 - y_true)[threshold_idxs]
            tp_total = tps[-1]  # Total positives
            fp_total = fps[-1]  # Total negatives

            # False Negatives and True Negatives at each threshold
            fns = tp_total - tps
            tns = fp_total - fps

            sl = slice(None, None, -1)
            tps = tps[sl]
            fps = fps[sl]
            fns = fns[sl]
            tns = tns[sl]
            thresholds = thresholds[sl]

            # Calculate precision and recall
            precision = tps / (tps + fps)
            recall = tps / tp_total

            # F1 Score
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)  # Handle NaN

            # Accuracy
            accuracy = (tps + tns) / y_true.size

            # Average Precision (AP)
            ap = auc(recall, precision)

            # AUROC (Area Under the ROC Curve)
            tpr = recall  # True Positive Rate
            fpr = fps / fp_total  # False Positive Rate
            auroc = auc(fpr, tpr)

            return precision, recall, f1_scores, accuracy, ap, auroc, tps, tns, fps, fns, thresholds

        scores = {
            ScoreEnum.HEATMAP: heatmap_scores,
            ScoreEnum.DIFF_HEATMAP: diff_heatmap_scores,
            ScoreEnum.SSIM: ssim_scores
        }

        full_metric_dict: Dict[str, Dict[str, Union[int, float]]] = {}
        y_true = ground_truth_list
        for y_score_name, y_score in scores.items():
            precision_scores, recall_scores, f1_scores, accuracy_scores, ap_score, auroc_score, tps, tns, fps, fns, thresholds = ths_curve(y_true=(1 - y_true), y_score=y_score)
            
            if self.metric == MetricEnum.F1:
                metric_value = f1_scores
            elif self.metric == MetricEnum.PRECISION:
                metric_value = precision_scores
            elif self.metric == MetricEnum.RECALL:
                metric_value = recall_scores
            elif self.metric == MetricEnum.ACCURACY:
                metric_value = accuracy_scores
            else:
                raise ValueError(f'Invalid metric: {self.metric}')
            
            best_index = np.argmax(metric_value)
            best_value = float(metric_value[best_index])
            precision = float(precision_scores[best_index])
            recall = float(recall_scores[best_index])
            f1 = float(f1_scores[best_index])
            accuracy = float(accuracy_scores[best_index])
            ap = float(ap_score)
            auroc = float(auroc_score)
            tp = int(tps[best_index])
            tn = int(tns[best_index])
            fp = int(fps[best_index])
            fn = int(fns[best_index])
            threshold = float(thresholds[best_index])

            if y_score_name == self.score:
                self.best_metric_value = best_value

            print()
            print('#' * 10)
            print(f'SCORE: {y_score_name}')
            print(f'Best metric criteria: {self.metric.name}: {best_value}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            print(f'Accuracy: {accuracy}')
            print(f'AP: {ap}')
            print(f'AUROC: {auroc}')
            print(f'Threshold: {threshold}')
            print('CONFUSION MATRIX:')
            print(f'TP: {tp}\tFP: {fp}')
            print(f'FN: {fn}\tTN: {tn}')

            metric_dict: Dict[str, Union[int, float]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'ap': ap,
                'auroc': auroc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'threshold': threshold
            }
            full_metric_dict[y_score_name] = metric_dict
        
        for key, value in losses_mean_dict.items():
            tf.summary.scalar(f'TEST AVERAGE LOSS: {key}', value, step=epoch)
        
        for key, dict_ in full_metric_dict.items():
            for key2, value in dict_.items():
                tf.summary.scalar(f'TEST AVERAGE METRIC: {key} - {key2}', value, step=epoch)

        return full_metric_dict, self.best_metric_value
