from base.data_loader import DataLoader
from utils.grasp import Grasp
import tensorflow as tf
import multiprocessing
from typing import Tuple, Dict
import random
import numpy as np

def draw_grasp_images(grasps, shape):
    q = np.zeros(shape, np.float32)
    a = np.zeros(shape, np.float32)
    w = np.zeros(shape, np.float32)

    for g in grasps:
        if not (g[0:2].numpy() < 0.0).any():
            grasp = Grasp((g[1],g[0]), g[2], g[3]/3.0, g[3]/2.0)
            rr, cc = grasp.as_bb.polygon_coords(shape)
            q[rr, cc] = 1.0
            a[rr, cc] = g[2]
            w[rr, cc] = g[3]

    a_sin = np.sin(2*a)
    a_cos = np.cos(2*a)
    return q, a_sin, a_cos, w

class TFRecordDataLoader(DataLoader):
    def __init__(self, config: dict, mode: str) -> None:
        """
        An example of how to create a dataset using tfrecords inputs
        :param config: global configuration
        :param mode: current training mode (train, test, predict)
        """
        super().__init__(config, mode)

        # Get a list of files in case you are using multiple tfrecords
        if self.mode == 'train':
            self.file_names = self.config['train_files']
            self.batch_size = self.config['train_batch_size']
        elif self.mode == 'val':
            self.file_names = self.config['eval_files']
            self.batch_size = self.config['eval_batch_size']
        else:
            self.file_names = self.config['test_files']

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a tf.Dataset using tfrecords as inputs, use parallel
        loading and augmentation using the CPU to
        reduce bottle necking of operations on the GPU
        :return: a Dataset function
        """
        dataset = tf.data.TFRecordDataset(self.file_names)
        # create a parallel parsing function based on number of cpu cores
        dataset = dataset.map(
            map_func=self._parse_example, num_parallel_calls=multiprocessing.cpu_count()
        )

        # only shuffle training data
        if self.mode == 'train':
            # shuffles and repeats a Dataset returning a new permutation for each epoch. with serialised compatibility
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(
                    buffer_size=self.config['train_shuffle_buffer_size']
                )
            )
        else:
            dataset = dataset.repeat(self.config['num_epochs'])
        # create batches of data
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=10*self.batch_size)
        # dataset = dataset.map_and_batch(lambda x: self._parse_example(x), self.batch_size)
        return dataset

    def _parse_example(
        self, example: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Used to read in a single example from a tf record file and do any augmentations necessary
        :param example: the tfrecord for to read the data from
        :return: a parsed input example and its respective label
        """
        # do parsing on the cpu
        with tf.device("/cpu:0"):
            # define input shapes

            features = {
                'id': tf.io.FixedLenFeature([], tf.string),
                'depth': tf.io.FixedLenFeature((640*480), tf.float32),
                'segmentation': tf.io.FixedLenFeature((640*480), tf.int64),
                'width': tf.io.FixedLenFeature((), tf.int64),
                'height': tf.io.FixedLenFeature((), tf.int64),
                'grasps': tf.io.VarLenFeature(dtype=tf.float32)
            }
            if self.config['grasp_annotation_format'] == 'grasp_images':
                features.update({
                    'quality': tf.io.FixedLenFeature((640*480), tf.float32),
                    'angle_sin': tf.io.FixedLenFeature((640*480), tf.float32),
                    'angle_cos': tf.io.FixedLenFeature((640*480), tf.float32),
                    'gripper_width': tf.io.FixedLenFeature((640*480), tf.float32)
                })
            parsed_features = tf.io.parse_single_example(example, features)
            #grasps = tf.reshape(tf.sparse_tensor_to_dense(parsed_features['grasps']), (None,4))

            grasps = parsed_features['grasps']

            # grasps = tf.ragged.from_sparse(parsed_features['grasps'])
            w, h = parsed_features['width'], parsed_features['height']
            dim = (h,w,1)


            depth = tf.reshape(parsed_features['depth'], dim)
            seg = tf.reshape(parsed_features['segmentation'], dim)

            if self.config['grasp_annotation_format'] == 'grasp_images':
                quality =  tf.reshape(parsed_features['quality'], dim)
                angle_sin =  tf.reshape(parsed_features['angle_sin'], dim)
                angle_cos =  tf.reshape(parsed_features['angle_cos'], dim)
                gripper_width =  tf.reshape(parsed_features['gripper_width'], dim)
            elif self.config['grasp_annotation_format'] == 'grasp_configurations':
                grasps_list = tf.reshape(tf.sparse_tensor_to_dense (grasps, -1), (-1, 4))
                quality, angle_sin, angle_cos, gripper_width = tf.py_function(
                    draw_grasp_images, [grasps_list, dim], [tf.float32, tf.float32, tf.float32, tf.float32])

            feature_dict = {
                'input': depth
            }
            target_dict = {
                'seg': seg,
                'quality': quality,
                'angle_sin': angle_sin,
                'angle_cos': angle_cos,
                'gripper_width': gripper_width,
                'grasps': grasps ,
                'id': parsed_features['id']
            }
            return feature_dict, target_dict

    #@staticmethod
    #def _augment(example: tf.Tensor) -> tf.Tensor:
    #    """
    #    Randomly augment the input image to try improve training variance
    #    :param example: parsed input example
    #    :return: the same input example but possibly augmented
    #    """
    #    # random rotation
    #    if random.uniform(0, 1) > 0.5:
    #        example = tf.contrib.image.rotate(
    #            example, tf.random_uniform((), minval=-0.2, maxval=0.2)
    #        )
    #    # random noise
    #    if random.uniform(0, 1) > 0.5:
    #        # assumes values are normalised between 0 and 1
    #        noise = tf.random_normal(
    #            shape=tf.shape(example), mean=0.0, stddev=0.2, dtype=tf.float32
    #        )
    #        example = example + noise
    #        example = tf.clip_by_value(example, 0.0, 1.0)
    #        # random flip
    #        example = tf.image.random_flip_up_down(example)
    #    return tf.image.random_flip_left_right(example)
    #

    def __len__(self) -> int:
        """
        Get number of records in the dataset
        :return: number of samples in all tfrecord files
        """
        return sum(
            1 for fn in self.file_names for _ in tf.python_io.tf_record_iterator(fn)
        )
