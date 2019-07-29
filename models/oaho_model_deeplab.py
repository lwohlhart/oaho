import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import applications as ka
from models.oaho_model import OAHOModel
from typing import Dict, Tuple

import sys
sys.path.append('tensorflow_models/research')
sys.path.append('tensorflow_models/research/slim')

import deeplab.common as dlc
import deeplab.model as dlm
from deeplab.core import utils

class OAHOModelDeeplab(OAHOModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

        # config['model_variant'] = 'mobilenet_v2'
        # config['model_variant'] = 'xception_65'
        config['model_variant'] = 'resnet_v1_50'

        tf.flags.FLAGS.model_variant = config['model_variant']

        if 'resnet' in config['model_variant'] or 'xception' in config['model_variant']:
            tf.flags.FLAGS.decoder_output_stride = [4]
            atrous_rates = [6, 12, 18]
        elif 'mobilenet' in config['model_variant']:
            tf.flags.FLAGS.decoder_output_stride = None
            atrous_rates = None


        self.model_options = dlc.ModelOptions(4, atrous_rates=atrous_rates, crop_size=[480,640])


    def _create_model(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """

        tf.logging.info("Constructing OAHO Model Deeplab")

        model_options = self.model_options
        reuse = None
        weight_decay=0.0001
        fine_tune_batch_norm=is_training # False
        nas_training_hyper_parameters = None

        input_tensor = x
        x = tf.keras.layers.concatenate([x,x,x])
        features_encoder, endpoints = dlm.extract_features(x, model_options,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            nas_training_hyper_parameters=nas_training_hyper_parameters)
        features_decoder = dlm.refine_by_decoder(features_encoder,
            endpoints,
            crop_size=model_options.crop_size,
            decoder_output_stride=model_options.decoder_output_stride,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            use_bounded_activation=model_options.use_bounded_activation
            )


        x = utils.resize_bilinear(features_decoder, [480,640], features_decoder.dtype)


        # ===================================================================================================
        # Output layers
        seg_head = kl.Conv2D(32, kernel_size=2, padding='same', name='seg_head_1', activation='relu')(x)
        seg_output = kl.Conv2D(4, kernel_size=2, padding='same', name='seg_out')(seg_head)

        grasp_head = kl.Conv2D(32, kernel_size=2, padding='same', name='grasp_head_1', activation='relu')(x)
        pos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='pos_out')(grasp_head)
        cos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='cos_out')(grasp_head)
        sin_output = kl.Conv2D(1, kernel_size=2, padding='same', name='sin_out')(grasp_head)
        width_output = kl.Conv2D(1, kernel_size=2, padding='same', name='width_out')(grasp_head)
        return seg_output, pos_output, cos_output, sin_output, width_output