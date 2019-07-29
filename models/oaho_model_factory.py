from typing import Tuple, Dict
import os
import yaml
import tensorflow as tf
from models.oaho_model_deeplab import OAHOModelDeeplab
from models.oaho_model import OAHOModel
from models.oaho_model_fcn import OAHOModelFCN
FLAGS = tf.app.flags.FLAGS


def oaho_model_from_config(config: Dict) -> OAHOModel:
    
    model_cfg = {
        'architecture': 'blabla'
    }    
    if FLAGS.model_def:
        if os.path.exists(FLAGS.model_def):
            with open(FLAGS.model_def, 'r') as f:
                model_cfg.update(yaml.full_load(f))
        else:
            tf.logging.warn('Model def file {} doesn\'t exist'.format(FLAGS.model_def))
    
    config['model'] = model_cfg
    
    if model_cfg['architecture'] == 'resnet-deconv':
        return OAHOModel(config)
    elif model_cfg['architecture'] == 'fcn':
        return OAHOModelFCN(config)
    elif model_cfg['architecture'] == 'deeplab':
        return OAHOModelDeeplab(config)
    else:
        raise Exception('Unknown model architecture {}'.format(model_cfg['architecture']))