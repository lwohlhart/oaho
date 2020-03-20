"""Export estimator as a saved_model"""

__author__ = "LW"

from data_loader.oaho_loader import TFRecordDataLoader
from models.oaho_model_factory import oaho_model_from_config
from utils.task_utils import get_args, process_config

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
    

if __name__ == '__main__':

    
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}

    # initialise model
    model = oaho_model_from_config(config)
    # allow memory usage to me scaled based on usage

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # get number of steps required for one pass of data

    run_config = tf.estimator.RunConfig(
        session_config=config,
    )
    # set output directory
    run_config = run_config.replace(model_dir=FLAGS.job_dir)

        
    # intialise the estimator with your model
    estimator = tf.estimator.Estimator(model_fn=model.model, config=run_config)

    features = {'depth': tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 1], name='depth')}
    if model.config['model']['input_format'] == 'rgbd':
        features.update({'rgb': tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 3], name='rgb')})
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
    # export the saved model
    estimator.export_saved_model(FLAGS.export_path, export_input_fn)
