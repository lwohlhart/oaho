# import argparse
import os
import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_spaceseplist(
    "train_files", None,
    "GCS or local paths to training data"
)
flags.DEFINE_integer(
    "num_epochs", None, "Maximum number of training data epochs on which to train."
)
flags.DEFINE_integer(
    "train_batch_size", 32, "Batch size for training steps"
)
flags.DEFINE_integer(
    "train_shuffle_buffer_size",  1000, "Shuffle buffer size for training steps"
)
flags.DEFINE_integer(
    "eval_batch_size", 32, "Batch size for evaluation steps"
)
flags.DEFINE_string(
    "export_path", '', "Where to export the saved model to locally or on GCP",
)

flags.DEFINE_spaceseplist(
    "eval_files", None, "GCS or local paths to evaluation data"
)
flags.DEFINE_spaceseplist(
    "test_files", None, "GCS or local paths to test data"
)
# Training arguments
flags.DEFINE_float(
    "learning_rate", 0.01, "Learning rate for the optimizer"
)
flags.DEFINE_float(
    "learning_rate_decay", 0.96, "Learning rate decay for the optimizer"
)
flags.DEFINE_float(
    "learning_rate_decay_steps", 1e4, "Learning rate decay steps for the optimizer"
)
flags.DEFINE_string(
    "learning_rate_decay_type", 'none', "Learning rate decay type for the optimizer"
)
flags.DEFINE_string(
    "job_dir", None, "GCS location to write checkpoints and export models"
)
flags.DEFINE_enum(
    "verbosity", "DEBUG", ["DEBUG", "ERROR", "FATAL", "INFO", "WARN"], "Set logging verbosity",
)

flags.DEFINE_float(
    "keep_prob", 0.5, "Keep probability for dropout"
)

flags.DEFINE_string(
    "warm_start_dir", None, "Where to load previously stored checkpoints for model initialization from"
)

flags.DEFINE_string(
    "model_def", 'config/model.yaml', "YAML file parameterizing the model builder"
)
flags.DEFINE_enum(
    "grasp_annotation_format", "grasp_configurations", ["grasp_configurations", "grasp_images"],
    help="Specify if grasp_images are stored in dataset or if they should be generated from grasp_configurations",
)

FLAGS = flags.FLAGS


def process_config() -> dict:
    """
    Add in any static configuration that is unlikely to change very often
    :return: a dictionary of static configuration data
    """
    config = {"exp_name": "example_model_train"}

    return config


def get_args() -> dict:
    """
    Get command line arguments add and remove any needed by your project
    :return: Namespace of command arguments
    """
    # parser = argparse.ArgumentParser(description=__doc__)


    # args, unknown = flags.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(FLAGS.verbosity)
    flags.mark_flag_as_required('train_files')

    # Set C++ Graph Execution level verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf.logging.__dict__[FLAGS.verbosity] / 10)

    # if unknown:
    #     tf.logging.warn("Unknown arguments: {}".format(unknown))

    return {}
