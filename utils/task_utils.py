import argparse
import os
import tensorflow as tf


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-files",
        help="GCS or local paths to training data",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--num-epochs",
        help="Maximum number of training data epochs on which to train.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--train-batch-size", help="Batch size for training steps", type=int, default=32
    )
    parser.add_argument(
        "--train-shuffle-buffer-size", help="Shuffle buffer size for training steps", type=int, default=1000
    )
    parser.add_argument(
        "--eval-batch-size",
        help="Batch size for evaluation steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--export-path",
        type=str,
        help="Where to export the saved model to locally or on GCP",
    )

    parser.add_argument(
        "--eval-files",
        help="GCS or local paths to evaluation data",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--test-files", help="GCS or local paths to test data", nargs="+", required=True
    )
    # Training arguments
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for the optimizer",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="DEBUG",
        help="Set logging verbosity",
    )

    parser.add_argument(
        "--keep-prob", help="Keep probability for dropout", default=0.5, type=int
    )

    parser.add_argument(
        "--warm-start-dir",
        type=str,
        help="Where to load previously stored checkpoints for model initialization from",
    )

    parser.add_argument(
        "--model-def",
        type=str,
        help="YAML file parameterizing the model builder"
    )
    parser.add_argument(
        "--grasp-annotation-format",
        choices=["grasp_configurations", "grasp_images"],
        default="grasp_configurations",
        help="Specify if grasp_images are stored in dataset or if they should be generated from grasp_configurations",
    )

    args, unknown = parser.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf.logging.__dict__[args.verbosity] / 10)

    if unknown:
        tf.logging.warn("Unknown arguments: {}".format(unknown))

    return vars(args)
