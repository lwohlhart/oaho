from data_loader.oaho_loader import TFRecordDataLoader
from models.oaho_model_factory import oaho_model_from_config
from trainers.oaho_train import OAHOTrainer
from utils.utils import get_args, process_config


def init() -> None:
    """
    The main function of the project used to initialise all the required classes
    used when training the model
    """
    # get input arguments
    args = get_args()
    # get static config information
    config = process_config()
    # combine both into dictionary
    config = {**config, **args}

    # initialise model
    model = oaho_model_from_config(config)
    # create your data generators for each mode
    train_data = TFRecordDataLoader(config, mode="train")

    val_data = TFRecordDataLoader(config, mode="val")

    test_data = TFRecordDataLoader(config, mode="test")

    # initialise the estimator
    trainer = OAHOTrainer(config, model, train_data, val_data, test_data)

    # start training
    trainer.run()


if __name__ == "__main__":
    init()
