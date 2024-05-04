
from ..entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from ..constants import *
from ..utils.common import read_yaml, create_directories




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILLS=config.ALL_REQUIRED_FILLS
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_validation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name
        )


        return data_validation_config
    
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_ckpt = config.model_ckpt,
            output_dir = params.output_dir,
            evaluation_strategy = params.evaluation_strategy,
            learning_rate = params.learning_rate,
            num_train_epochs = params.num_train_epochs,
            per_device_train_batch_size = params.per_device_train_batch_size,
            per_device_eval_batch_size = params.per_device_eval_batch_size,
            warmup_steps = params.warmup_steps,
            weight_decay = params.weight_decay,
            save_total_limit = params.save_total_limit,
            gradient_accumulation_steps = params.gradient_accumulation_steps,
            logging_dir = params.logging_dir,
            save_strategy = params.save_strategy,
            load_best_model_at_end = params.load_best_model_at_end,
            metric_for_best_model = params.metric_for_best_model,
            greater_is_better = params.greater_is_better,
            fp16 = params.fp16
        )

        return model_trainer_config