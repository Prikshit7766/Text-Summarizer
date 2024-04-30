import os
from ..entity import DataValidationConfig
from ..logging import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files(self) -> bool:

        try:

            validation_status = None
            all_files  =  os.listdir(os.path.join("artifacts", "data_ingestion", "samsum"))

            for file in self.config.ALL_REQUIRED_FILLS:
                if file not in all_files:
                    logger.error(f"{file} is missing")
                    validation_status = False
                    break
                else:
                    logger.info(f"{file} is present")
                    validation_status = True

            with open(self.config.STATUS_FILE, "w") as file:
                file.write(f"validation status: {validation_status}")
            
            return validation_status
        except Exception as e:
            logger.error(e)
            return False
