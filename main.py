from chest_cancer_classification import logger
from chest_cancer_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from chest_cancer_classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from chest_cancer_classification.pipeline.stage_03_model_train import ModelTrainingPipeline

STAGE_NAME ="Data Ingestion stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"**************************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"

try:
    logger.info(f"**************************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e