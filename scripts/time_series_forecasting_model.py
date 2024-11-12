import logging , os
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_test_split(df , train_size=0.8):
    logger.info("spliting the data into the train and test df")
    try:
        split_index = int(len(df) * train_size)
        train_data , test_data = df[:split_index],df[split_index:]
        return train_data ,test_data
    except Exception as e:
        logger.error(f"error occured while spliting the data into the train and test:{e}")

