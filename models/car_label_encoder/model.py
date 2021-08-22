from typing import Any
from layer import Featureset, Train, Dataset
from sklearn import preprocessing



def train_model(train: Train, ci: Dataset("carimages")) -> Any:
    cars_df = ci.to_pandas()

    le = preprocessing.LabelEncoder()
    le.fit(cars_df.year.values)

    train.log_parameter("Label Count", len(cars_df.year.unique()))
    train.log_parameter("Min", min(cars_df.year))
    train.log_parameter("Max", max(cars_df.year))

    return le
