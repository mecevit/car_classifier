# Layer Car Classifier Project

Here are the components (datasets, featuresets and ml models) of this Layer project in the order of execution.

| Component  | Type | Description |
| ------------- | ------------- | ------------- |
| [labeled_cars](https://github.com/mecevit/car_classifier/tree/main/data/dataset_labeled_cars)  | Dataset | Annotated car images dataset. Includes annotations of left and right backlight images of the car |
| [car_images](https://github.com/mecevit/car_classifier/tree/main/data/dataset_car_images)  | Dataset | Car images dataset with different build years. |
| [car_parts_features](https://github.com/mecevit/car_classifier/tree/main/features/car_parts_features)  | Featureset | A featureset that includes the backlights of the cars, extracted by the help of the `backlight_detector` model  |
| [backlight_detector](https://github.com/mecevit/car_classifier/tree/main/models/backlight_detector)  | Model | A Pytorch [model](https://github.com/mecevit/car_classifier/blob/main/models/backlight_detector/model.py) with a custom [CarsDataset](https://github.com/mecevit/car_classifier/blob/main/models/backlight_detector/cars_dataset.py) dataset to extract backlight from the car images. Dataset is populated with the annotated `labeled_cars` dataset. |
| [car_classifier](https://github.com/mecevit/car_classifier/tree/main/models/car_classifier)  | Model | A Pytorch [model](https://github.com/mecevit/car_classifier/blob/main/models/car_classifier/model.py) with a custom [Backlights](https://github.com/mecevit/car_classifier/blob/main/models/car_classifier/backlights_dataset.py) dataset to classify the backlights to detect the build year of the related car. Dataset is populated with the `car_images` dataset` |
| [car_label_encoder](https://github.com/mecevit/car_classifier/tree/main/models/car_label_encoder) | Model | A SKLearn [model](https://github.com/mecevit/car_classifier/blob/main/models/car_label_encoder/model.py) to encode the build year of cars. |

## Installation & Running

To check out the project, run:

```bash
layer clone https://github.com/mecevit/car_classifier
cd car_classifier/
```

To run the project:

```bash
layer start
```

## Sample Notebook

You can use the following notebook to see how you can use the trained models and the features in a notebook environment

[Layer Car Classifier Notebook](https://github.com/mecevit/car_classifier/blob/main/notebooks/LayerCarClassifier.ipynb)

