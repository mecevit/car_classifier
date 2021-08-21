from layer import Dataset, Model, Context
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision
import pandas as pd
import gzip
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, models
from torchvision.transforms import functional as F


def debug(msg):
    print("MECEVIT: " + str(msg), flush=True)


def build_feature(context: Context, ds: Dataset("carimages"),
                  backlight_detector_model: Model("backlight_detector")) -> any:
    debug("Building feature started")

    sc = context.spark().sparkContext
    carimages_df = ds.to_spark()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    debug("Device: " + str(device))

    trained_model = backlight_detector_model.get_train()
    debug("Trained model: " + str(trained_model))
    model_state = trained_model.state_dict()
    debug("Model state: " + str(model_state))
    bc_model_state = sc.broadcast(model_state)

    debug("Broadcasted model")

    @pandas_udf(StringType())
    def predict_batch_udf(carids: pd.Series, carimages: pd.Series) -> pd.Series:
        feature_data = []

        debug("Constructing broadcasted model")

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        debug("Model state: "+ str(bc_model_state))
        debug("Model state value: " + str(bc_model_state.value))

        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        #     pretrained=True)
        #
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        # model.roi_heads.mask_predictor = None
        # # model.load_state_dict(bc_model_state.value)
        #
        # debug("Model is:" + str(model))
        # model.eval()
        # model.to(device)
        #
        for index, content in carimages.iteritems():
            #     debug("Predicting car #" + str(carids[index]))
            #     img = Image.open(BytesIO(base64.b64decode(content)))
            #     img_input = F.to_tensor(img)
            #     prediction = model([img_input.to(device)])
            #     debug(prediction)
            #
            #     img_str = None
            #     if len(prediction) > 0 and len(prediction[0]['scores']) > 0 and \
            #             prediction[0]['scores'][0] > 0.85:
            #         box = prediction[0]['boxes'][0]
            #         x1 = float(box[0])
            #         y1 = float(box[1])
            #         box_w = float(box[2]) - x1
            #         box_h = float(box[3]) - y1
            #         img1 = img.crop((x1, y1, x1 + box_w, y1 + box_h))
            #
            #         buffered = BytesIO()
            #         img1.save(buffered, format="JPEG")
            #         image_bytes = base64.b64encode(buffered.getvalue())
            #         img_str = image_bytes.decode()
            img_str = "test"
            feature_data.append(img_str)

        return pd.Series(feature_data)

    carimages_df = carimages_df.limit(5)
    debug("Limited the rows")
    predictions_df = carimages_df.select(col('id'), predict_batch_udf(col('id'),
                                                                      col(
                                                                          'content')).alias(
        "prediction"))

    debug("Set the processes")

    returning_df = predictions_df.toPandas()

    debug("Returning dataframe ready:" + str(returning_df))

    return returning_df
