from layer import Dataset, Model, Context
import base64
from io import BytesIO
from PIL import Image
import torch
import pandas as pd
from torchvision.transforms import functional as F


def build_feature(ds: Dataset("carimages"),
                  backlight_detector_model: Model("backlight_detector")) -> any:
    df = ds.to_pandas()

    feature_data = []
    trained_backlight_detector = backlight_detector_model.get_train()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        for index, row in df.iterrows():
            img = Image.open(BytesIO(base64.b64decode(row.content)))
            img_input = F.to_tensor(img)

            losses, prediction = trained_backlight_detector(
                [img_input.to(device)])

            img_str = None
            if len(prediction) > 0 and len(prediction[0]['scores']) > 0 and \
                    prediction[0]['scores'][0] > 0.85:
                box = prediction[0]['boxes'][0]
                x1 = float(box[0])
                y1 = float(box[1])
                box_w = float(box[2]) - x1
                box_h = float(box[3]) - y1
                img1 = img.crop((x1, y1, x1 + box_w, y1 + box_h))

                buffered = BytesIO()
                img1.save(buffered, format="JPEG")
                image_bytes = base64.b64encode(buffered.getvalue())
                img_str = image_bytes.decode()

            feature_data.append([row.id, img_str])

    feature_df = pd.DataFrame(feature_data, columns=['id', 'backlight_image'])
    return feature_df
