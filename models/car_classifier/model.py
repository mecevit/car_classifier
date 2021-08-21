from typing import Any
from layer import Featureset, Train, Dataset
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from sklearn import preprocessing
from .backlights_dataset import BacklightsDataset
from .trainer import train_base_model


def train_model(train: Train,
                cpf: Featureset("car_parts_features"),
                lc: Dataset("carimages")) -> Any:
    carparts_df = cpf.to_pandas()
    cars_df = lc.to_pandas()

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    df = carparts_df.join(cars_df, lsuffix='_car', rsuffix='_carparts')
    df = df[df['backlight_image'].notnull()]

    train.log_parameter("Sample size", df.shape[0])

    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df.year.values)

    # Channel wise mean and standard deviation for normalizing according to
    # ImageNet Statistics
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # Transforms to be applied to Train-Test-Validation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    test_valid_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    train_dataset = BacklightsDataset(df, transform=train_transforms)
    valid_dataset = BacklightsDataset(df, transform=test_valid_transforms)
    test_dataset = BacklightsDataset(df, transform=test_valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=5,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    n_classes = 14
    model = models.resnet34(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)
    base_model = model.to(device)

    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    dataset_sizes = {d: len(data_loaders[d]) for d in data_loaders}

    base_model, history = train_base_model(base_model, data_loaders,
                                           dataset_sizes, device, n_epochs=10)

    scripted_pytorch_model = torch.jit.script(base_model)
    return scripted_pytorch_model
