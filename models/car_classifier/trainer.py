import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
from layer import Train
import numpy as np
import torch
import torch.nn as nn


def debug(msg):
    print(msg, flush=True)


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train_base_model(train: Train, model, data_loaders, dataset_sizes, device):
    n_epochs = 10
    lr = 0.001
    momentum = 0.9
    step_size = 7
    gamma = 0.1

    train.log_parameter("n_epochs", n_epochs)
    train.log_parameter("lr", lr)
    train.log_parameter("momentum", momentum)
    train.log_parameter("step_size", step_size)
    train.log_parameter("gamma", gamma)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):

        debug(f'Epoch {epoch + 1}/{n_epochs}')
        debug('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            data_loaders['train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            dataset_sizes['train']
        )

        debug(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            data_loaders['val'],
            loss_fn,
            device,
            dataset_sizes['val']
        )

        debug(f'Val   loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    debug(f'Best val accuracy: {best_accuracy}')
    train.log_parameter("accuracy", best_accuracy)

    model.load_state_dict(torch.load('best_model_state.bin'))

    return model
