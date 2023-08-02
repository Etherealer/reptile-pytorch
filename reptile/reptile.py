import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .variables import VariableState, average_vars, interpolate_vars

__all__ = ['Reptile']


class Reptile:
    def __init__(self, model, optimizer, criterion, device, transductive):
        self.model = model
        self.optimizer = optimizer
        self._model_state = VariableState(self.model, trainable=True)
        self._full_state = VariableState(self.model, self.optimizer, trainable=False)
        self.criterion = criterion
        self.device = device
        self._transductive = transductive

    def train(self, dataset, num_classes, num_shots, meta_batch_size, inner_batch_size, inner_iters, replacement,
              meta_step_size):
        old_vars = self._model_state.export_variables()
        new_vars = []
        for n in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            self.train_step(mini_dataset, inner_batch_size, inner_iters, replacement)
            new_vars = average_vars(new_vars, self._model_state.export_variables(), n + 1)
            self._model_state.import_variables(old_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

    def evaluate(self, dataset, num_classes, num_shots, inner_batch_size, inner_iters, replacement):
        train_set, test_set = _spilt_train_test(_sample_mini_dataset(dataset, num_classes, num_shots + 1))
        old_vars = self._full_state.export_variables()
        self.train_step(train_set, inner_batch_size, inner_iters, replacement)

        acc = self._test_predictions(test_set)
        self._full_state.import_variables(old_vars)
        return acc

    def train_step(self, train_set, inner_batch_size, inner_iters, replacement):
        self.model.train()
        for inputs, targets in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            recon = self.model(inputs)
            loss = self.criterion(recon, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _test_predictions(self, test_set):
        self.set_model_mode()
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        _, indices = torch.max(outputs, dim=1)
        num_correct = sum([pred == target for pred, target in zip(indices, targets)])
        acc = num_correct / len(targets)
        return acc

    def set_model_mode(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train(self._transductive)


def _spilt_train_test(samples, test_shots=1):
    labels = set(item[1] for item in samples)
    train_set_indices = list(range(len(samples)))
    test_set_indices = []
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(samples):
                if item[1] == label:
                    test_set_indices.append(i)
                    break
    if len(test_set_indices) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    train_set_indices = [index for index in train_set_indices if index not in test_set_indices]
    return Subset(samples, train_set_indices), Subset(samples, test_set_indices)


def _sample_mini_dataset(dataset, num_classes, num_shots):
    classes = torch.randperm(len(dataset))[:num_classes]
    return dataset.sample(classes, num_shots)


def _mini_batches(samples, batch_size, num_batches, replacement):
    indices = []
    if replacement:
        indices = random.choices(range(len(samples)), k=batch_size * num_batches)
    else:
        while len(indices) < num_batches * batch_size:
            indices += random.sample(range(len(samples)), len(samples))
    dataset = Subset(samples, indices[:num_batches * batch_size])
    dataloader = DataLoader(dataset, batch_size)
    return dataloader
