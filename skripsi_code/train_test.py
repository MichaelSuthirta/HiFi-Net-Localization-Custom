import os
import csv
import torch
import torch.nn as nn

from train import train
from skripsi_code.test import evaluate_and_visualize
from plot_metrics import plot_metrics
from dataset import ForgeryDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == "main":
    device = get_device()
    print("Using device: " + device)

    baseDir = "E:/College/Pre-Thesis & Thesis/Dataset Fix April/FaShifter_extracted/"

    trainFake = baseDir + "Train/fake"
    trainMask = baseDir + "Train/mask"

    testFake = baseDir + "Test/fake"
    testMask = baseDir + "Test/mask"

    dataset_train = ForgeryDataset(fake_dir=trainFake, mask_dir=trainMask)
    dataset_test = ForgeryDataset(fake_dir=testFake, mask_dir=testMask)

    train(device=device, dataset_train=dataset_train, batchSize=8, epochs=10)
    evaluate_and_visualize(device=device, dataset_test=dataset_test, batchSize=8)
    plot_metrics()