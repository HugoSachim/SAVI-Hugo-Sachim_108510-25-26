import glob
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import seaborn
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm


class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        # For testing we typically set shuffle to false

        # Define loss for the epochs
        self.train_epoch_losses = []
        self.test_epoch_losses = []

        # Setup loss function
        self.loss = nn.MSELoss()  # Mean Squared Error Loss

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        # Setup the figure
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.1])  # type: ignore

    def train(self):

        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        # -----------------------------------------
        # Iterate all epochs
        # -----------------------------------------
        for epoch_idx in range(self.args['num_epochs']):  # number of epochs
            print('\nEpoch index = ' + str(epoch_idx))

            # -----------------------------------------
            # Train - Iterate over batches
            # -----------------------------------------
            self.model.train()  # set model to training mode
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.train_dataloader), total=num_batches):  # type: ignore

                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()  # the actual backpropagation
                self.optimizer.step()

            # -----------------------------------------
            # Test - Iterate over batches
            # -----------------------------------------
            self.model.eval()  # set model to evaluation mode

            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                    enumerate(self.test_dataloader), total=num_batches):  # type: ignore
                # print('\nBatch index = ' + str(batch_idx))
                # print('image_tensor shape: ' + str(image_tensor.shape))
                # print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # Compute the predicted labels
                label_pred_tensor = self.model.forward(image_tensor)

                # Compute the probabilities using softmax
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # Compute the loss using MSE
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # During test there is no model update

            # ---------------------------------
            # End of the epoch training
            # ---------------------------------
            print('Finished epoch ' + str(epoch_idx) + ' out of ' + str(self.args['num_epochs']))
            # print('batch_losses: ' + str(batch_losses))

            # update the training epoch losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            # update the testing epoch losses
            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            # Draw the updated training figure
            self.draw()

        print('Training completed.')
        print('Training losses: ' + str(self.train_epoch_losses))
        print('Test losses: ' + str(self.test_epoch_losses))

    def draw(self):

        # plot training
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        # plot testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        plt.legend(['Train', 'Test'])

        plt.savefig('training.png')


    def evaluate(self):

        print("\nEvaluating model on test dataset...")

        # -----------------------------------------
        # Iterate over test batches and compute predictions
        # -----------------------------------------
        self.model.eval()
        num_batches = len(self.test_dataloader)

        gt_classes = []
        predicted_classes = []

        with torch.no_grad():
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=num_batches
            ):
                # Ground truth
                batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

                # Predicted
                logits = self.model.forward(image_tensor)
                probs = torch.softmax(logits, dim=1)
                batch_pred_classes = probs.argmax(dim=1).tolist()

                gt_classes.extend(batch_gt_classes)
                predicted_classes.extend(batch_pred_classes)

        # -----------------------------------------
        # Confusion Matrix
        # -----------------------------------------
        confusion_matrix = np.zeros((10, 10), dtype=int)

        for gt, pred in zip(gt_classes, predicted_classes):
            confusion_matrix[gt][pred] += 1

        # -----------------------------------------
        # Plot confusion matrix
        # -----------------------------------------
        plt.figure(figsize=(8,6))
        seaborn.heatmap(confusion_matrix, annot=True, fmt="d",
                        cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        # -----------------------------------------
        # Compute per-class statistics
        # -----------------------------------------
        statistics = {"classes": [], "global": {}}

        global_TP = 0
        global_FP = 0
        global_FN = 0

        for i in range(10):

            TP = int(confusion_matrix[i][i])
            FP = int(confusion_matrix[:, i].sum() - TP)
            FN = int(confusion_matrix[i, :].sum() - TP)

            precision, recall, f1 = self.getPrecisionRecall(TP, FP, FN)

            statistics["classes"].append({
                "digit": str(i),
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            # accumulate global counts
            global_TP += TP
            global_FP += FP
            global_FN += FN

        # -----------------------------------------
        # Compute global precision, recall, f1
        # -----------------------------------------
        g_precision, g_recall, g_f1 = self.getPrecisionRecall(global_TP, global_FP, global_FN)

        statistics["global"] = {
            "TP": global_TP,
            "FP": global_FP,
            "FN": global_FN,
            "precision": g_precision,
            "recall": g_recall,
            "f1_score": g_f1
        }

        print("\nPer-class statistics:")
        print(json.dumps(statistics["classes"], indent=4))

        print("\nGlobal statistics:")
        print(json.dumps(statistics["global"], indent=4))

        # -----------------------------------------
        # Save JSON
        # -----------------------------------------
        with open("statistics.json", "w") as f:
            json.dump(statistics, f, indent=4)


    def getPrecisionRecall(self, TP, FP, FN):

        # Precision
        if (TP + FP) == 0:
            precision = None
        else:
            precision = TP / (TP + FP)

        # Recall
        if (TP + FN) == 0:
            recall = None
        else:
            recall = TP / (TP + FN)

        # F1-Score
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1