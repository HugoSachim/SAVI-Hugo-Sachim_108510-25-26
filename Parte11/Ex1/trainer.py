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
import wandb



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

        wandb.init(
            project="mnist_savi",
            name=self.args['experiment_full_name'],
            config=self.args
        )


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

            wandb.log({
                "train_loss": train_epoch_loss,
                "test_loss": test_epoch_loss,
                "epoch": epoch_idx
            })

            self.log_epoch_metrics(epoch_idx)

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

        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))

    def evaluate(self):

        # -----------------------------------------
        # Iterate over test batches and compute the ground truth and predicted values
        # -----------------------------------------
        self.model.eval()
        num_batches = len(self.test_dataloader)

        gt_classes = []
        predicted_classes = []

        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(
                enumerate(self.test_dataloader), total=num_batches):

            # Ground truth
            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()

            # Prediction
            logits = self.model.forward(image_tensor)
            probs = torch.softmax(logits, dim=1)
            batch_predicted_classes = probs.argmax(dim=1).tolist()

            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        # -----------------------------------------
        # Create confusion matrix
        # -----------------------------------------
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for gt, pred in zip(gt_classes, predicted_classes):
            confusion_matrix[gt][pred] += 1

        # -----------------------------------------
        # Plot confusion matrix
        # -----------------------------------------
        plt.figure(2)
        class_names = [str(i) for i in range(10)]
        seaborn.heatmap(confusion_matrix,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        cbar=True,
                        xticklabels=class_names,
                        yticklabels=class_names)

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))

        # -----------------------------------------
        # Compute statistics per class
        # -----------------------------------------
        statistics = {}
        per_class_f1 = []

        total_TP = 0
        total_FP = 0
        total_FN = 0

        for c in range(10):
            TP = int(confusion_matrix[c][c])
            FP = int(confusion_matrix[:, c].sum() - TP)
            FN = int(confusion_matrix[c, :].sum() - TP)

            precision, recall = self.getPrecisionRecall(TP, FP, FN)
            f1 = self.getF1(precision, recall)

            statistics[c] = {
                "digit": c,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            # For global metrics
            if precision is not None and recall is not None:
                per_class_f1.append(f1)

            total_TP += TP
            total_FP += FP
            total_FN += FN

        # -----------------------------------------
        # Global metrics
        # -----------------------------------------

        # Macro = média simples das classes
        global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else None
        global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else None
        global_f1 = self.getF1(global_precision, global_recall)

        statistics["global"] = {
            "precision": global_precision,
            "recall": global_recall,
            "f1_score": global_f1
        }

        print("Global metrics:", statistics["global"])

        # -----------------------------------------
        # Save JSON
        # -----------------------------------------
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(statistics, f, indent=4)

        wandb.log({
            "final_confusion_matrix": wandb.Image(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))
        })


    def getPrecisionRecall(self, TP, FP, FN):

        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None

        return precision, recall


    def getF1(self, precision, recall):
        if precision is None or recall is None or (precision + recall == 0):
            return None
        return 2 * precision * recall / (precision + recall)


    def log_epoch_metrics(self, epoch_idx):

        # Avaliação rápida para obter preds/GT
        self.model.eval()
        gt_classes = []
        predicted_classes = []

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                gt = labels.argmax(dim=1)
                pred = torch.softmax(self.model(images), dim=1).argmax(dim=1)
                gt_classes.extend(gt.tolist())
                predicted_classes.extend(pred.tolist())

        # Construir matriz de confusão
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for gt, pred in zip(gt_classes, predicted_classes):
            confusion_matrix[gt][pred] += 1

        # Calcular métricas globais
        total_TP = sum(confusion_matrix[c][c] for c in range(10))
        total_FP = sum(confusion_matrix[:, c].sum() - confusion_matrix[c][c] for c in range(10))
        total_FN = sum(confusion_matrix[c, :].sum() - confusion_matrix[c][c] for c in range(10))

        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Plot matriz de confusão
        plt.figure(figsize=(6, 6))
        seaborn.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch_idx}")

        # Log para wandb
        wandb.log({
            "precision_global": precision,
            "recall_global": recall,
            "f1_global": f1,
            "confusion_matrix": wandb.Image(plt),
            "epoch": epoch_idx
        })

        plt.close()
