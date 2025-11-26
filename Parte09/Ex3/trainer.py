import glob
import os
import random
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn


class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        # self.train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        # self.test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

        self.train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

        # For testing we typically set shuffle to false

        # Setup loss function
        #self.loss = nn.MSELoss()  # Mean Squared Error Loss
        self.loss = nn.BCEWithLogitsLoss()

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

    def train(self):

        print('Training started ...')

        for epoch_idx in range(self.args['num_epochs']):
        #for epoch_idx in range(5):  # number of epochs
            print('\nEpoch index = ' + str(epoch_idx))

            batch_losses = []
            # Iterate over batches
            for batch_idx, (image_tensor, label_gt_tensor) in enumerate(self.train_dataloader):
                print('\nBatch index = ' + str(batch_idx))

                print('image_tensor shape: ' + str(image_tensor.shape))
                print('label_gt_tensor shape: ' + str(label_gt_tensor.shape))

                # # Compute the predicted labels
                # label_pred_tensor = self.model.forward(image_tensor)

                # # Compute the probabilities using softmax
                # label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                # # Compute the loss using MSE
                # batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                # batch_losses.append(batch_loss.item())
                # print('batch_loss: ' + str(batch_loss.item()))

                # Compute the predicted labels (logits)
                label_pred_tensor = self.model.forward(image_tensor)

                # ❌ NÃO calcular softmax aqui
                # (BCEWithLogitsLoss já faz a função logística internamente)

                # Compute the loss directly on the logits
                batch_loss = self.loss(label_pred_tensor, label_gt_tensor)

                batch_losses.append(batch_loss.item())
                print('batch_loss: ' + str(batch_loss.item()))


                # Update model

                # In PyTorch, gradients accumulate by default. This means that if you perform loss.backward() multiple times without zeroing the gradients, the new gradients will be added to the existing ones. While this can be useful in some advanced scenarios (like accumulating gradients over smaller mini-batches to simulate a larger batch size), it's generally not what you want for a standard training step.
                # For each new batch, you want to calculate the gradients based only on the loss from that specific batch. zero_grad() ensures that all the gradients stored in the model's parameters (specifically, in the .grad attribute of each parameter tensor) are reset to zero before you start calculating the gradients for the current batch.
                self.optimizer.zero_grad()  # resets the gradients from previous batches

                #  If batch_loss is how "wrong" your model's predictions were for the current batch, backward() is like tracing back through every calculation that led to that "wrongness" and figuring out precisely how much each individual dial (parameter) in your model contributed to it. It then records those contributions as gradients.
                batch_loss.backward()  # the actual backpropagation

                # Once batch_loss.backward() has computed and stored the gradients for all parameters, optimizer.step() uses these gradients to update the model's parameters according to the specific optimization algorithm (e.g., SGD, Adam).
                self.optimizer.step()

            # End of batches
            print('Finished epoch ' + str(epoch_idx))
            print('batch_losses: ' + str(batch_losses))
            epoch_loss = np.mean(batch_losses)
            print('epoch_loss: ' + str(epoch_loss))

    def test_all_show_random(self, n_examples_to_show=10):
        print("\nTesting on full dataset...")

        self.model.eval()  # modo avaliação

        total = 0
        correct = 0

        # Para escolher random sem guardar tudo:
        selected = []  # lista de tuplas (img_tensor, gt_label, pred_label)
        rng = random.Random(42)  # semente opcional para reproducibilidade

        with torch.no_grad():
            for image_tensor, label_gt_tensor in self.test_dataloader:

                # Forward
                logits = self.model.forward(image_tensor)
                probs = torch.softmax(logits, dim=1)

                # Predicted labels
                pred_labels = torch.argmax(probs, dim=1)
                gt_labels = torch.argmax(label_gt_tensor, dim=1)

                total += gt_labels.size(0)
                correct += (pred_labels == gt_labels).sum().item()

                # Escolher aleatoriamente algumas imagens do batch
                for i in range(image_tensor.size(0)):
                    if len(selected) < n_examples_to_show:
                        selected.append((image_tensor[i], gt_labels[i].item(), pred_labels[i].item()))
                    else:
                        # Reservoir sampling: substituir aleatoriamente para manter n_examples_to_show
                        j = rng.randint(0, total-1)
                        if j < n_examples_to_show:
                            selected[j] = (image_tensor[i], gt_labels[i].item(), pred_labels[i].item())

        accuracy = correct / total
        print(f"\nFinal Test Accuracy = {accuracy*100:.2f}%")

        # Mostrar as imagens selecionadas
        to_pil = transforms.ToPILImage()
        for img_tensor, gt, pred in selected:
            img = to_pil(img_tensor)
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.title(f"GT: {gt} | Pred: {pred}")
            plt.axis("off")
            plt.show()