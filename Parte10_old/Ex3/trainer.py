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

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

from tqdm import tqdm
import json

class Trainer():
    def __init__(self, args, train_dataset, test_dataset, model):
        self.args = args
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001)
        self.train_losses = []
        self.test_losses = []

    # ----------------------------
    # Treino
    # ----------------------------
    def train(self):
        print('Training started ...')

        for epoch_idx in range(self.args['num_epochs']):
            batch_losses = []

            # ----------------------
            # Treino
            # ---------------------
            for batch_idx, (image_tensor, label_gt_tensor) in enumerate(
                    tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}", leave=False)):

                logits = self.model.forward(image_tensor)
                batch_loss = self.loss(logits, label_gt_tensor)
                batch_losses.append(batch_loss.item())

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            # Média da loss da época
            epoch_loss = np.mean(batch_losses)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch_idx+1} Train Loss: {epoch_loss:.4f}")

            # ----------------------
            # Avaliação no dataset de teste
            # ----------------------
            test_loss = []
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.test_dataloader:
                    logits = self.model.forward(images)
                    batch_loss = self.loss(logits, labels)
                    test_loss.append(batch_loss.item())
            test_epoch_loss = np.mean(test_loss)
            self.test_losses.append(test_epoch_loss)
            print(f"Epoch {epoch_idx+1} Test Loss: {test_epoch_loss:.4f}")
            self.model.train()

            # ----------------------
            # Desenhar gráfico
            # ----------------------
            self.draw()
            plt.pause(0.1)  # mostra 2 segundos
            # ----------------------
            # Avaliar métricas no dataset de teste
            # ----------------------
            statistics = self.evaluate()  # chama o método evaluate que atualiza statistics.json
            print(f"Epoch {epoch_idx+1} Statistics saved!")

        print("Training finished.")
        self.evaluate()

    # ----------------------------
    # Visualização loss
    # ----------------------------
    def draw(self):
        plt.clf()
        plt.plot(range(1, len(self.train_losses)+1), self.train_losses, label='Train Loss', marker='o')
        if self.test_losses:
            plt.plot(range(1, len(self.test_losses)+1), self.test_losses, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.draw()

    # ----------------------------
    # Avaliação
    # ----------------------------
    def evaluate(self):
        print("\nEvaluating model on test dataset...")
        self.model.eval()
        all_preds = []
        all_gt = []

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                logits = self.model.forward(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                gt_labels = torch.argmax(labels, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_gt.extend(gt_labels.cpu().numpy())

        # Mostra matriz de confusão rapidamente
        cm = confusion_matrix(all_gt, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")
        plt.show(block=False)
        #plt.pause(2)  # mostra 2 segundos
        plt.close()

        # Calcula estatísticas
        stats = self.compute_statistics(all_gt, all_preds)
        return stats

    # ----------------------------
    # Computar estatísticas
    # ----------------------------
    def compute_statistics(self, all_gt, all_preds):
        num_classes = 10
        statistics = {"classes": [], "global": {}}

        TP_global = FP_global = FN_global = 0

        for c in range(num_classes):
            TP = np.sum((np.array(all_gt) == c) & (np.array(all_preds) == c))
            FP = np.sum((np.array(all_gt) != c) & (np.array(all_preds) == c))
            FN = np.sum((np.array(all_gt) == c) & (np.array(all_preds) != c))

            TP_global += TP
            FP_global += FP
            FN_global += FN

            statistics["classes"].append({
                "digit": str(c),
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "precision": None,
                "recall": None,
                "f1_score": None
            })

        precision_global = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else 0
        recall_global = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else 0
        f1_global = 2 * precision_global * recall_global / (precision_global + recall_global) \
                    if (precision_global + recall_global) > 0 else 0

        statistics["global"] = {
            "TP": int(TP_global),
            "FP": int(FP_global),
            "FN": int(FN_global),
            "precision": precision_global,
            "recall": recall_global,
            "f1_score": f1_global
        }

        for c_stat in statistics["classes"]:
            TP = c_stat["TP"]
            FP = c_stat["FP"]
            FN = c_stat["FN"]
            c_stat["precision"] = TP / (TP + FP) if (TP + FP) > 0 else 0
            c_stat["recall"] = TP / (TP + FN) if (TP + FN) > 0 else 0
            c_stat["f1_score"] = 2 * c_stat["precision"] * c_stat["recall"] / (c_stat["precision"] + c_stat["recall"]) \
                                 if (c_stat["precision"] + c_stat["recall"]) > 0 else 0

        with open("statistics.json", "w") as f:
            json.dump(statistics, f, indent=4)

        print("Statistics saved to statistics.json")
        return statistics





















# class Trainer():
#     def __init__(self, args, train_dataset, test_dataset, model):
#         self.args = args
#         self.model = model

#         self.train_dataloader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args['batch_size'], shuffle=True)
#         self.test_dataloader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=args['batch_size'], shuffle=False)

#         # Loss function
#         self.loss = torch.nn.BCEWithLogitsLoss()
#         # Optimizer
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

#         # Guardar losses
#         self.train_losses = []
#         self.test_losses = []

#     def train(self):
#         print('Training started ...')

#         for epoch_idx in range(self.args['num_epochs']):
#             print('\nEpoch index = ' + str(epoch_idx))

#             batch_losses = []

#             # Loop pelos batches com tqdm
#             pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}", leave=False)
#             for batch_idx, (image_tensor, label_gt_tensor) in enumerate(pbar):
#                 # Forward pass
#                 label_pred_tensor = self.model.forward(image_tensor)

#                 # Loss direta (BCEWithLogitsLoss inclui sigmoid)
#                 batch_loss = self.loss(label_pred_tensor, label_gt_tensor)
#                 batch_losses.append(batch_loss.item())

#                 # Backward
#                 self.optimizer.zero_grad()
#                 batch_loss.backward()
#                 self.optimizer.step()

#                 # Atualiza barra com batch_loss
#                 pbar.set_postfix({'batch_loss': batch_loss.item()})

#             # Fim da época
#             epoch_loss = np.mean(batch_losses)
#             print(f"Finished epoch {epoch_idx}, epoch_loss: {epoch_loss:.4f}")

#             # Guarda loss da época
#             self.train_losses.append(epoch_loss)

#             # Avaliar loss no dataset de teste
#             test_batch_losses = []
#             for image_tensor, label_gt_tensor in self.test_dataloader:
#                 with torch.no_grad():  # sem gradientes
#                     logits = self.model.forward(image_tensor)
#                     batch_loss = self.loss(logits, label_gt_tensor)
#                     test_batch_losses.append(batch_loss.item())

#             epoch_test_loss = np.mean(test_batch_losses)
#             print(f"Epoch {epoch_idx} test_loss: {epoch_test_loss:.4f}")

#             # Guardar loss do teste
#             self.test_losses.append(epoch_test_loss)

#             # Desenhar curvas (treino vs teste)
#             self.draw()
        
#         plt.ioff()  # desativa modo interativo
#         plt.show()  # garante que o gráfico final fique aberto

#     def test_all_show_random(self, n_examples_to_show=10):
#         print("\nTesting on full dataset...")

#         self.model.eval()  # modo avaliação

#         total = 0
#         correct = 0

#         # Para escolher random sem guardar tudo:
#         selected = []  # lista de tuplas (img_tensor, gt_label, pred_label)
#         rng = random.Random(42)  # semente opcional para reproducibilidade

#         with torch.no_grad():
#             for image_tensor, label_gt_tensor in self.test_dataloader:

#                 # Forward
#                 logits = self.model.forward(image_tensor)
#                 probs = torch.softmax(logits, dim=1)

#                 # Predicted labels
#                 pred_labels = torch.argmax(probs, dim=1)
#                 gt_labels = torch.argmax(label_gt_tensor, dim=1)

#                 total += gt_labels.size(0)
#                 correct += (pred_labels == gt_labels).sum().item()

#                 # Escolher aleatoriamente algumas imagens do batch
#                 for i in range(image_tensor.size(0)):
#                     if len(selected) < n_examples_to_show:
#                         selected.append((image_tensor[i], gt_labels[i].item(), pred_labels[i].item()))
#                     else:
#                         # Reservoir sampling: substituir aleatoriamente para manter n_examples_to_show
#                         j = rng.randint(0, total-1)
#                         if j < n_examples_to_show:
#                             selected[j] = (image_tensor[i], gt_labels[i].item(), pred_labels[i].item())

#         accuracy = correct / total
#         print(f"\nFinal Test Accuracy = {accuracy*100:.2f}%")

#         # Mostrar as imagens selecionadas
#         to_pil = transforms.ToPILImage()
#         for img_tensor, gt, pred in selected:
#             img = to_pil(img_tensor)
#             plt.figure()
#             plt.imshow(img, cmap='gray')
#             plt.title(f"GT: {gt} | Pred: {pred}")
#             plt.axis("off")
#             plt.show()

#     def evaluate(self):
#         print("\nEvaluating model on test dataset...")

#         self.model.eval()
#         all_preds = []
#         all_gt = []

#         with torch.no_grad():
#             for images, labels in self.test_dataloader:
#                 logits = self.model.forward(images)
#                 preds = torch.argmax(logits, dim=1)  # argmax direto sobre logits
#                 gt_labels = torch.argmax(labels, dim=1)

#                 all_preds.extend(preds.cpu().numpy())
#                 all_gt.extend(gt_labels.cpu().numpy())

#         # Matriz de confusão
#         cm = confusion_matrix(all_gt, all_preds)
        
#         plt.figure(figsize=(8,6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel("Predicted")
#         plt.ylabel("Ground Truth")
#         plt.title("Confusion Matrix")
#         plt.show()  # aguarda o utilizador fechar a janela

#         # Métricas de avaliação por classe
#         precision = precision_score(all_gt, all_preds, average=None)
#         recall = recall_score(all_gt, all_preds, average=None)
#         f1 = f1_score(all_gt, all_preds, average=None)

#         for i in range(10):
#             print(f"Digit {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1-Score={f1[i]:.3f}")

#         # Métricas agregadas (macro average)
#         print(f"\nMacro-Average: Precision={precision.mean():.3f}, Recall={recall.mean():.3f}, F1-Score={f1.mean():.3f}")

#     def draw(self):
#         import matplotlib.pyplot as plt

#         plt.ion()  # modo interativo ligado
#         plt.clf()  # limpa o gráfico atual
#         plt.plot(range(1, len(self.train_losses)+1), self.train_losses,
#                 label='Train Loss', marker='o')
#         if self.test_losses:
#             plt.plot(range(1, len(self.test_losses)+1), self.test_losses,
#                     label='Test Loss', marker='o')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Loss vs Epoch')
#         plt.legend()
#         plt.grid(True)
#         plt.draw()
#         plt.pause(0.1)  # mostra o gráfico por 2 segundos antes de continuar