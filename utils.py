import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from sklearn.metrics import confusion_matrix
from typing import Tuple


# === Classes ===

class SmallModel(nn.Module):
	def __init__(self, lx: int, lz: int) -> None:
		super(SmallModel, self).__init__()

		self.conv1 = nn.Conv2d(lz, 96, kernel_size=3)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.drop1 = nn.Dropout(0.2)

		self.conv2 = nn.Conv2d(96, 192, kernel_size=3)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.drop2 = nn.Dropout(0.2)

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(self._calculate_output_size(lx, lz), 1024)
		self.drop3 = nn.Dropout(0.5)

		self.fc2 = nn.Linear(1024, 43)

	def _calculate_output_size(self, lx: int, lz: int) -> int:
		cinput = torch.zeros(1, lz, lx, lx)
		coutput = self.pool1(F.relu(self.conv1(cinput)))
		coutput = self.pool2(F.relu(self.conv2(coutput)))
		return coutput.numel() // coutput.size(0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = self.drop1(x)

		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = self.drop2(x)

		x = self.flatten(x)
		x = F.relu(self.fc1(x))
		x = self.drop3(x)

		x = F.softmax(self.fc2(x), dim=1)

		return x
	
	class BigModel(nn.Module):
		def __init__(self, lx: int, lz: int) -> None:
			super(BigModel, self).__init__()

			self.conv1 = nn.Conv2d(lz, 32, kernel_size=3)
			self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.drop1 = nn.Dropout(0.5)

			self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
			self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.drop2 = nn.Dropout(0.5)

			self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
			self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.drop3 = nn.Dropout(0.5)

			self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
			self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.drop4 = nn.Dropout(0.5)

			self.flatten = nn.Flatten()
			self.fc1 = nn.Linear(self._calculate_output_size(lx, lz), 1024)
			self.drop5 = nn.Dropout(0.5)

			self.fc2 = nn.Linear(1024, 43)

		def _calculate_output_size(self, lx: int, lz: int) -> int:
			cinput = torch.zeros(1, lz, lx, lx)
			coutput = self.pool1(F.relu(self.conv1(cinput)))
			coutput = self.pool2(F.relu(self.conv2(coutput)))
			coutput = self.pool3(F.relu(self.conv3(coutput)))
			coutput = self.pool4(F.relu(self.conv4(coutput)))
			return coutput.numel() // coutput.size(0)
		
		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = F.relu(self.conv1(x))
			x = self.pool1(x)
			x = self.drop1(x)

			x = F.relu(self.conv2(x))
			x = self.pool2(x)
			x = self.drop2(x)

			x = F.relu(self.conv3(x))
			x = self.pool3(x)
			x = self.drop3(x)

			x = F.relu(self.conv4(x))
			x = self.pool4(x)
			x = self.drop4(x)

			x = self.flatten(x)
			x = F.relu(self.fc1(x))
			x = self.drop5(x)

			x = F.softmax(self.fc2(x), dim=1)

			return x


# === Enums ===

class ModelName(Enum):
	"""
	Enumeration of the available models.
	"""

	SMALL = 'smallCNN'
	BIG = 'bigCNN'


# === Functions ===

def get_model(name: ModelName, lx: int, ly: int, lz: int) -> nn.Module:
	"""
	Returns a model given by name.

	Args:
		name (Model): The name of the model to retrieve.
		lx (int): The width of the input images.
		ly (int): The height of the input images.
		lz (int): The number of channels of the input images.

	Returns:
		torch.nn.Module: The model.
	"""

	if lx != ly:
		raise ValueError('Only square images are supported.')

	if name == ModelName.SMALL:
		return SmallModel(lx, lz)
	elif name == ModelName.BIG:
		return BigModel(lx, lz)
	else:
		raise ValueError(f'Invalid model name: {name}')

def load_dataset(enhanced_dir: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Reads a dataset from an HDF5 file and returns the training, testing, and meta data.

	Args:
		enhanced_dir (str): The directory where the HDF5 file is located.
		dataset_name (str): The name of the dataset (without the .h5 extension).

	Returns:
		tuple: A tuple containing six elements:
			- x_train (numpy.ndarray): The training data features.
			- y_train (numpy.ndarray): The training data labels.
			- x_test (numpy.ndarray): The testing data features.
			- y_test (numpy.ndarray): The testing data labels.
			- x_meta (numpy.ndarray): The meta data features.
			- y_meta (numpy.ndarray): The meta data labels.
	"""

	filename = f'{enhanced_dir}/{dataset_name}.h5'

	with h5py.File(filename,'r') as f:
		x_train = f['x_train'][:]
		y_train = f['y_train'][:]
		x_test  = f['x_test'][:]
		y_test  = f['y_test'][:]
		x_meta  = f['x_meta'][:]
		y_meta  = f['y_meta'][:]

	return x_train, y_train, x_test, y_test, x_meta, y_meta

def plot_imgs(num_imgs: int, cols: int, images: np.ndarray, labels: np.ndarray, preds: np.ndarray = None) -> None:
	"""
	Plots a grid of images with their corresponding labels.

	Args:
		images (numpy.ndarray): List or array of images to be plotted.
		labels (numpy.ndarray): List of labels corresponding to the images.
		preds (numpy.ndarray): List of predicted labels.
		num_imgs (int): Number of images to plot.
		cols (int): Number of columns in the grid.

	Returns:
		None
	"""

	rows = (num_imgs + cols - 1) // cols

	fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
	fig.subplots_adjust(hspace=0.5, wspace=0.5)

	# Show the images
	for i in range(num_imgs):
		ax = axes[i // cols, i % cols]
		ax.imshow(images[i], cmap='gray' if images[i].shape[2] == 1 else None)
		if preds == None:
			ax.set_xlabel(labels[i], color='black')
		elif preds[i] == labels[i]:
			ax.set_xlabel(preds[i].item(), color='green')
		else:
			ax.set_xlabel(f'{preds[i].item()} ({labels[i].item()})', color='red')
		ax.set_xticks([])
		ax.set_yticks([])

	# Fill the empty plots with white images
	for j in range(num_imgs, rows * cols):
		ax = axes[j // cols, j % cols]
		ax.axis('off')

	plt.show()

def plot_history(train_losses: list, test_losses: list, train_accuracies: list, test_accuracies: list, epoch: int) -> None:
	"""
	Plots the training and testing loss and accuracy over epochs.

	Parameters:
	train_losses (list): List of training losses for each epoch.
	test_losses (list): List of testing losses for each epoch.
	train_accuracies (list): List of training accuracies for each epoch.
	test_accuracies (list): List of testing accuracies for each epoch.
	epoch (int): The number of epochs.

	Returns:
	None
	"""

	_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	# Loss plot
	ax1.set_title('Loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.plot(train_losses, 'b', label='Train')
	ax1.plot(test_losses, 'r', label='Test')
	ax1.legend()
	ax1.annotate(
		f'Train: {train_losses[-1]:.2f}',
		xy         = (epoch, train_losses[-1]),
		xytext     = (epoch, train_losses[-1]),
		color      = 'b',
		arrowprops = None
	)
	ax2.annotate(
		f'Train: {train_accuracies[-1]:.2f}',
		xy         = (epoch, train_accuracies[-1]),
		xytext     = (epoch, train_accuracies[-1]),
		color      = 'b',
		arrowprops = None
	)

	# Accuracy plot
	ax2.set_title('Accuracy')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')
	ax2.plot(train_accuracies, 'b', label='Train')
	ax2.plot(test_accuracies, 'r', label='Test')
	ax2.legend()
	ax1.annotate(
		f'Test: {test_losses[-1]:.2f}',
		xy         = (epoch, test_losses[-1]),
		xytext     = (epoch, test_losses[-1]),
		color      = 'r',
		arrowprops = None
	)
	ax2.annotate(
		f'Test: {test_accuracies[-1]:.2f}',
		xy         = (epoch, test_accuracies[-1]),
		xytext     = (epoch, test_accuracies[-1]),
		color      = 'r',
		arrowprops = None
	)

	plt.tight_layout()
	plt.show()

def plot_confusion_matrix(labels: list, preds: list) -> None:
	"""
	Plots a confusion matrix using the provided true labels and predicted labels.

	Args:
		labels (list): A list of true labels.
		preds (list): A list of predicted labels.

	Returns:
		None
	"""

	conf_matrix = confusion_matrix(labels, preds)

	plt.figure(figsize=(10, 8))
	sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')
	plt.show()