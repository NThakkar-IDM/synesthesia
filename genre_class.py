""" genre_class.py

A PyTorch based convolutional neural net for genre classification, based on 
Spotify's implementation (described here: http://benanne.github.io/2014/08/05/spotify-cnns.html) 
Also worth noting: https://hackernoon.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194"""

## Standard imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## For progress bars
from tqdm import tqdm

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 28.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Garamond"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

## Convolutional neural net classes
###########################################################################
class GenreClassifier2D(nn.Module):

	""" A genre classifier that treats the spectrogram as an image. This is based on the example
	here: https://hackernoon.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194 """

	def __init__(self):

		## Inherit from the base class
		super(GenreClassifier2D, self).__init__()

		## Network structure
		## Initial pooling
		self.avg = nn.AvgPool1d(4,stride=4)

		## Convolutional layers
		self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=2)
		self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2)
		self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=2)
		self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=2)

		## Fully connected layers
		self.fully_connected1 = nn.Linear(512*7*7,1024)
		self.fully_connected2 = nn.Linear(1024,8)

		## Regularization
		self.drop_out = nn.Dropout(p=0.5)

	def forward(self,x):

		## Scale to a 128 by 128 image, telling PyTorch
		## it's a single image channel
		x = self.avg(x).unsqueeze(1)

		## Apply the convolutions
		x = F.max_pool2d(F.elu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.elu(self.conv2(x)),(2,2))
		x = F.max_pool2d(F.elu(self.conv3(x)),(2,2))
		x = F.max_pool2d(F.elu(self.conv4(x)),(2,2))

		## Map to the fully connected layers
		x = x.view(-1, 512*7*7)
		x = F.elu(self.fully_connected1(x))
		x = self.drop_out(x)
		x = self.fully_connected2(x)

		return x

class GenreClassifier1D(nn.Module):

	""" PyTorch based deep-net classifier, adapted from the implementation found here:
	https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html 
	to the audio case. 
	
	This class has the 2 required functions,

	self.__init__: This specifies the network topology
	self.forward: This specifies how input data is propogated through
				  the network.

	self.forward is automatically differentiated, allowing the model to be fit
	with mini-batch gradient descent."""

	def __init__(self):

		## Inherit from PyTorch's NN module
		super(GenreClassifier1D, self).__init__()

		## Set up the network topology using PyTorch's
		## built in layer classes. The number of elements
		## in each layer is hard coded in this function.
		self.conv1 = nn.Conv1d(in_channels=128,out_channels=64,kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=2)
		self.fully_connected1 = nn.Linear(in_features=32*3,out_features=8)

	def forward(self,x):

		## Output from the first layer is pooled and then passed through
		## the convolution.
		x = F.max_pool1d(torch.tanh(self.conv1(x)),2,stride=2)
		x = F.max_pool1d(torch.tanh(self.conv2(x)),2,stride=2)

		## Output from the first layer is pooled globally 3 ways to
		## create the next layer. The results are then concatenated
		## to create a fully connected layer input.
		time_steps = x.shape[-1]
		x = torch.cat((F.avg_pool1d(x,time_steps),
					   F.max_pool1d(x,time_steps),
					   F.lp_pool1d(x,2,time_steps)))
		x = x.view(-1,32*3)

		## Pass through the fully connected layer
		x = self.fully_connected1(x)

		return x

## Preparing the data for ML
###########################################################################
def GetSpectrograms(df,entropy_loss=True):

	""" Simple loop over spectrograms to create the input stack and return torch.tensor
	classes of the inputs and outputs. """

	## Loop over spectrograms, highlighting missing songs due to corrupted 
	## files. When the spectrogram is loaded, data is sliced and scaled.
	X = []
	for track_id in df.index:
		fname = "_spectrograms\\"+str(track_id)+".npz"
		try:
			x = np.load(fname)["log_mel"][:,1000:1512]
			x = x/x.max()
		except FileNotFoundError:
			raise FileNotFoundError("Track {} is a corrupted file and must be excluded!".format(track_id))
		X.append(x)

	## Convert to torch tensors. Entropy loss functions
	## assume Y is a vector of genre ids while MSE-style loss functions
	## take the one-hot representation.
	X = torch.Tensor(np.stack(X))
	if entropy_loss:
		Y = torch.tensor(np.argmax(df.values,axis=1))
	else:
		Y = torch.tensor(df.values)

	return X, Y

if __name__ == "__main__":

	## Test
	df = pd.read_pickle("_data\\response.pkl")

	## Drop entries associated with corrupted MP3 or low resolution files.
	corrupted = ["29350","99134","108925","133297","17631","17632","17633","17634","17635","107535",
				 "17636","17637","29355","54568","54576","54578","55783","98565","98567","98569","136928"]
	df = df[~df.index.isin(corrupted)]

	## Get the training and validation sets, randomizing the order of the 
	## training data
	training_df = df[df["split"] == "training"].drop("split",axis=1).sample(frac=1)
	validation_df = df[df["split"] == "validation"].drop("split",axis=1)

	## Get the validation spectrograms. For the training set, this is
	## done in mini-batches for memory reasons.
	X_validate, Y_validate = GetSpectrograms(validation_df)

	## Initialize the net
	net = GenreClassifier2D()

	## Set up the SGD parameters
	mini_batch_size = 32
	num_epochs = 10
	learning_rate = 0.01
	momentum = 0.9
	optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum)

	## Set up the loss function
	loss_function = torch.nn.CrossEntropyLoss()

	## Calculate the total number of mini-batches
	num_batches = int(np.ceil(len(training_df)/mini_batch_size))
	print("Training data is {} spectrograms, split into {} mini batches.".format(len(training_df),num_batches))

	## Check performance on the validation set
	with torch.no_grad():
		Y_valid_prediction = torch.softmax(net(X_validate),dim=1)
		classifications = torch.argmax(Y_valid_prediction,dim=1)
		num_correct = (classifications == Y_validate).sum().item()
	print("...validation accuracy = {}/{}\n".format(num_correct,len(Y_validate)))

	## Training loop (applying SGD on mini batches)
	training_loss = []
	validation_accuracy = [num_correct/len(Y_validate)]
	for epoch in range(num_epochs):

		print("Starting epoch {}...".format(epoch))

		## Loop over mini-batches
		batch_losses = 0.
		for i, batch in tqdm(enumerate(np.array_split(training_df,num_batches))):

			## Get the spectrograms and genres for this batch
			X_batch, Y_batch = GetSpectrograms(batch)

			## Initialize the optimizer
			optimizer.zero_grad()

			## Compute the loss
			Y_pred = net(X_batch)
			loss = loss_function(Y_pred,Y_batch)

			## Update the parameters by back computing
			## the gradient with respect to the loss function.
			loss.backward()
			optimizer.step()

			## Update the running error estimate
			batch_losses += loss.item()

		## Store the training loss
		training_loss.append(batch_losses)
		print("...training loss = {}".format(training_loss[-1]))

		## Check performance on the validation set
		with torch.no_grad():
			Y_valid_prediction = torch.softmax(net(X_validate),dim=1)
			classifications = torch.argmax(Y_valid_prediction,dim=1)
			print(classifications)
			num_correct = (classifications == Y_validate).sum().item()
			validation_accuracy.append(num_correct/len(Y_validate))
		print("...validation accuracy = {}/{}\n".format(num_correct,len(Y_validate)))

		## Randomize the training data
		training_df = training_df.sample(frac=1)

	## Plot the training and validation scores across 
	## epochs.
	fig, axes = plt.subplots(figsize=(14,12))
	axes.grid(color="grey",alpha=0.3)
	axes.plot(np.arange(num_epochs),training_loss,
			  color="k",marker="s",markersize=13,lw=2)
	axes.set(xlabel="Epoch",ylabel="Cross entropy loss")
	plt.tight_layout()
	plt.savefig("_plots\\training_loss.png")

	fig, axes = plt.subplots(figsize=(14,12))
	axes.grid(color="grey",alpha=0.3)
	axes.plot([-1,0],validation_accuracy[:2],marker="o",ls="dashed",color="k")
	axes.plot(np.arange(num_epochs),validation_accuracy[1:],
			  color="C3",marker="o",markersize=13,lw=2)
	axes.set(xlabel="Epoch",ylabel="Fraction correct")
	plt.tight_layout()
	plt.savefig("_plots\\validation_accuracy.png")

	plt.show()





