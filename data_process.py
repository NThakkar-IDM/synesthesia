""" data_process.py

Functions to create mel-spectrograms, load data, meta-data,
etc. all based on librosa. """

## Standard imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 28.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.serif"] = "Garamond"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

## For working with audio
import librosa
import librosa.display
from librosa.feature import melspectrogram

## Meta-data related
import os

## For progress bars
from tqdm import tqdm

### Meta-data related functions
###########################################################################################################
def GetSongFilenames():

	""" This function simply parses the files in fma_small. It's not a very flexible
	function, but it retrieves useful data. """

	## Loop through each directory
	song_files = []
	for root, dirs, fnames in os.walk("_data\\fma_small\\"):
		
		## Skip the first level
		if root == "_data\\fma_small\\":
			continue

		## Otherwise collect the files, appending
		## the root path.
		song_files += [root+"\\"+f for f in fnames]

	return song_files

def GetTrackMetaData():

	""" Get the track.csv file as a pandas df. A little extra work has to be done
	here to handle the formating of the track_id column."""
	
	## Get the raw file. The low_memory option
	## suppresses a warning regarding mismatched datatypes in
	## the track_id column. That's due to spacing in the original file.
	df = pd.read_csv("_data\\fma_metadata\\tracks.csv",header=1,index_col=0,low_memory=False)
	
	## Fix the track_id column by dropping
	## the problematic row and renaming.
	df = df.drop("track_id",axis=0)
	df.index.rename("track_id",inplace=True)

	return df

def GetGenreMetaData():

	""" Use panda's csv reader to get the genre metadata."""
	
	df = pd.read_csv("_data\\fma_metadata\\genres.csv",header=0)
	return df


if __name__ == "__main__":

	## Set a mode for the script
	## 1: Make some plots, simple example
	## 2: Loop over fnames and create serialized spectrograms
	## 3: 1-hot representation of the top genres
	## 4: distributional representation of the genres (using full genre list)
	_mode = 1

	if _mode == 1:

		test_path = "_data\\fma_small\\006\\006357.mp3"

		## Load the data
		audio, sr = librosa.load(test_path,sr=None,mono=True)
		print('Duration: {:.2f}s, {} samples'.format(audio.shape[-1]/sr,audio.size))

		## Create a wave plot
		fig, axes = plt.subplots(figsize=(16,8))
		librosa.display.waveplot(audio,sr=sr,color="grey",alpha=0.5)
		plt.tight_layout()
		plt.savefig("_plots\\waveplot.png")
		
		## Compute a mel-spectrogram
		mel = melspectrogram(audio,sr=sr)
		log_mel = librosa.power_to_db(mel)#[:,1000:1512]#,ref=np.max)

		## Plot it
		plt.figure(figsize=(16,8))
		librosa.display.specshow(log_mel,sr=sr,
								 y_axis='mel',x_axis='time')
		plt.colorbar(format="%+2.0f dB")
		plt.title("Mel spectrogram")
		plt.tight_layout()
		plt.savefig("_plots\\spectrogram.png")
		plt.show()

	elif _mode == 2:

		## Get all the songs' paths.
		song_fnames = GetSongFilenames()

		## Loop over songs and create log spectrograms.
		print("Computing {} spectrograms...".format(len(song_fnames)))
		for song in tqdm(song_fnames):

			## Get the audio and sampling rate
			audio, sr = librosa.load(song,sr=None,mono=True)

			## Compute the spectrogram
			mel = melspectrogram(audio,sr=sr)
			log_mel = librosa.power_to_db(mel)#,ref=np.max)

			## Create a NPZ name for the spectrogram
			track_id = int(song[-10:-4])
			npz_name = str(track_id)+".npz"

			## Save the spectrogram
			np.savez("_spectrograms\\"+npz_name,log_mel=log_mel)

	elif _mode == 3:

		## Get the relevant track dataframe
		track_df = GetTrackMetaData()[["split","subset","genre_top","genres"]]

		## Subset to the small dataset
		track_df = track_df[track_df["subset"] == "small"]

		## Create the 1-hot representation for the genres, where each
		## track id is associated with a vector of len = number of genres (8 here)
		## and each vector has a single non-zero entry corresponding to the appropriate genre.
		response = pd.get_dummies(track_df["genre_top"])
		response = pd.concat([response,track_df["split"]],axis=1)

		## Serialize the output
		pd.to_pickle(response,"_data\\response.pkl")
		print(response)




		