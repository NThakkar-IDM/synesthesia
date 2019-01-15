# synesthesia
Deep learning to predict music genre from raw audio via spectrogram image processing.
> This repository is for the ML/DL group meeting on 1/15/2019. 

## Installation
To run this code, you need to install [LibROSA](https://librosa.github.io/librosa/), [PyTorch](https://pytorch.org/), and [tqdm](https://tqdm.github.io/) on top of Anaconda's Python 3.6 distribution.

## The data
I used the small dataset from the [Free Music Archive](https://github.com/mdeff/fma). Download the data and place it in the synesthesia directory in the subfolder called `_data`.
Three songs in the dataset were corrupted and deleted. They're easy to find via `data_process.py`, included in the repository.

## Getting started
To use the code, first precompute spectrograms using `data_process.py` with `_mode` = 2. This computation takes about 45 minutes on my laptop. Once that's done, you can experiment with the deep nets in `genre_class.py`. 

