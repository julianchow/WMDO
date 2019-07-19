# WMDO
Word Mover's Distance with word order penalty for automatic machine translation evaluation.

## Getting started
The `wmdo.py` file contains the code necessary to run the WMDO metric calculation. 

## Prerequisites
The code is written in Python 3, using the packages `numpy`, `sklearn`, `pyemd`, `nltk`, `gensim` and `bisect`. To use this metric, a word embedding is required. This can be a pre-trained embedding or a purpose-trained embedding, ideally of high dimension and vocabulary size for good results and high quality vectors.

## Calculation
The calculation of the metric is done through the `wmdo()` method, which takes several parameters. `wvvecs` are the word vectors, which can be loaded from the `load_wv` method. `ref` refers to the reference translation and `cand` the candidate translation, which are passed in as pre-processed strings. `missing` is a dictionary matching words not in the embedding to corresponding vectors - this can be initialised as `{}` or as a custom set of vectors. The `dim` parameter refers to the dimensionality of the word embedding. `delta` and `alpha` are the weights of the word order penalty and the missing word penalty, respectively. These can be tuned for each language, but from experiment it was found that 0.18 for `delta` and 0.10 for `alpha` is a good starting place.

## Results
The value returned from the metric calculation is a non-negative number. This is an error metric, where the lower the WMDO score the higher the translation quality, so should give a negative correlation if a series of translations are compared with human scores. 
