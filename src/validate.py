"""Predict a title for a recipe."""
from os import path
import random
import json
import pickle
import h5py
import numpy as np
from utils import str_shape
import keras.backend as K
import argparse
import pandas as pd

from config import path_models, path_data
from constants import FN1, FN0, nb_unknown_words, eos
from model import create_model
from sample_gen import gensamples
from rouge import Rouge

# set seeds in random libraries
seed = 42
random.seed(seed)
np.random.seed(seed)


def load_weights(model, filepath):
    """Load all weights possible into model from filepath.

    This is a modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print('Loading', filepath, 'to', model.name)
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except:
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values


def main(n_samples=None):
    """Predict a title for a recipe."""
    # load model parameters used for training
    with open(path.join(path_models, 'model_params.json'), 'r') as f:
        model_params = json.load(f)

    # create placeholder model
    model = create_model(**model_params)

    # load weights from training run
    load_weights(model, path.join(path_models, '{}.hdf5'.format(FN1)))

    # load recipe titles and descriptions
    with open(path.join(path_data, 'vocabulary-embedding.data.pkl'), 'rb') as fp:
        X_data, Y_data = pickle.load(fp)

    # load vocabulary
    with open(path.join(path_data, '{}.pkl'.format(FN0)), 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    vocab_size, embedding_size = embedding.shape
    oov0 = vocab_size - nb_unknown_words

    rouge = Rouge()
    rouge1_f = []
    rouge1_p = []
    rouge1_r = []
    rouge2_f = []
    rouge2_p = []
    rouge2_r = []
    rougel_f = []
    rougel_p = []
    rougel_r = []
    orig = []
    gen = []


    for index in range(n_samples):
        # load random recipe description if none provided
        i = np.random.randint(len(X_data))
        sample_str = ''
        sample_title = ''
        for w in X_data[i]:
            sample_str += idx2word[w] + ' '
        for w in Y_data[i]:
            sample_title += idx2word[w] + ' '
        y = Y_data[i]
        print('Randomly sampled recipe:')
        print(sample_title)
        print(sample_str)


        x = [word2idx[w.rstrip('^')] for w in sample_str.split()]

        samples = gensamples(
            skips=2,
            k=1,
            batch_size=2,
            short=False,
            temperature=1.,
            use_unk=True,
            model=model,
            data=(x, y),
            idx2word=idx2word,
            oov0=oov0,
            glove_idx2idx=glove_idx2idx,
            vocab_size=vocab_size,
            nb_unknown_words=nb_unknown_words,
        )

        headline = samples[0][0][len(samples[0][1]):]
        gen_head = ' '.join(idx2word[w] for w in headline)
        orig_head = ' '.join(idx2word[w] for w in y)
        rouge_scores = rouge.get_scores(gen_head, orig_head)
        orig.append(orig_head)
        gen.append(gen_head)
        {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
         'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
        rouge1_f.append(rouge_scores[0]['rouge-1']['f'])
        rouge1_p.append(rouge_scores[0]['rouge-1']['p'])
        rouge1_r.append(rouge_scores[0]['rouge-1']['r'])
        rouge2_f.append(rouge_scores[0]['rouge-2']['f'])
        rouge2_p.append(rouge_scores[0]['rouge-2']['p'])
        rouge2_r.append(rouge_scores[0]['rouge-2']['r'])
        rougel_f.append(rouge_scores[0]['rouge-l']['f'])
        rougel_p.append(rouge_scores[0]['rouge-l']['p'])
        rougel_r.append(rouge_scores[0]['rouge-l']['r'])
        print(rouge_scores)
    validation_data = {
        'rouge1-f': rouge1_f,
        'rouge1-p': rouge1_p,
        'rouge1-r': rouge1_r,
        'rouge2-f': rouge2_f,
        'rouge2-p': rouge2_p,
        'rouge2-r': rouge2_r,
        'rougel-f': rougel_f,
        'rougel-p': rougel_p,
        'rougel-r': rougel_r,
        'orig': orig,
        'gen': gen
    }
    validation_df = pd.DataFrame(validation_data)
    validation_df.to_csv(path.join(path_models, 'validation-{}.dat'.format(FN0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1, help='number of samples')
    args = parser.parse_args()
    main(n_samples=args.n_samples)
