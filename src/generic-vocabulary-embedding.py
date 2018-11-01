"""Generate intial word embedding for headlines and description.

The embedding is limited to a fixed vocabulary size (`vocab_size`) but
a vocabulary of all the words that appeared in the data is built.
"""
from os import path
import config
import argparse
import _pickle as pickle
from collections import Counter
import numpy as np

from prep_data import plt

# static vars
FN = 'vocabulary-embedding'
seed = 42
vocab_size = 40000
embedding_dim = 300
lower = False

# index words
empty = 0  # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos + 1  # first real word

# set random seed
np.random.seed(seed)


def build_vocab(lst, vocab_file):
    """Return vocabulary for iterable `lst`."""
    vocab_count = Counter(w for txt in lst for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
    if vocab_file is not None:
        constrained_vocab = np.load(vocab_file)
        vocab = [word for word in vocab if word in constrained_vocab]
    vocab_size = len(vocab)
    return vocab, vocab_count


def load_text():
    """Return vocabulary for pickled headlines and descriptions."""
    # read tokenized headlines and descriptions
    with open(path.join(config.path_data, 'tokens.pkl'), 'rb') as fp:
        headlines, desc = pickle.load(fp)

    # map headlines and descriptions to lower case
    if lower:
        headlines = [h.lower() for h in headlines]
        desc = [h.lower() for h in desc]

    return headlines, desc


def print_most_popular_tokens(vocab):
    """Print th most popular tokens in vocabulary dictionary `vocab`."""
    print('Most popular tokens:')
    print(vocab[:50])
    print('Total vocab size: {:,}'.format(len(vocab)))


def plot_word_distributions(vocab, vocab_count):
    """Plot word distribution in headlines and discription."""
    plt.plot([vocab_count[w] for w in vocab])
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_yscale("log", nonposy='clip')
    title = 'word distribution in headlines and discription'
    plt.title(title)
    plt.xlabel('rank')
    plt.ylabel('total appearances')
    plt.savefig(path.join(config.path_outputs, '{}.png'.format(title)))


def get_idx(vocab):
    """Add empty and end-of-sentence tokens to vocabulary and return tuple (vocabulary, reverse-vocabulary)."""
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx, word) for word, idx in word2idx.items())
    return word2idx, idx2word


def get_embeddings(emb_file):
    """Load embedding weights and indices."""
    emb_n_symbols = sum(1 for line in open(emb_file))
    print('{:,} Emb symbols'.format(emb_n_symbols))

    # load embedding weights and index dictionary
    emb_index_dict = {}
    emb_embedding_weights = np.empty((emb_n_symbols, embedding_dim))
    globale_scale = .1
    with open(emb_file, 'r') as fp:
        i = 0
        for l in fp:
            l = l.strip().split()
            w = l[0]
            emb_index_dict[w] = i
            emb_embedding_weights[i, :] = list(map(float, l[1:]))
            i += 1
    emb_embedding_weights *= globale_scale
    print('GloVe std dev: {:.4f}'.format(emb_embedding_weights.std()))

    # add lower case version of the keys to the dict
    for w, i in emb_index_dict.items():
        w = w.lower()
        if w not in emb_index_dict:
            emb_index_dict[w] = i

    return emb_embedding_weights, emb_index_dict


def initialize_embedding(vocab_size, embedding_dim, embedding_weights):
    """Use GloVe to initialize random embedding matrix with same scale as glove."""
    shape = (vocab_size, embedding_dim)
    scale = embedding_weights.std() * np.sqrt(12) / 2  # uniform and not normal
    embedding = np.random.uniform(low=-scale, high=scale, size=shape)
    print('random-embedding/glove scale: {:.4f} std: {:.4f}'.format(scale, embedding.std()))
    return embedding


def copy_emb_weights(embedding, idx2word, embedding_weights, emb_index_dict):
    """Copy from embs weights of words that appear in our short vocabulary (idx2word)."""
    c = 0
    for i in range(vocab_size):
        w = idx2word[i]
        g = emb_index_dict.get(w, emb_index_dict.get(w.lower()))
        if g is None and w.startswith('#'):  # glove has no hastags (I think...)
            w = w[1:]
            g = emb_index_dict.get(w, emb_index_dict.get(w.lower()))
        if g is not None:
            embedding[i, :] = embedding_weights[g, :]
            c += 1
    print('number of tokens, in small vocab: {:,} found in embeddings and copied to embedding: {:.4f}'.format(c, c / float(vocab_size)))
    return embedding


def build_word_to_embedding(embedding, word2idx, idx2word, embedding_index_dict, embedding_weights):
    """Map full vocabulary to embeddings based on cosine distance."""
    embedding_thr = 0.5
    word2embedding = {}
    for w in word2idx:
        if w in embedding_index_dict:
            g = w
        elif w.lower() in embedding_index_dict:
            g = w.lower()
        elif w.startswith('#') and w[1:] in embedding_index_dict:
            g = w[1:]
        elif w.startswith('#') and w[1:].lower() in embedding_index_dict:
            g = w[1:].lower()
        else:
            continue
        word2embedding[w] = g

    # for every word outside the embedding matrix find the closest word inside the embedding matrix.
    # Use cos distance of Embedding vectors.
    # Allow for the last `nb_unknown_words` words inside the embedding matrix to be considered to be outside.
    # Dont accept distances below `embedding_thr`
    normed_embedding = embedding / np.array(
        [np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]

    nb_unknown_words = 100

    embedding_match = []
    for w, idx in word2idx.items():
        if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2embedding:
            gidx = embedding_index_dict[word2embedding[w]]
            gweight = embedding_weights[gidx, :].copy()

            # find row in embedding that has the highest cos score with gweight
            gweight /= np.sqrt(np.dot(gweight, gweight))
            score = np.dot(normed_embedding[:vocab_size - nb_unknown_words], gweight)
            while True:
                embedding_idx = score.argmax()
                s = score[embedding_idx]
                if s < embedding_thr:
                    break
                if idx2word[embedding_idx] in word2embedding:
                    embedding_match.append((w, embedding_idx, s))
                    break
                score[embedding_idx] = -1

    embedding_match.sort(key=lambda x: -x[2])
    print()
    print('# of Embedding substitutes found: {:,}'.format(len(embedding_match)))

    # manually check that the worst substitutions we are going to do are good enough
    for orig, sub, score in embedding_match[-10:]:
        print('{:.4f}'.format(score), orig, '=>', idx2word[sub])

    # return a lookup table of index of outside words to index of inside words
    return dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in embedding_match)


def to_dense_vector(word2idx, corpus, description, bins=50):
    """Create a dense vector representation of headlines."""
    data = [[word2idx[token] for token in txt.split()] for txt in corpus]
    plt.hist(list(map(len, data)), bins=bins)
    plt.savefig(path.join(config.path_outputs, '{}_distribution.png'.format(description)))
    return data


def summarize_vocab(vocab, vocab_count):
    """Print the most popular tokens and plot token distributions."""
    print_most_popular_tokens(vocab)
    plot_word_distributions(vocab, vocab_count)


def main(emb_file, vocab_file, emb_type):
    """Generate intial word embedding for headlines and description."""
    headlines, desc = load_text()  # load headlines and descriptions
    vocab, vocab_count = build_vocab(headlines + desc, vocab_file)  # build vocabulary
    summarize_vocab(vocab, vocab_count)  # summarize vocabulary
    word2idx, idx2word = get_idx(vocab)  # add special tokens and get reverse vocab lookup
    glove_embedding_weights, glove_index_dict = get_embeddings(emb_file)  # load GloVe data

    # initialize embedding
    embedding = initialize_embedding(vocab_size, embedding_dim, glove_embedding_weights)
    embedding = copy_emb_weights(embedding, idx2word, glove_embedding_weights, glove_index_dict)

    # map vocab to GloVe using cosine similarity
    glove_idx2idx = build_word_to_embedding(embedding, word2idx, idx2word, glove_index_dict, glove_embedding_weights)

    # create a dense vector representation of headlines and descriptions
    description_vector = to_dense_vector(word2idx, desc, 'description')
    headline_vector = to_dense_vector(word2idx, headlines, 'headline')

    # write vocabulary to disk
    with open(path.join(config.path_data, '{}-{}.pkl'.format(emb_type, FN)), 'wb') as fp:
        pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, 2)

    # write data to disk
    with open(path.join(config.path_data, '{}-{}.data.pkl'.format(emb_type, FN)), 'wb') as fp:
        pickle.dump((description_vector, headline_vector), fp, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', type=str, default=None, help='file with embeddings')
    parser.add_argument('--emb_type', type=str, default=None, help='type of embeddings')
    parser.add_argument('--vocab_file', type=str, default=None, help='file with vocabulary')
    args = parser.parse_args()
    main(emb_file=args.emb_file, vocab_file=args.vocab_file, emb_type=args.emb_type)
