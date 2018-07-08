import numpy as np

from nltk.tokenize import word_tokenize

def gather_word_freqs(split_text, subsampling = False, sampling_rate = 0.001):
    vocab = {}
    ix_to_word = {}
    word_to_ix = {}
    total = 0.0
    for word in split_text:
        if word not in vocab:
            vocab[word] = 0
            ix_to_word[len(word_to_ix)] = word
            word_to_ix[word] = len(word_to_ix)
        vocab[word] += 1.0
        total += 1.0
    if subsampling:
        for i, word in enumerate(split_text):
            val = np.sqrt(sampling_rate * total / vocab[word])
            prob = val * (1 + val)
            sampling = np.random.sample()
            if (sampling <= prob):
                del [split_text[i]]
                i -= 1
    return split_text, vocab, word_to_ix, ix_to_word

def gather_training_data(split_text, word_to_ix, context_size, model_type = "skipgram"):
    training_data = []
    for i, word in enumerate(split_text):
        if (model_type == "skipgram"):
            back_i = i - 1
            back_c = 0
            forw_i = i + 1
            forw_c = 0
            while (back_i >= 0 and back_c < context_size):
                training_data.append(([word_to_ix[word]], word_to_ix[split_text[back_i]]))
                back_i -= 1
                back_c += 1
            while (forw_i < len(split_text) and forw_c < context_size):
                training_data.append(([word_to_ix[word]], word_to_ix[split_text[forw_i]]))
                forw_i += 1
                forw_c += 1
        elif (model_type == "cbow"):
            point = []
            back_i = i - 1
            back_c = 0
            forw_i = i + 1
            forw_c = 0
            while (back_i >= 0 and back_c < context_size):
                point.append(word_to_ix[split_text[back_i]])
                back_i -= 1
                back_c += 1
            while (forw_i < len(split_text) and forw_c < context_size):
                point.append(word_to_ix[split_text[forw_i]])
                forw_i += 1
                forw_c += 1
            training_data.append((point, word_to_ix[word]))
        else:
            raise ValueError("Inappropriate argument value for model_type - either `skipgram` or `cbow`.")
    return training_data

def load_data(filename, context_size, model_type = "skipgram", subsampling = False, sampling_rate = 0.001):
    with open(filename, "rb") as file:
        processed_text = word_tokenize(file.read().decode("utf-8").strip())
        processed_text, vocab, word_to_ix, ix_to_word = gather_word_freqs(processed_text,
            subsampling = subsampling, sampling_rate = sampling_rate)
        training_data = gather_training_data(processed_text, word_to_ix, context_size,
                                             model_type = model_type)
        return processed_text, vocab, word_to_ix, ix_to_word, training_data
