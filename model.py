import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

''' Continuous bag-of-words model for word2vec.

Parameters:
    vocab_size: number of defined words in the vocab
    embedding_dim: desired embedded vector dimension
    context_size: number of context words used

'''
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out)
        return log_probs


''' Skip-gram bag-of-words model for word2vec.

Parameters:
    vocab_size: number of defined words in the vocab
    embedding_dim: desired embedded vector dimension

'''
class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out)
        return log_probs


