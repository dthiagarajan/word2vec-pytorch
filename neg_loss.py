import torch
import torch.nn as nn
import torch.nn.functional as F

class NEGLoss(nn.Module):
    def __init__(self, ix_to_word, word_freqs, num_negative_samples=5,):
        super(NEGLoss, self).__init__()
        self.num_negative_samples = num_negative_samples
        self.num_words = len(ix_to_word)
        self.distr = F.normalize(torch.Tensor(
            [word_freqs[ix_to_word[i]] for i in range(len(word_freqs))]).pow(0.75), dim=0
        )

    def sample(self, num_samples, positives=[]):
        weights = torch.zeros((self.num_words, 1))
        for w in positives: weights[w] += 1.0
        for _ in range(num_samples):
            w = torch.multinomial(self.distr, 1)[0]
            while (w in positives):
                w = torch.multinomial(self.distr, 1)[0]
            weights[w] += 1.0
        return weights

    def forward(self, input, target):
        return F.nll_loss(input, target,
            self.sample(self.num_negative_samples, positives=target.data.numpy()))
