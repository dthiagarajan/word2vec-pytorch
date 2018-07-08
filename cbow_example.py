import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import *
from neg_loss import *
from utils import *

progressbar.streams.wrap_stderr()

CONTEXT_SIZE = 2
EMBEDDING_DIM = 2
NUM_EPOCHS = 5
NEGATIVE_SAMPLING = True

filename = "medium_text.txt"
print("Parsing text and loading training data...")
processed_text, vocab, word_to_ix, ix_to_word, training_data = load_data(filename,
                                                             CONTEXT_SIZE, model_type="cbow", subsampling=True, sampling_rate=0.001)

losses = []
if NEGATIVE_SAMPLING: 
    loss_function = NEGLoss(ix_to_word, vocab)
else:
    loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

print("Starting training")
for epoch in range(NUM_EPOCHS):
    total_loss = torch.Tensor([0])
    print("Beginning epoch %d" % epoch)
    progress_bar = progressbar.ProgressBar()
    for context, target in progress_bar(training_data):
        context_var = autograd.Variable(torch.LongTensor(context))
        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([target])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print("Epoch %d Loss: %.5f" % (epoch, total_loss[0]))
    losses.append(total_loss)

# Visualize embeddings
if EMBEDDING_DIM == 2:
    indices = np.random.choice(np.arange(len(vocab)), size=10, replace=False)
    for ind in indices:
        word = list(vocab.keys())[ind]
        input = autograd.Variable(torch.LongTensor([word_to_ix[word]]))
        vec = model.embeddings(input).data[0]
        x, y = vec[0], vec[1]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.savefig("w2v.png")
