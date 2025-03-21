import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
blockSize = 8 # what is the maximum context length for predictions?
max_iters = 5000
evalInterval = 500
learningRate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
evalIters = 200
nEmbd = 32
nHead = 4
nLayers = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocabSize = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blockSize, (batch_size,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+blockSize+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class LayerNorm1D:
    """ example code """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
    def __call__(self, x):
        # calculate the forward pass 
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Head(nn.Module):
    """ Creating one Head in self attention """

    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbd, headSize, bias=False)
        self.query = nn.Linear(nEmbd, headSize, bias=False)
        self.value = nn.Linear(nEmbd, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))
        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B,T,T
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.droupout(wei)
        # perform the weighted aggregation of the values 
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize) for _ in range(numHeads)])
        self.proj = nn.Linear(nEmbd, nEmbd)
        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # at this point this is similar to group convolution
        out = self.droupout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity  """

    def __init__(self, nEmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbd, 4 * nEmbd), 
            nn.ReLU(), 
            nn.Linear(4 * nEmbd, nEmbd),
            nn.Dropout(dropout),
            ) 

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation(understanding) """

    def __init__(self, nEmbd, nHead):
        # nEmbd: Embedding dimension, nHead: the number of heads we'd like
        super().__init__()
        headSize = nEmbd // nHead
        self.sa = MultiHeadAttention(nHead, headSize)
        self.ffwd = FeedFoward(nEmbd)
        self.ln1 = nn.LayerNorm(nEmbd)
        self.ln2 = nn.LayerNorm(nEmbd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # create token embeddings
        self.token_embedding_table = nn.Embedding(vocabSize, nEmbd) # dont directly recreate logits we do intermediate phase 
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd) # positional embeddings
        self.blocks = nn.Sequential(*[Block(nEmbd, nHead=nHead) for _ in range(nLayers)])
        self.lnf = nn.LayerNorm(nEmbd) # final layer norm
        # from token to logits we need a linear.
        self.lmHead = nn.Linear(nEmbd, vocabSize)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tokenEmbd = self.token_embedding_table(idx) # (B,T,C)
        positionEmbd = self.positionEmbeddingTable(torch.arange(T, device=device)) # (T,C)
        x = tokenEmbd + positionEmbd #(B, T, C)
        # x = self.saHead(x) # (B, T, C)
        # x = self.ffwd(x) #(B, T, C)
        x = self.blocks(x) #(B, T, C)
        logits = self.lmHead(x) # (B,T,vocabSize)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #check if the idx is more than blocksize
            idxCond = idx[:, -blockSize:]
            # get the predictions
            logits, loss = self(idxCond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % evalInterval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))