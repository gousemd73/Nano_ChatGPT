#!/usr/bin/env python
# coding: utf-8
#downloading the dataset from github
# get_ipython().system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#reading the input files....
with open('input.txt','r',encoding='utf-8') as f :
    text = f.read()
    
print(len(text))
print(text[:100])



#creating the vocabulary of the dataset with identifying the number of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


#tokenization
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[c] for c in l])

print(encode("hii there"))
print(decode(encode("hii there")))


#encoding the dataset into a torch.tensor with defined tokenizer above
data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape)
print(data[:10])


#train and validation dataset
n = int(.9*len(text))
train_data = data[:n]
val_data   = data[n:]

train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target  = y[t]
    print(f"When input is {context} the target : {target}")


def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x  = torch.stack([data[i:i+block_size] for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i  in ix])
    x  = x.to(device)
    y  = y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()
    
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#bi-gram language model....
class BigramLM(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets=None):
        
        #logits and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) #outputs of size (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,idx,max_tokens):
        #idx is the input in (B,T) shape in current context
        
        for _ in range(max_tokens):
            #get the prediction
            logits,loss = self(idx)
            #focus only on the last time step
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax to get the probabilities
            probs = F.softmax(logits,dim=1)
            #sample from the distribution
            idx_nxt = torch.multinomial(probs,num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx,idx_nxt),dim=1) #(B,T+1)
        return idx



model = BigramLM(vocab_size)
m = model.to(device)


#pytorch optimizer....
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

for iter  in range(max_iters):

    #every once in a while evaluate loss on train and val sets
    if iter%eval_interval==0:
        losses = estimate_loss()
        print(f"step{iter}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    #sample the batch of data
    xb,yb = get_batch('train')
    
    #eval loss
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1),dtype=torch.long,device=device)
predictions= decode(m.generate(context,100)[0].tolist())

print(predictions)




