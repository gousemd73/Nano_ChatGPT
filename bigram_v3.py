#!/usr/bin/env python
# coding: utf-8
#downloading the dataset from github
# get_ipython().system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# CUDA_LAUNCH_BLOCKING=1
batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


#reading the input files....
with open('input.txt','r',encoding='utf-8') as f :
    text = f.read()
    
# print(len(text))
# print(text[:100])



#creating the vocabulary of the dataset with identifying the number of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


#tokenization
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[c] for c in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))


#encoding the dataset into a torch.tensor with defined tokenizer above
data = torch.tensor(encode(text),dtype=torch.long)
# print(data.shape)
# print(data[:10])


#train and validation dataset
n = int(.9*len(text))
train_data = data[:n]
val_data   = data[n:]

train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target  = y[t]
#     print(f"When input is {context} the target : {target}")


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


class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        
        self.key    = nn.Linear(n_embed,head_size,bias=False)
        self.value  = nn.Linear(n_embed,head_size,bias=False)
        self.query  = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):

        B,T,C = x.shape

        k = self.key(x) # (B,T,hs)
        q = self.query(x) #(B,T,hs)

        wei = q @ k.transpose(-2,-1) # (B,T,hs) (B,T,hs) -----> (B,T,T)
        wei = wei * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        v  = self.value(x)
        out = wei@v #(B,T,T) , (B,T,hs) ---> (B,T,hs)

        return out 

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out 
    


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed) , #skip connection from the multihead attention
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
    


class Block(nn.Module):

    def __init__(self,n_embed,n_head) :
        super().__init__()
        head_size = n_embed // n_head
        self.mas  = MultiHeadAttention(n_head,head_size=head_size)
        self.ffn  = FeedForward(n_embed=n_embed)
        self.ln1  = nn.LayerNorm(n_embed) #Layer normalization
        self.ln2  = nn.LayerNorm(n_embed) #Layer Normalization
    
    def forward(self,x):

        x  =  x + self.mas(self.ln1(x)) #applying LN before sending it into Multi-Head attention
        x  =  x + self.ffn(self.ln2(x)) # applying LN before sending it to FeedForward Network

        return x




#bi-gram language model....
class BigramLM(nn.Module):
    
    def __init__(self,):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)  #(B,T) ---> (B,T,n_embeds) 
        self.positional_embedding_table = nn.Embedding(block_size,n_embed) #(B,T) ---> (T,n_embds)
        # self.sa_head = Head(4,head_size) #(B,T,n_embds) ----> (B,T,head_size)
        # self.msa_head  = MultiHeadAttention(num_heads=4,head_size=16//4) # i.e 4 heads with 8-dimensional head size of self attention 
        
        self.block = nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])
        
        
        # self.ffn = FeedForward(n_embed) #input dimension for FFN should be num_head*head_size
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size) #(B,T,head_size) ---> (B,T,vocab_size)



    def forward(self,idx,targets=None):
        
        B,T = idx.shape
        #logits and targets are both (B,T) tensors of integers
        tokn_embd = self.token_embedding_table(idx) #outputs of size (B,T,embed_size)
        pos_embd  = self.positional_embedding_table(torch.arange(T,device=device)) # (T,embed_size)
        x = tokn_embd + pos_embd

        # x = self.sa_head(x) #apply one self-attention head (B,T,C) ---> (B,T,head_size)
        # x = self.msa_head(x) #(B,T,emd_size) ---> (B,T,n_head*head_size)
       
        x = self.block(x)
        
        # x = self.ffn(x) #(B,T,n_heads*head_size)
        
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # print(logits.shape)
        # print(f"x:{idx.shape},tok:{tokn_embd.shape},pos:{pos_embd.shape},sa:{x.shape},log:{logits.shape}")
        
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
            #crop the additional blocksize
            idx_cond = idx[:,-block_size:]
            #get the prediction
            logits,loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax to get the probabilities
            probs = F.softmax(logits,dim=1)
            #sample from the distribution
            idx_nxt = torch.multinomial(probs,num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx,idx_nxt),dim=1) #(B,T+1)
        return idx



model = BigramLM()
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
predictions= decode(m.generate(context,10000)[0].tolist())

print(predictions)




