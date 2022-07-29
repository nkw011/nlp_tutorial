import torch

def encode(data, token2id, tokenizer):
    encoded = [ torch.tensor(list(map(lambda x: token2id.get(x, token2id['<unk>']), tokens))).long() for tokens in map(tokenizer, data)]
    return torch.cat(encoded)

def decode(id_sequence, id2token):
    return " ".join([ id2token[s_id] for s_id in id_sequence ])

def batchfy(data, batch_size, seq_len):
    samples = data.size()[0] // (batch_size * seq_len)
    data = data[:samples*batch_size*seq_len]
    data = data.view(batch_size,-1,seq_len).transpose(0,1)
    return data