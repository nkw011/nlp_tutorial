import torch
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch_samples):
    pad_token_id = Language.pad_token_id

    src_sentences = pad_sequence([torch.tensor(src).long() for src, _ in batch_samples], batch_first=True, padding_value=pad_token_id)
    tgt_sentences = pad_sequence([torch.tensor(tgt).long() for _, tgt in batch_samples], batch_first=True, padding_value=pad_token_id)

    return src_sentences, tgt_sentences

def batch_sampling(sequence_lengths, batch_size):
    '''
    sequence_length: (source 길이, target 길이)가 담긴 리스트이다.
    batch_size: batch 크기
    '''

    seq_lens = [(i, seq_len, tgt_len) for i,(seq_len, tgt_len) in enumerate(sequence_lengths)]
    seq_lens = sorted(seq_lens, key=lambda x: x[1])
    seq_lens = [sample[0] for sample in seq_lens]
    sample_indices = [ seq_lens[i:i+batch_size] for i in range(0,len(seq_lens), batch_size)]

    random.shuffle(sample_indices) # 모델이 길이에 편향되지 않도록 섞는다.

    return sample_indices

def make_dataloader(dataset, batch_size):
    sequence_lengths = list(map(lambda x: (len(x[0]), len(x[1])), dataset))
    batch_sampler = batch_sampling(sequence_lengths, batch_size)

    return DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler)