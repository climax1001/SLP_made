import torch.nn as nn
import torch
import sentencepiece
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import helpers

device = torch.device("cpu")

# 전체 문장 리스트 생성
def get_line(text_name):
    cnt = 0
    lines = []

    f = open(text_name, 'r', encoding='UTF-8')
    while(1):
        cnt += 1
        line = f.readline()
        if line == '':
            break
        line = line.replace('.\n','')
        line = line.replace('.','')


        lines.append(line)
    return lines

def get_trans_corpus(text_list : list):
    trans_corpus = []
    for i in range(0, len(text_list)-1):
        trans_corpus.extend(text_list[i].split(' '))

    common_words = Counter(trans_corpus).most_common()

    vocabulary = dict()
    for idx, word in enumerate(common_words):
        vocabulary[word[0]] = (idx + 1)
    print(len(vocabulary))
    return vocabulary

def tokenized_list(vocab : dict, text_list : list):
    tokenized = []
    seq_len = 0

    for text in text_list:
        sentence = []
        for word in text.split(' '):
            if word in vocab.keys():
                sentence.append(vocab[word])
        tokenized.append(sentence)

    for i in range(0,len(tokenized)-1):
        if seq_len < len(tokenized[i]):
            seq_len = len(tokenized[i])

    return tokenized, seq_len

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table
def vocab_transfered(filename):
    text_list = get_line(filename)
    toked, max_seq_len = tokenized_list(get_trans_corpus(text_list), text_list)
    tok_corpus = np.array(toked)
    tok_corpus = get_numpy_from_nonfixed_2d_array(tok_corpus, fixed_length=max_seq_len, padding_value=0)
    return tok_corpus



if __name__ == '__main__':
    input = vocab_transfered('text/phoenix2014T.train.de')
    input = torch.Tensor(input).to(device).long()
    # print(input.size())
    d_hidn = 128
    nn_emb = nn.Embedding(len(input), d_hidn)
    ####### INPUT ENBEDING ###############
    input_embs = nn_emb(input) # Input Embedding
    # print(input_embs)

    n_seq = 64
    ####### POSTION ENCODING #############
    pos_encoding = get_sinusoid_encoding_table(n_seq, d_hidn) # shape : (64, 128)

    pos_encoding = torch.FloatTensor(pos_encoding)
    nn_pos = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

    positions = torch.arange(input.size(1), device=input.device, dtype=input.dtype).expand(input.size(0),
                                                                                              input.size(
                                                                                                  1)).contiguous() + 1
    pos_mask = input.eq(0)

    positions.masked_fill_(pos_mask, 0)
    pos_embs = nn_pos(positions)  # position embedding

    # print(input)
    # print(positions)
    # print(pos_embs.size())

    input_sums = input_embs + pos_embs

    Q = input_sums
    K = input_sums
    V = input_sums # size : [7096, 53, 128]
    print("V" , V.size())
    attn_mask = input.eq(0).unsqueeze(1).expand(Q.size(0), Q.size(1), K.size(1))
    print(attn_mask.size())
    print(attn_mask[0])

    scores = torch.matmul(Q, K.transpose(-1,-2)) # size : [7096, 53, 53]
    print(scores.size())
    print(scores[0]) # size : [53, 53]

    d_head = 64
    scores = scores.mul_(1/d_head**0.5)
    print(scores.size()) # size : [7096, 53, 53]
    print(scores[0])

    scores.masked_fill_(attn_mask, -1e9)
    print(scores.size()) # size : [7096, 53, 53]
    print(scores[0])

    attn_prob = nn.Softmax(dim=-1)(scores)
    print(attn_prob.size())  # size : [7096, 53, 53]
    print(attn_prob[0])

    context = torch.matmul(attn_prob, V)
    print(context.size()) # size : [7096, 53, 128]

    attn_pad_mask = input.eq(0).unsqueeze(1).expand(Q.size(0), Q.size(1), K.size(1))
    print(attn_pad_mask)
    attn_dec_mask = helpers.get_attn_decoder_mask(input)
    print(attn_dec_mask[1])
    attn_mask = torch.gt((attn_pad_mask + attn_dec_mask), 0)
    print(attn_mask[1])

    batch_size = Q.size(0)
    n_head = 2

    attention = helpers.MultiHeadAttention(d_hidn, n_head, d_head)
    output, attn_prob = attention(Q, K , V, attn_mask)
    print(output.size(), attn_prob.size()) # size [7096, 53, 128] , [7096, 2, 53, 53]

    conv1 = nn.Conv1d(in_channels=d_hidn, out_channels=d_hidn * 4, kernel_size=1)
    ff_1 = conv1(output.transpose(1,2))
    print(ff_1.size()) # size [7096, 512, 53]

    activate = F.gelu
    ff_2 = activate(ff_1)
    conv2 = nn.Conv1d(in_channels=d_hidn * 4, out_channels=d_hidn, kernel_size=1)

    ff_3 = conv2(ff_2).transpose(1,2)
    print(ff_3.size()) # size [7096, 53, 128]

