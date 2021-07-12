import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Model.transformer import Transformer
from constants import config
from words import vocab_transfered, device


class Prediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output,
                                    bias = False)\

    def forward(self, enc_inputs, dec_inputs):
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
        dec_outputs, _ = torch.max(dec_outputs, dim = 1)

        logits = self.projection(dec_outputs)

        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs


def collate_fn(inputs):
    print(len(inputs))
    # labels, enc_inputs, dec_inputs = list(zip(*inputs))
    enc_inputs =

    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        enc_inputs,
        dec_inputs,
    ]
    return batch


def eval_epoch(config, model, data_loader):
    matchs = []
    model.eval()

    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]
            _, indices = logits.max(1)

            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0]

            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

if __name__ == '__main__':
    n_epoch = 10
    learning_rate = 5e-5
    batch_size = 128
    config.n_output = 2
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_text = vocab_transfered('text/phoenix2014T.train.de')
    input_text = torch.Tensor(input_text).to(device).long()
    train_loader = torch.utils.data.DataLoader(input_text, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    out_gt = vocab_transfered('text/phoenix2014T.train.gloss')
    out_gt = torch.Tensor(out_gt).to(device)

    train_loader = torch.utils.data.DataLoader(input_text, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(out_gt, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn)
    print(train_loader)
    print(test_loader)
    model = Prediction(config)
    model.to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch, best_loss, best_score = 0, 0, 0
    losses, scores = [], []
    for epoch in range(n_epoch):
        loss = train_epoch(config, epoch, model, criterion, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        losses.append(loss)
        scores.append(score)

        if best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
    print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")


