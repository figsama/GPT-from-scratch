import math
import torch
import torch.nn as nn


class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network (no nn.RNN, manual implementation)
    3) Output linear layer
    """

    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        self.hidden_size = hidden_size
        self.input_size = dim
        self.num_layers = num_layers
        self.W_ih = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        hidden_size, self.input_size if l == 0 else hidden_size
                    )
                )
                for l in range(num_layers)
            ]
        )
        self.W_hh = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for l in range(num_layers)
            ]
        )
        self.b_ih = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.b_hh = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for l in range(num_layers)]
        )
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        for l in range(self.num_layers):
            nn.init.uniform_(self.W_ih[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.W_hh[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_ih[l], -init_uniform, init_uniform)
            nn.init.uniform_(self.b_hh[l], -init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        embeddings = self.drop(self.encoder(input))
        seq_len, batch_size, _ = embeddings.size()
        if hidden is None:
            hidden = [
                embeddings.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            hidden = [h for h in hidden]
        outputs = []
        for t in range(seq_len):
            x = embeddings[t]
            h_t = []
            for l in range(self.num_layers):
                h_prev = hidden[l]
                W_ih = self.W_ih[l]
                W_hh = self.W_hh[l]
                b_ih = self.b_ih[l]
                b_hh = self.b_hh[l]
                h = torch.tanh(
                    torch.matmul(x, W_ih.t())
                    + b_ih
                    + torch.matmul(h_prev, W_hh.t())
                    + b_hh
                )
                x = h
                h_t.append(h)
            hidden = h_t
            outputs.append(h_t[-1].unsqueeze(0))
        output = torch.cat(outputs, dim=0)
        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, dim=256, nhead=8, num_layers=4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you Transformer model here. You can add additional parameters to the function.
        self.dim = dim
        self.pos_encoder = PositionalEncoding(dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        # print(input.device)
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(0)
        src_mask = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1).to(
            input.device
        )
        src = embeddings * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network
    3) Output linear layer
    """

    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        # Construct your LSTM model here.
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        # TODO: use your defined LSTM network
        output, hidden = self.lstm(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden
