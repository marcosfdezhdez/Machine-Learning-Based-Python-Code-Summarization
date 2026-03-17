#CELL 2
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # Embedding Layer: maps code tokens to dense vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # GRU Layer: we add 'batch_first=True' to match our DataLoader output
        # This ensures the input shape is [batch_size, seq_len, emb_dim]
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        # Dropout: regularization technique to improve generalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        # outputs:all hidden states
        # hidden:final state (context vector)
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden
    

#CELL 3
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # self.attn: combines decoder hidden state and encoder outputs
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        # self.v: final importance score per token
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # Repeat decoder hidden state src_len times to compare it with every code token
        # hidden: [batch_size, src_len, hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate relevance between hidden state and each code token
        # energy: [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention weights
        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        # Normalize with softmax (probabilities sum to 1)
        return F.softmax(attention, dim=1)


#CELL 4
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Decoder GRU: receives (embedding + context_vector)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True)

        # Final linear layer: predicts next token
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
  
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        # Calculate Attention Weights [batch_size, src_len]
        # We use hidden[-1] to get the last layer's hidden state
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1) 

        # Context Vector Calculation
        weighted = torch.bmm(a, encoder_outputs)

        # Combine current word embedding with attention context
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # GRU processes the combined input
        output, hidden = self.rnn(rnn_input, hidden)

        # Final Prediction (Concatenate output, context, and embedding)
        # Squeezing the sequence dimension for the linear layer
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))

        return prediction, hidden
    

#CELL 5
import random
import torch

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):


        batch_size = src.shape[0] #batch first =true
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        #Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encoder processes the code snippet
        encoder_outputs, hidden = self.encoder(src)

        # 3. First input is the <sos> token for the whole batch
        input = trg[:, 0]
        #Loop to generate the summary word by word
        for t in range(1, trg_len):
            # Decoder prediction using attention context
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # Store prediction at time step t
            outputs[:, t, :] = output

            # Teacher forcing logic
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            # Input for the next step
            input = trg[:, t] if teacher_force else top1

        return outputs