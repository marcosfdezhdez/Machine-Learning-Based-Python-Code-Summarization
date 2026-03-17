#CELL 6
INPUT_DIM = len(vocab_code)      # Vocabulary sizes (encoder input / decoder output)
OUTPUT_DIM = len(vocab_summary)  # Now it will be thousands of tokens
ENC_EMB_DIM = 256                # Embedding and hidden dimensions
DEC_EMB_DIM = 256                # Increased for better learning
HID_DIM = 512                    # More "memory" for real code
N_LAYERS = 1
ENC_DROPOUT = 0.5                # Dropout applied explicitly to embeddings (single-layer GRU)
DEC_DROPOUT = 0.5

# Constructions of modules
attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

# Weight initialization
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.apply(init_weights)

# Adam optimizer with a standard learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Ignore the padding token when calculating loss
TRG_PAD_IDX = vocab_summary['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

print(f"Model initialized with {INPUT_DIM} code tokens and {OUTPUT_DIM} summary tokens.")


#CELL 7
import torch
import os

MODEL_PATH = "/content/models/model_final_project.pth"

assert os.path.exists(MODEL_PATH), "Model not found!"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model found!")



#CELL 8
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Function to process and pad the sequences in each batch
def collate_fn(batch):
    src_list, trg_list = [], []
    for (src, trg) in batch:
        src_indices = [vocab_code['<sos>']] + [vocab_code[token] for token in tokenizer_code(src)] + [vocab_code['<eos>']]
        trg_indices = [vocab_summary['<sos>']] + [vocab_summary[token] for token in tokenizer_summary(trg)] + [vocab_summary['<eos>']]

        src_list.append(torch.tensor(src_indices))
        trg_list.append(torch.tensor(trg_indices))

    return pad_sequence(src_list, batch_first=True, padding_value=vocab_code['<pad>']).to(device), \
           pad_sequence(trg_list, batch_first=True, padding_value=vocab_summary['<pad>']).to(device)


def evaluate_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for src, trg in dataloader:
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            # We adjust for batch_first=True [batch, seq, dim]
            output = output[:, 1:].reshape(-1, output_dim)
            trg2 = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg2)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)

# Create the DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_data,  batch_size=16, shuffle=False, collate_fn=collate_fn)

  # Training execution
model.train()
N_EPOCHS = 10
CLIP = 1
print(f"Starting training on {len(train_data)} examples...")

for epoch in range(N_EPOCHS):
      epoch_loss = 0.0
      for src, trg in train_loader:
          optimizer.zero_grad()
          output = model(src, trg, teacher_forcing_ratio=0.5)

          output_dim = output.shape[-1]
          # We adjust for batch_first=True
          output = output[:, 1:].reshape(-1, output_dim)
          trg_flat = trg[:, 1:].reshape(-1)

          loss = criterion(output, trg_flat)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
          optimizer.step()
          epoch_loss += loss.item()

      train_loss = epoch_loss / len(train_loader)
      val_loss = evaluate_loss(model, val_loader, criterion)
      print(f"Epoch: {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

  # Save
torch.save(model.state_dict(), "model_final_project.pth")