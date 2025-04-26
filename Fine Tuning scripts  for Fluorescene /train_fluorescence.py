import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ensure we're importing from tape/ as a package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tape.tokenizers import TAPETokenizer
from tape.datasets import FluorescenceDataset
from tape.models.modeling_bert import ProteinBertModel

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer & datasets
tokenizer = TAPETokenizer(vocab='iupac')

# Check for PAD token
pad_token = '[PAD]'
if pad_token not in tokenizer.vocab:
    print(f"Warning: {pad_token} not found in tokenizer vocab. Adding it manually.")
    # Get the current vocabulary size to use as the ID for the PAD token
    pad_token_id = len(tokenizer.vocab)
    tokenizer.vocab[pad_token] = pad_token_id
    if not hasattr(tokenizer, 'inv_vocab'):
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.inv_vocab[pad_token_id] = pad_token
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = pad_token_id
    print(f"PAD token '{pad_token}' manually added to vocabulary with ID: {tokenizer.pad_token_id}")
else:
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = tokenizer.vocab[pad_token]
    print(f"PAD token '{pad_token}' already in vocabulary with ID: {tokenizer.pad_token_id}")

# Initialize the datasets
train_dataset = FluorescenceDataset(
    data_path='tape/data',
    split='train',
    tokenizer=tokenizer
)

valid_dataset = FluorescenceDataset(
    data_path='tape/data',
    split='valid',
    tokenizer=tokenizer
)

# Custom collate function
def custom_collate(batch):
    input_ids = [torch.tensor(item[0]) for item in batch]  # Convert input IDs to Tensor
    attention_mask = [torch.ones_like(torch.tensor(item[0])) for item in batch] # Convert to Tensor for mask
    targets = [item[2] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    targets = torch.tensor(targets)

    return input_ids, attention_mask, targets

# Try a smaller batch size
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Load pretrained ProteinBERT
bert_model = ProteinBertModel.from_pretrained('bert-base')
bert_model = bert_model.to(device)

# Add regression head
regression_head = nn.Linear(bert_model.config.hidden_size, 1).to(device)

# Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(bert_model.parameters()) + list(regression_head.parameters()),
    lr=5e-5
)

# Training loop
for epoch in range(1, 20):
    bert_model.train()
    regression_head.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, targets = batch

        #print(f"Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_mask.shape}, Targets shape: {targets.shape}")
        #print(f"Input IDs dtype: {input_ids.dtype}, Attention Mask dtype: {attention_mask.dtype}, Targets dtype: {targets.dtype}")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device).float()  # Ensure float attention mask
        targets = targets.float().to(device)

        #print(f"Input IDs on device: {input_ids.device}, dtype: {input_ids.dtype}")
        #print(f"Attention Mask on device: {attention_mask.device}, dtype: {attention_mask.dtype}")
        #print(f"Targets on device: {targets.device}, dtype: {targets.dtype}")

        outputs = bert_model(input_ids, attention_mask)[0]
        #print(f"BERT outputs shape: {outputs.shape}, device: {outputs.device}, dtype: {outputs.dtype}")
        #break # Uncomment to check the first batch

        cls_output = outputs[:, 0, :]
        predictions = regression_head(cls_output).squeeze(-1)

        targets = targets.squeeze(-1)

        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

# Save model
os.makedirs('tape/results/fluorescence', exist_ok=True)
torch.save({
    'bert_model': bert_model.state_dict(),
    'regression_head': regression_head.state_dict()
}, 'tape/results/fluorescence/bert_fluorescence.pt')

print("Training complete and model saved.")
