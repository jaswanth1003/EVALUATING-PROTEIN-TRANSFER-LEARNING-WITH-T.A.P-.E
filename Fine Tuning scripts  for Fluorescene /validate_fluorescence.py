import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure we're importing from tape/ as a package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tape.tokenizers import TAPETokenizer
from tape.datasets import FluorescenceDataset
from tape.models.modeling_bert import ProteinBertModel

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer
tokenizer = TAPETokenizer(vocab='iupac')
pad_token = '[PAD]'
if pad_token not in tokenizer.vocab:
    print(f"Warning: {pad_token} not found in tokenizer vocab. Adding it manually.")
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
test_dataset = FluorescenceDataset(
    data_path='tape/data',
    split='test',
    tokenizer=tokenizer
)


# Custom collate function (same as in training)
def custom_collate(batch):
    input_ids = [torch.tensor(item[0]) for item in batch]
    attention_mask = [torch.ones_like(torch.tensor(item[0])) for item in batch]
    targets = [item[2] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    targets = torch.tensor(targets).float()
    return input_ids, attention_mask, targets

# Create the validation DataLoader
valid_loader = DataLoader(test_dataset, batch_size=16, collate_fn=custom_collate)

# Load the trained ProteinBERT model
bert_model = ProteinBertModel.from_pretrained('bert-base')
bert_model.to(device)

# Load the trained regression head
regression_head = torch.nn.Linear(bert_model.config.hidden_size, 1).to(device)

# Load the saved state dictionaries
checkpoint = torch.load('/Users/jaswanthgurujala/Downloads/tape/results/fluorescence/bert_fluorescence.pt', map_location=device) # Adjust path if needed
bert_model.load_state_dict(checkpoint['bert_model'])
regression_head.load_state_dict(checkpoint['regression_head'])

# Set the model to evaluation mode
bert_model.eval()
regression_head.eval()

# Loss function
loss_fn = torch.nn.MSELoss()

# Validation loop
total_valid_loss = 0
with torch.no_grad():  # Disable gradient calculations during validation
    for batch in valid_loader:
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = bert_model(input_ids, attention_mask)[0]
        cls_output = outputs[:, 0, :]
        predictions = regression_head(cls_output).squeeze(-1)

        loss = loss_fn(predictions, targets)
        total_valid_loss += loss.item()

avg_valid_loss = total_valid_loss / len(valid_loader)
print(f"Validation Loss: {avg_valid_loss:.4f}")

print("Validation complete.")
