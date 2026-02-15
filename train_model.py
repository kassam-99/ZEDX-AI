import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import pandas as pd
import os










# ==========================================
# 1. CONFIGURATION (Tweaked for RTX 4060)
# ==========================================
BATCH_SIZE = 16          # Lower if you get "Out of Memory"
EPOCHS = 10              # Number of times to look at the dataset
LEARNING_RATE = 0.0001
MAX_LEN = 128            # Max length of instruction/code
EMBED_DIM = 256          # Vector size (Keep small for "scratch" learning)
HEADS = 4                # Attention heads
LAYERS = 4               # Encoder/Decoder layers
VOCAB_SIZE = 10000       # Size of your custom dictionary
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Training on: {DEVICE}")








# ==========================================
# 2. TRAIN TOKENIZER (The "Vocabulary")
# ==========================================
def train_tokenizer(csv_path):
    print("⏳ Training BPE Tokenizer...")
    # Read all text to train the tokenizer
    df = pd.read_csv(csv_path)
    # Combine instruction and output for the tokenizer to learn all symbols
    all_text = df['instruction'].tolist() + df['output'].dropna().tolist()
    
    # Save temporarily to train tokenizer
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        for line in all_text:
            f.write(str(line) + "\n")

    # Initialize and train
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["temp_corpus.txt"], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    
    # Save for later
    if not os.path.exists("alpaca_tokenizer"):
        os.makedirs("alpaca_tokenizer")
    tokenizer.save_model("alpaca_tokenizer")
    
    # Post-processor for adding special tokens automatically
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=MAX_LEN)
    tokenizer.enable_padding(length=MAX_LEN, pad_id=tokenizer.token_to_id("<pad>"))
    
    print("✅ Tokenizer trained and saved!")
    return tokenizer












# ==========================================
# 3. PREPARE DATASET (The "Loader")
# ==========================================
class CodeDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine Instruction + Input (if exists)
        input_text = row['instruction']
        if pd.notna(row['input']):
            input_text += " \nInput: " + str(row['input'])
            
        target_text = str(row['output'])

        # Encode (Turn text into numbers)
        source = self.tokenizer.encode(input_text)
        target = self.tokenizer.encode(target_text)
        
        return {
            "src": torch.tensor(source.ids, dtype=torch.long),
            "tgt": torch.tensor(target.ids, dtype=torch.long)
        }












# ==========================================
# 4. DEFINE MODEL (Transformer from Scratch)
# ==========================================
class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super(Seq2SeqTransformer, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        
        # Position Encoder (Simple learnable)
        self.pos_encoder = nn.Embedding(MAX_LEN, EMBED_DIM)
        
        # The Main Transformer Block (PyTorch Built-in)
        self.transformer = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            num_encoder_layers=LAYERS,
            num_decoder_layers=LAYERS,
            batch_first=True
        )
        
        # Output Layer (Predicts next word)
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        
    def forward(self, src, tgt):
            # Create masks (so decoder doesn't cheat and see future words)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
            
            # Add position info (FIXED: separate positions for src and tgt)
            src_positions = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(DEVICE)
            tgt_positions = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).to(DEVICE)
            
            src = self.embedding(src) + self.pos_encoder(src_positions)
            tgt = self.embedding(tgt) + self.pos_encoder(tgt_positions)
            
            # Pass through Transformer
            out = self.transformer(src, tgt, tgt_mask=tgt_mask)
            return self.fc_out(out)
    















# ==========================================
# 5. TRAINING LOOP (The "Brain" Learning)
# ==========================================
def train():
    # Load Data & Tokenizer
    tokenizer = train_tokenizer("train.csv")
    dataset = CodeDataset("train.csv", tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = Seq2SeqTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    
    print("🔥 Starting Training...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            src = batch['src'].to(DEVICE)
            tgt = batch['tgt'].to(DEVICE)
            
            # Target Input (current word) vs Target Output (next word)
            tgt_input = tgt[:, :-1] 
            tgt_output = tgt[:, 1:] 
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(src, tgt_input)
            
            # Calculate Loss (Flatten output to match CrossEntropy expectations)
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        
        print(f"✅ Epoch {epoch+1} Complete! Average Loss: {total_loss / len(dataloader):.4f}")
        
    # Save the model
    torch.save(model.state_dict(), "alpaca_code_model.pth")
    print("🎉 Model Saved as 'alpaca_code_model.pth'")






if __name__ == "__main__":
    train()