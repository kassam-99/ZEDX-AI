import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
import os

# ==========================================
# 1. SETUP (Must match your training settings)
# ==========================================
MAX_LEN = 128
EMBED_DIM = 256
HEADS = 4
LAYERS = 4
VOCAB_SIZE = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Tokenizer
tokenizer = ByteLevelBPETokenizer.from_file("alpaca_tokenizer/vocab.json", "alpaca_tokenizer/merges.txt")
tokenizer.enable_padding(length=MAX_LEN, pad_id=tokenizer.token_to_id("<pad>"))

# ==========================================
# 2. REBUILD MODEL ARCHITECTURE
# ==========================================
# We have to redefine the exact same structure to load the weights into it
class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = nn.Embedding(MAX_LEN, EMBED_DIM)
        self.transformer = nn.Transformer(
            d_model=EMBED_DIM, nhead=HEADS, num_encoder_layers=LAYERS, num_decoder_layers=LAYERS, batch_first=True
        )
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        
    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        src_positions = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(DEVICE)
        tgt_positions = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).to(DEVICE)
        src = self.embedding(src) + self.pos_encoder(src_positions)
        tgt = self.embedding(tgt) + self.pos_encoder(tgt_positions)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(out)

# ==========================================
# 3. LOAD THE TRAINED WEIGHTS
# ==========================================
print("Loading Model...")
model = Seq2SeqTransformer().to(DEVICE)
model.load_state_dict(torch.load("alpaca_code_model.pth", weights_only=True))
model.eval() # Set model to "evaluation" mode (turns off learning)
print("Model Loaded successfully!")

# ==========================================
# 4. CHAT WITH YOUR AI
# ==========================================
def generate_code(instruction, input_text=""):
    # Format the prompt exactly how it was trained
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    
    # Tokenize input
    src_tokens = tokenizer.encode(prompt).ids
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(DEVICE)
    
    # Start generation with a dummy starting token (like padding)
    tgt_tokens = [tokenizer.token_to_id("<s>")] if tokenizer.token_to_id("<s>") is not None else [0]
    
    print("\nGenerating code...\n" + "-"*40)
    
    # Autoregressive generation (word by word)
    for i in range(MAX_LEN):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad(): # Don't track gradients (saves memory)
            output = model(src_tensor, tgt_tensor)
            
        # Get the ID of the word with the highest probability
        next_token_id = output[0, -1, :].argmax().item()
        
        tgt_tokens.append(next_token_id)
        
        # Stop if the model predicts the "End of Sentence" token
        if next_token_id == tokenizer.token_to_id("</s>") or next_token_id == tokenizer.token_to_id("<pad>"):
            break
            
    # Decode numbers back to text
    generated_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return generated_text

# Let's test it!
if __name__ == "__main__":
    while True:
        user_instruct = input("\nAsk the AI to write Python (or type 'quit'): ")
        if user_instruct.lower() == 'quit':
            break
            
        user_input = input("Any specific inputs? (Leave blank if none): ")
        
        response = generate_code(user_instruct, user_input)
        print(response)
        print("-" * 40)