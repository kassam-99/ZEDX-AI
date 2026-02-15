import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Choose the Model
# Qwen2.5-Coder-1.5B is incredibly smart for Python and fits easily in 8GB VRAM
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print(f"⏳ Downloading and loading {model_name}...")
print("   (This might take a few minutes the first time to download ~3GB of weights)")

# 2. Load the Tokenizer and Model
# device_map="auto" automatically puts the model on your RTX 4060
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Loads in 16-bit to save VRAM
    device_map="auto" 
)

print("✅ Model loaded successfully into VRAM!")

# 3. Chat Loop
def generate_code(prompt):
    # Format the prompt so the AI knows you are asking a question
    messages = [
        {"role": "system", "content": "You are a helpful and expert Python coding assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Convert chat to the model's specific input format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Move inputs to your GPU
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerating code...\n" + "-"*40)
    
    # Generate the response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512, # How long the response can be
            temperature=0.3,    # Lower = more logical/strict, Higher = more creative
            do_sample=True
        )
        
    # Isolate just the AI's response (ignore our prompt in the output)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    while True:
        user_instruct = input("\nAsk the AI a coding question (or type 'quit'): ")
        if user_instruct.lower() == 'quit':
            break
            
        response = generate_code(user_instruct)
        print(response)
        print("-" * 40)