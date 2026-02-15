import os
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. LOAD MODEL (Offline Check Included)
# ==========================================
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
local_dir = "./qwen_local_model"

print(f"🔍 Checking if model exists in '{local_dir}'...")

if os.path.exists(local_dir) and "config.json" in os.listdir(local_dir):
    print("✅ Model found locally! Loading directly into VRAM (Offline Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, dtype=torch.float16, device_map="auto", local_files_only=True
    )
else:
    print("⏳ Model not found locally. Downloading from internet (~3GB)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )
    print(f"💾 Downloading complete! Saving raw files to '{local_dir}'...")
    os.makedirs(local_dir, exist_ok=True)
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    print("✅ Model saved successfully for future offline use!")

print("🚀 Launching GUI...")

# ==========================================
# 2. GENERATION LOGIC (Now accepts Full History)
# ==========================================
def generate_code(messages):
    # The tokenizer now processes the ENTIRE conversation history
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,  # Expanded to handle massive script generations
            temperature=0.3,    
            do_sample=True
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ==========================================
# 3. GUI CLASS
# ==========================================
class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local AI Coding Assistant (Memory Enabled)")
        self.root.geometry("900x700")
        
        # --- The AI's Memory ---
        self.system_prompt = {"role": "system", "content": "You are a senior Robotics Software Engineer and expert Python developer. You write clean, optimized code."}
        self.conversation_history = [self.system_prompt]
        
        self.last_ai_response = "" 
        
        # --- Chat History Area ---
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4")
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED) 
        
        # --- User Input Area ---
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
        
        self.user_input = tk.Text(self.input_frame, height=4, font=("Consolas", 11), bg="#2d2d2d", fg="#ffffff", insertbackground="white")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message_event)
        
        # --- Buttons Area ---
        self.button_frame = tk.Frame(self.input_frame)
        self.button_frame.pack(side=tk.RIGHT)
        
        self.send_btn = tk.Button(self.button_frame, text="Send (Enter)", width=15, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), command=self.handle_send)
        self.send_btn.pack(pady=(0, 2))
        
        self.copy_btn = tk.Button(self.button_frame, text="📋 Copy Code", width=15, bg="#FF9800", fg="white", font=("Arial", 10, "bold"), command=self.handle_copy, state=tk.DISABLED)
        self.copy_btn.pack(pady=(0, 2))
        
        self.export_btn = tk.Button(self.button_frame, text="💾 Export File", width=15, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), command=self.handle_export, state=tk.DISABLED)
        self.export_btn.pack(pady=(0, 2))
        
        self.clear_btn = tk.Button(self.button_frame, text="🗑️ Clear Memory", width=15, command=self.clear_chat)
        self.clear_btn.pack()

    # --- Actions ---
    def append_to_chat(self, text, sender, color_tag=None):
        self.chat_display.config(state=tk.NORMAL)
        
        self.chat_display.tag_config("user", foreground="#5ce1e6", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("ai", foreground="#a6e22e")
        self.chat_display.tag_config("system", foreground="#fd971f", font=("Consolas", 10, "italic"))
        self.chat_display.tag_config("error", foreground="#ff5555", font=("Consolas", 10, "bold"))
        
        if sender == "User":
            self.chat_display.insert(tk.END, f"\n🧑 You:\n{text}\n\n", "user")
        elif sender == "AI":
            self.chat_display.insert(tk.END, f"🤖 AI:\n{text}\n", "ai")
            self.chat_display.insert(tk.END, "-"*60 + "\n\n", "ai")
        elif sender == "System":
            # Apply error tag if passed, otherwise default system tag
            tag_to_use = color_tag if color_tag else "system"
            self.chat_display.insert(tk.END, f"⚙️ System: {text}\n", tag_to_use)
            self.chat_display.insert(tk.END, "-"*60 + "\n\n", tag_to_use)
            
        self.chat_display.see(tk.END) 
        self.chat_display.config(state=tk.DISABLED)

    def send_message_event(self, event):
        if not event.state & 0x1: 
            self.handle_send()
            return "break" 

    def handle_send(self):
        prompt = self.user_input.get("1.0", tk.END).strip()
        if not prompt:
            return
            
        self.append_to_chat(prompt, "User")
        self.user_input.delete("1.0", tk.END)
        
        # Add user prompt to the AI's memory
        self.conversation_history.append({"role": "user", "content": prompt})
        
        self.send_btn.config(state=tk.DISABLED, text="Thinking...")
        self.export_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self._background_generate, daemon=True).start()

    def _background_generate(self):
        try:
            # Pass the entire conversation history instead of just the prompt
            response = generate_code(self.conversation_history)
            self.root.after(0, self._generation_complete, response)
        except Exception as e:
            self.root.after(0, lambda: self.append_to_chat(f"Error generating code: {str(e)}", "System", "error"))
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL, text="Send (Enter)"))

    def _generation_complete(self, response):
        self.last_ai_response = response
        
        # Save AI's response to memory so it remembers for next time
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep memory from getting too large (limits to last 10 messages + system prompt)
        if len(self.conversation_history) > 11:
            self.conversation_history = [self.system_prompt] + self.conversation_history[-10:]
            
        self.append_to_chat(response, "AI")
        self.send_btn.config(state=tk.NORMAL, text="Send (Enter)")
        self.export_btn.config(state=tk.NORMAL) 
        self.copy_btn.config(state=tk.NORMAL)

    def handle_copy(self):
        if not self.last_ai_response:
            return
        # Extract pure Python code to copy to clipboard
        code_match = re.search(r"```python(.*?)```", self.last_ai_response, re.DOTALL)
        pure_code = code_match.group(1).strip() if code_match else self.last_ai_response.strip()
        
        self.root.clipboard_clear()
        self.root.clipboard_append(pure_code)
        self.append_to_chat("Code copied to clipboard!", "System")

    def handle_export(self):
        if not self.last_ai_response:
            return
        self.export_btn.config(state=tk.DISABLED, text="Naming File...")
        threading.Thread(target=self._background_export, daemon=True).start()
        
    def _background_export(self):
        try:
            code_match = re.search(r"```python(.*?)```", self.last_ai_response, re.DOTALL)
            pure_code = code_match.group(1).strip() if code_match else self.last_ai_response.strip()
                
            naming_prompt = f"Based on this Python code, generate a short, descriptive snake_case filename ending in .py. Reply with ONLY the filename.\n\nCode:\n{pure_code}"
            
            # Use a temporary strict message so it doesn't mess up our main conversation history
            temp_messages = [{"role": "system", "content": "You only output filenames."}, {"role": "user", "content": naming_prompt}]
            raw_name = generate_code(temp_messages)
            
            name_match = re.search(r"([a-zA-Z0-9_]+\.py)", raw_name)
            filename = name_match.group(1) if name_match else "ai_generated_script.py" 
                
            with open(filename, "w", encoding="utf-8") as f:
                f.write(pure_code)
                
            self.root.after(0, self._export_complete, filename)
            
        except Exception as e:
            self.root.after(0, lambda: self.append_to_chat(f"Export Error: {str(e)}", "System", "error"))
            self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL, text="💾 Export File"))
        
    def _export_complete(self, filename):
        self.append_to_chat(f"Code successfully exported to '{filename}'", "System")
        self.export_btn.config(state=tk.NORMAL, text="💾 Export File")

    def clear_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Reset memory back to just the system prompt
        self.conversation_history = [self.system_prompt]
        self.last_ai_response = ""
        self.export_btn.config(state=tk.DISABLED)
        self.copy_btn.config(state=tk.DISABLED)
        self.append_to_chat("Memory wiped. Starting fresh context.", "System")

# ==========================================
# 4. RUN APPLICATION
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AIChatApp(root)
    root.mainloop()