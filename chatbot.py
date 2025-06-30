from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load pre-trained DialoGPT-small model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize conversation history
chat_history_ids = None
max_history_tokens = 256  # Tighter context control

def clean_input(user_input):
    # Clean input: remove excessive whitespace, limit length, filter profanity
    user_input = re.sub(r'\s+', ' ', user_input.strip())
    if len(user_input) > 200:
        user_input = user_input[:200]
    if re.search(r'\b(fuck|shit)\b', user_input, re.IGNORECASE):
        return "Please keep the conversation friendly!"
    # Add pseudo-prompt
    return f"User: {user_input}"

def is_repetitive(response):
    # Check for repeated words (3+ times)
    words = response.split()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    # Check for concatenated words (e.g., "HiHi") or short nonsense
    if re.search(r'\w{2,}\w{2,}', response) or len(response.strip()) < 3:
        return True
    # Check for excessive punctuation or repeated phrases
    if re.search(r'[.!?]{2,}', response) or len(set(words)) < len(words) / 2:
        return True
    return False

def get_response(user_input, chat_history_ids):
    # Clean andрав

    # Clean user input
    user_input = clean_input(user_input)
    if not user_input:
        return "Sorry, I didn't understand. Could you say something else?", chat_history_ids
    
    # Encode user input with pseudo-prompt
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Create attention mask for the input
    attention_mask = (new_input_ids != tokenizer.eos_token_id).long()
    
    # Append new input to chat history (if exists)
    if chat_history_ids is not None:
        # Truncate history to fit within max tokens
        history_length = chat_history_ids.shape[-1]
        if history_length + new_input_ids.shape[-1] > max_history_tokens:
            excess_tokens = history_length + new_input_ids.shape[-1] - max_history_tokens
            chat_history_ids = chat_history_ids[:, excess_tokens:]
            history_attention_mask = (chat_history_ids != tokenizer.eos_token_id).long()
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
            attention_mask = torch.cat([history_attention_mask, attention_mask], dim=-1)
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
            history_attention_mask = (chat_history_ids != tokenizer.eos_token_id).long()
            attention_mask = torch.cat([history_attention_mask, attention_mask], dim=-1)
    else:
        bot_input_ids = new_input_ids
    
    # Generate response with beam search
    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=False,  # Use beam search
        num_beams=5,      # Explore 5 beams for better coherence
        max_new_tokens=50  # Limit new tokens
    )
    
    # Decode and extract only the bot's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, chat_history_ids

# Main interaction loop
print("Chatbot initialized! Type 'quit' to exit or 'reset' to clear history.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "reset":
        chat_history_ids = None
        print("Conversation history reset.")
        continue
    response, chat_history_ids = get_response(user_input, chat_history_ids)
    print(f"Bot: {response}")