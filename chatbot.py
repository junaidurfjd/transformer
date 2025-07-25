from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load pre-trained DialoGPT-small model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test inputs
test_inputs = [
    "Hi, I’m Alex, what’s up?",
    "I’m crazy about movies, you?",
    "What’s the best thing to do on a Friday night?",
    "I love sci-fi books!",
    "What’s your favorite animal?",
    "I’m chilling today, you?",
    "I’m into hiking, what about you?",
    "What’s your favorite drink?",
    "I’m pumped for the weekend!",
    "I love playing guitar!",
    "What’s the coolest place you’ve ever been?",
    "Hey there, I’m Sam, how’s it going?"
]

# Parameter configurations
configs = [
    {
        "name": "Balanced Sampling",
        "params": {
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.7,
            "temperature": 0.65,
            "max_new_tokens": 150,
            "repetition_penalty": 1.1
        }
    },
    {
        "name": "Very Engaging Sampling",
        "params": {
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.99,
            "temperature": 0.4,
            "max_new_tokens": 150,
            "repetition_penalty": 1.1
        }
    }
]

def clean_input(user_input):
    # Clean input: remove excessive whitespace, limit length
    user_input = re.sub(r'\s+', ' ', user_input.strip())
    if len(user_input) > 200:
        user_input = user_input[:200]
    return user_input

def get_response(user_input, config_params, chat_history_ids=None, max_history_tokens=512):
    try:
        # Clean input
        user_input = clean_input(user_input)
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        attention_mask = (new_input_ids != tokenizer.eos_token_id).long()
        
        # Handle chat history
        if chat_history_ids is not None:
            history_length = chat_history_ids.shape[-1]
            if history_length + new_input_ids.shape[-1] > max_history_tokens:
                excess = history_length + new_input_ids.shape[-1] - max_history_tokens
                chat_history_ids = chat_history_ids[:, excess:]
                history_attention_mask = (chat_history_ids != tokenizer.eos_token_id).long()
                bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
                attention_mask = torch.cat([history_attention_mask, attention_mask], dim=-1)
            else:
                bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
                history_attention_mask = (chat_history_ids != tokenizer.eos_token_id).long()
                attention_mask = torch.cat([history_attention_mask, attention_mask], dim=-1)
        else:
            bot_input_ids = new_input_ids
            attention_mask = attention_mask
        
        # Generate response
        output = model.generate(
            input_ids=bot_input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **config_params
        )
        
        # Debug: Print token counts
        print(f"Input tokens: {bot_input_ids.shape[-1]}, Generated tokens: {output.shape[-1]}")
        
        # Decode full output
        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # Remove input from response
        if response.startswith(user_input):
            response = response[len(user_input):].strip()
        
        return response, output
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error processing your request.", chat_history_ids

def run_tests():
    print("\n=== Running Configuration Tests ===\n")
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing Config: {config['name']}")
        print(f"Parameters: {config['params']}")
        print("-"*50)
        
        for i, user_input in enumerate(test_inputs):
            print(f"\nInput {i+1}: {user_input}")
            response, _ = get_response(user_input, config["params"])
            print(f"Response: {response}")
            print("-"*50)

def interactive_mode():
    print("\n=== Interactive Mode ===")
    print("Available configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['name']}")
    
    try:
        choice = int(input("\nSelect a configuration (1-2): ")) - 1
        if 0 <= choice < len(configs):
            config = configs[choice]
            print(f"\nSelected: {config['name']}")
            print("Type 'quit' to exit or 'reset' to clear history\n")
            
            chat_history_ids = None
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'reset':
                    chat_history_ids = None
                    print("Conversation history reset.")
                    continue
                    
                response, chat_history_ids = get_response(
                    user_input, 
                    config["params"], 
                    chat_history_ids
                )
                print(f"Bot: {response}")
                
        else:
            print("Invalid selection. Please try again.")
    except ValueError:
        print("Please enter a valid number.")

if __name__ == "__main__":
    print("DialoGPT Chatbot Tester")
    print("1. Run configuration tests")
    print("2. Interactive chat mode")
    
    choice = input("\nSelect an option (1-2): ")
    if choice == '1':
        run_tests()
    elif choice == '2':
        interactive_mode()
    else:
        print("Invalid choice. Exiting.")