from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# Load BlenderBot-400M model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)

# Happy-case inputs
inputs = [
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
        "name": "Very Engaging Sampling",
        "params": {
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.99,
            "temperature": 0.4,
            "max_new_tokens": 200
        }
    }
]

# Function to generate response
def get_response(user_input, config_params, chat_history_ids=None, max_history_tokens=512):
    input_ids = tokenizer([user_input], return_tensors="pt", padding=True, truncation=True)

    if chat_history_ids is not None:
        history_length = chat_history_ids.shape[-1]
        if history_length + input_ids.input_ids.shape[-1] > max_history_tokens:
            excess_tokens = history_length + input_ids.input_ids.shape[-1] - max_history_tokens
            chat_history_ids = chat_history_ids[:, excess_tokens:]
            history_attention_mask = (chat_history_ids != tokenizer.pad_token_id).long()
            bot_input_ids = torch.cat([chat_history_ids, input_ids.input_ids], dim=-1)
            attention_mask = torch.cat([history_attention_mask, input_ids.attention_mask], dim=-1)
        else:
            bot_input_ids = torch.cat([chat_history_ids, input_ids.input_ids], dim=-1)
            history_attention_mask = (chat_history_ids != tokenizer.pad_token_id).long()
            attention_mask = torch.cat([history_attention_mask, input_ids.attention_mask], dim=-1)
    else:
        bot_input_ids = input_ids.input_ids
        attention_mask = input_ids.attention_mask

    reply_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        **config_params
    )
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return response, reply_ids

# Run experiment
print("Experiment Results:")
print("=" * 50)
for user_input in inputs:
    print(f"Input: {user_input}")
    for config in configs:
        response, _ = get_response(user_input, config["params"])
        print(f"{config['name']}: {response}")
        print("---")
    print("=" * 50)

# Interactive loop
