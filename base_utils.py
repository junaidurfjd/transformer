from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BlenderbotTokenizer, BlenderbotForConditionalGeneration

def load_model(model_type: str = "blenderbot", model_name: str = None):
    """
    Load a pre-trained model and tokenizer based on the model type.
    
    Args:
        model_type: Type of model ("blenderbot" or "dialogpt")
        model_name: Specific model name to load (optional)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if model_type == "blenderbot":
        model_name = model_name or "facebook/blenderbot-400M-distill"
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(
            model_name, use_safetensors=True
        )
    elif model_type == "dialogpt":
        model_name = model_name or "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer

def clean_input(user_input: str) -> str:
    """Clean and preprocess user input."""
    return user_input.strip()

def get_blenderbot_response(
    user_input: str,
    model: BlenderbotForConditionalGeneration,
    tokenizer: BlenderbotTokenizer,
    config_params: Dict[str, Any],
    chat_history_ids: Optional[torch.Tensor] = None,
    max_history_tokens: int = 512
) -> Tuple[str, torch.Tensor]:
    """
    Generate a response using BlenderBot model.
    
    Args:
        user_input: The user's input text
        model: The BlenderBot model
        tokenizer: The BlenderBot tokenizer
        config_params: Dictionary of generation parameters
        chat_history_ids: Previous chat history tokens
        max_history_tokens: Maximum number of tokens to keep in history
        
    Returns:
        Tuple of (response_text, updated_chat_history)
    """
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
            attention_mask = torch.cat([(chat_history_ids != tokenizer.pad_token_id).long(), 
                                      input_ids.attention_mask], dim=-1)
    else:
        bot_input_ids = input_ids.input_ids
        attention_mask = input_ids.attention_mask
    
    # Generate response
    output = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        **config_params
    )
    
    # Get the response text
    response = tokenizer.decode(
        output[0], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    return response, output
