from base_utils import load_model, get_blenderbot_response
import torch

def setup_agents():
    """Initialize two BlenderBot agents with their models and tokenizers."""
    print("Loading BlenderBot models...")
    # Both agents use the same model but will maintain separate conversation histories
    model, tokenizer = load_model("blenderbot")
    return [
        {"model": model, "tokenizer": tokenizer, "history": None, "name": "BlenderBot 1"},
        {"model": model, "tokenizer": tokenizer, "history": None, "name": "BlenderBot 2"}
    ]

def chat_loop(agents, num_turns=5, initial_message=None):
    """Run a conversation between two BlenderBot agents.
    
    Args:
        agents: List of two agent dictionaries
        num_turns: Number of conversation turns (back and forth)
        initial_message: Not used, kept for backward compatibility
    """
    config = {
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.99,
        "temperature": 0.8,  # Higher temperature for more creative initial message
        "max_new_tokens": 200,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3
    }
    
    print("\nStarting conversation...")
    print("First bot is thinking of something to say...\n")
    
    # First agent generates its own initial message
    current_speaker = 0
    agent = agents[current_speaker]
    
    # Generate initial message with a prompt that encourages the bot to start a conversation
    initial_prompt = ""
    input_ids = agent['tokenizer']([initial_prompt], return_tensors="pt")
    output = agent['model'].generate(
        **input_ids,
        **config
    )
    message = agent['tokenizer'].decode(
        output[0], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    # Store the initial message in history
    agent['history'] = output
    
    for turn in range(num_turns * 2):  # Multiply by 2 for back-and-forth
        agent = agents[current_speaker]
        print(f"{agent['name']}:")
        print(f"  {message}")
        
        # Get response from current speaker
        response, agent['history'] = get_blenderbot_response(
            message,
            agent['model'],
            agent['tokenizer'],
            config,
            agent['history']
        )
        
        # Switch speakers for next turn
        current_speaker = 1 - current_speaker
        message = response
        
        print()  # Add spacing between turns

def main():
    print("BlenderBot Conversation Experiment")
    print("===============================")
    print("Two AI agents will have a conversation with each other.")
    print("The first bot will generate its own initial message.\n")
    
    # Set fixed number of turns
    num_turns = 8  # Fixed number of conversation turns
    
    # Initialize agents
    agents = setup_agents()
    
    # Run the conversation (initial_message is not used, kept for backward compatibility)
    chat_loop(agents, num_turns, None)
    
    print("\nConversation complete!")

if __name__ == "__main__":
    main()
