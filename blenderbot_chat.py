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
        initial_message: Optional initial message to start the conversation
    """
    config = {
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.99,
        "temperature": 0.4,
        "max_new_tokens": 200
    }
    
    # Start with a greeting if no initial message is provided
    if initial_message is None:
        initial_message = "Hello! How are you today?"
    
    print(f"\nStarting conversation for {num_turns} turns...\n")
    print(f"Initial message: {initial_message}\n")
    
    # First agent starts the conversation
    current_speaker = 0
    message = initial_message
    
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
    
    # Get user input
    try:
        num_turns = int(input("Enter number of conversation turns (default: 5): ") or 5)
        initial_message = input("Enter initial message (press Enter for default): ") or None
    except ValueError:
        print("Invalid input. Using default values.")
        num_turns = 5
        initial_message = None
    
    # Initialize agents
    agents = setup_agents()
    
    # Run the conversation
    chat_loop(agents, num_turns, initial_message)
    
    print("\nConversation complete!")

if __name__ == "__main__":
    main()
