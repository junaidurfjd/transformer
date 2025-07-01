from base_utils import load_model, get_blenderbot_response
import torch
import random
import argparse
from typing import Dict, List, Optional

# Topic-based conversation starters
TOPIC_STARTERS = {
    "science": [
        "What do you think about the latest advancements in quantum computing?",
        "How do you see artificial intelligence shaping our future?",
        "What's your take on the possibility of life on other planets?",
        "How do you think climate change will affect technology development?",
        "What scientific discovery from the past decade excites you the most?"
    ],
    "religion": [
        "How do different religions approach the concept of the afterlife?",
        "What role does faith play in modern society?",
        "How do you think religion influences moral decision-making?",
        "What are your thoughts on the relationship between science and religion?",
        "How has religion evolved in the digital age?"
    ],
    "philosophy": [
        "What does it mean to live a good life?",
        "Do you believe in free will? Why or why not?",
        "What's your perspective on the nature of consciousness?",
        "How do you define happiness?",
        "What's the most compelling philosophical argument you've encountered?"
    ],
    "technology": [
        "How will artificial intelligence change the job market in the next decade?",
        "What emerging technology are you most excited about?",
        "How do you think social media is affecting human relationships?",
        "What's your take on the ethics of genetic engineering?",
        "How will quantum computing revolutionize various industries?"
    ],
    "random": [
        "If you could have dinner with any historical figure, who would it be and why?",
        "What's the most interesting place you've ever visited?",
        "If you could instantly master any skill, what would it be?",
        "What's a book that changed your perspective on something?",
        "If you could time travel, would you go to the past or the future?"
    ]
}

def get_topic_starter(topic: str) -> str:
    """Get a random starter message for the given topic."""
    if topic not in TOPIC_STARTERS:
        topic = "random"
    return random.choice(TOPIC_STARTERS[topic])

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
    
    # First agent starts with the initial message
    current_speaker = 0
    agent = agents[current_speaker]
    message = initial_message
    agent['history'] = None  # Start with clean history
    
    # Print the starter message
    print(f"{agent['name']}:")
    print(f"  {message}\n")
    
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a conversation between two BlenderBot agents.')
    parser.add_argument('--turns', type=int, default=8,
                        help='Number of conversation turns (default: 8)')
    parser.add_argument('--topic', type=str, default='random',
                        choices=list(TOPIC_STARTERS.keys()) + ['all'],
                        help='Topic for conversation starters (default: random)')
    return parser.parse_args()

def main():
    print("BlenderBot Conversation Experiment")
    print("===============================")
    print("Two AI agents will have a conversation with each other.\n")
    
    # Parse command line arguments
    args = parse_arguments()
    num_turns = args.turns
    topic = args.topic
    
    # If 'all' topics is selected, pick a random topic
    if topic == 'all':
        topic = random.choice(list(TOPIC_STARTERS.keys()))
    
    print(f"Topic: {topic.capitalize()}")
    print(f"Number of turns: {num_turns}\n")
    
    # Initialize agents
    agents = setup_agents()
    
    # Get a topic-based starter message
    initial_message = get_topic_starter(topic)
    print(f"Starter: {initial_message}\n" + "-" * 50 + "\n")
    
    # Run the conversation with the topic-based starter
    chat_loop(agents, num_turns, initial_message)
    
    print("\nConversation complete!")

if __name__ == "__main__":
    main()
