# Basic Chatbot

A conversational AI chatbot built using Microsoft's DialoGPT-small model from Hugging Face Transformers. This chatbot is designed to have natural conversations with users while maintaining context and providing meaningful responses.

## Features

- Context-aware conversations using chat history
- Input validation and cleaning
- Profanity filtering
- Repetition and nonsense detection
- Resettable conversation history
- Optimized for efficient memory usage

## Requirements

- Python 3.8+
- transformers library
- torch (PyTorch)

## Installation

1. Install the required dependencies:
```bash
pip install transformers torch
```

## Usage

1. Run the chatbot:
```bash
python chatbot.py
```

2. Start chatting with the bot! Here are some commands:
- Type anything to chat with the bot
- Type 'reset' to clear the conversation history
- Type 'quit' to exit the chatbot

## Technical Details

- Uses Microsoft's DialoGPT-small model for conversation
- Implements beam search with 5 beams for better response coherence
- Maintains conversation history with controlled context length
- Includes input validation and response quality checks

## Notes

- The chatbot maintains conversation history to provide context-aware responses
- Responses are generated using beam search for better quality
- The model has been optimized for efficient memory usage

## License

This project is for educational and personal use only.
