# Transformer Chatbot Experiment

A conversational AI project exploring natural language models to create engaging, context-aware chatbots. This project evolved through three experiments, each building on the previous to enhance conversational quality and showcase AI development skills.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Summary](#experiment-summary)
  - [Experiment 1: Basic Chatbot](#experiment-1-basic-chatbot-with-dialo-gpt-small)
  - [Experiment 2: Conversational Chatbot](#experiment-2-conversational-chatbot-with-blenderbot-400m)
  - [Experiment 3: Dual Bot Conversation](#experiment-3-dual-blenderbot-400m-conversation)
- [Key Lessons Learned](#key-lessons-learned)
- [Technical Details](#technical-details)
- [Future Experiments](#future-experiments)
- [License](#license)

## Features

- Context-aware conversations with chat history management
- Input validation and cleaning
- Configurable response generation with sampling techniques
- Resettable conversation history
- Optimized for consumer hardware (e.g., 16 GB RAM)
- Extensive parameter tuning for response quality
- Debug logging for token counts and conversation analysis

## Requirements

- Python 3.8+
- transformers
- torch (PyTorch)
- safetensors for efficient model loading
- numpy<2.0 and urllib3<2.0 for compatibility

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv transformer
   source transformer/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install torch==2.2.2 safetensors transformers "numpy<2.0" "urllib3<2.0"
   ```

## Usage

### 1. Basic Chatbot (chatbot.py)
Run the DialoGPT-small chatbot:
```bash
python chatbot.py
```
- Select option 1 for configuration tests or 2 for interactive mode
- Commands:
  - Type `reset` to clear history
  - Type `quit` to exit

### 2. Conversational Chatbot (blenderbot.py)
Run the BlenderBot-400M chatbot:
```bash
python blenderbot.py
```
- Commands:
  - Type `reset` to clear history
  - Type `quit` to exit

### 3. Dual Bot Conversation (blenderbot_chat.py)
Run two BlenderBot-400M bots conversing:
```bash
python blenderbot_chat.py
```
- Enter number of turns (1-10, default 5)
- Optionally provide an initial message (default: "Hey, what's your favorite hobby?")
- Conversation is saved to `conversation_log.txt`

## Experiment Summary

### Experiment 1: Basic Chatbot with DialoGPT-small (chatbot.py)

**Objective:** Build a lightweight chatbot for general-purpose conversations using Microsoft's DialoGPT-small (137M parameters).

**Implementation:**
- Used beam search (2-4 beams) with configs like:
  - Balanced Creative: `top_k=40, top_p=0.92, temperature=0.8, max_new_tokens=80`
  - Precise & Focused: `top_k=10, top_p=0.85, temperature=0.5, max_new_tokens=60`
- Tested with 12 happy-case inputs (e.g., "I'm crazy about movies, you?", "What's your favorite animal?")
- Added input cleaning and context management with `max_history_tokens=512`

**Findings:**
- Initial outputs were generic and meme-heavy (e.g., "I'm not your guy, friend" for "Hi, I'm Alex, what's up?")
- Tuning to sampling-based configs (`top_k=10, top_p=0.99, temperature=0.4, max_new_tokens=150`) improved length but not coherence, producing gibberish (e.g., "lolololooolooolloooowwwwwwwnnnooooooooohhhhhhhhxxx")

**Lesson:** DialoGPT-small's Reddit-based training and small size limit its ability to handle structured, engaging conversations for hobbies and Q&A.

### Experiment 2: Conversational Chatbot with BlenderBot-400M (blenderbot.py)

**Objective:** Develop an engaging chatbot optimized for happy-case inputs using Facebook's BlenderBot-400M (400M parameters).

**Implementation:**
- Switched to BlenderBot-400M for better conversational flow
- Tested multiple configs, starting with high `temperature=1.3` and `top_p=0.97`
- Refined to Very Engaging Sampling:
  ```python
  {
      'top_k': 10,
      'top_p': 0.99,
      'temperature': 0.4,
      'max_new_tokens': 200
  }
  ```
- Used the same 12 happy-case inputs
- Implemented raw input (no system prompts)
- Set `max_history_tokens=512` for stability on 16 GB RAM

**Findings:**
- Early runs had truncation issues (e.g., "fi book?" for "I love sci-fi books!")
- Final config produced coherent, engaging responses (e.g., "I love hiking as well, especially in the mountains")
- Struggled with personalization (e.g., ignoring names like "Alex")

**Lessons:**
1. Proper tokenization (`skip_special_tokens=True`, `clean_up_tokenization_spaces=True`) eliminates stray characters
2. Low temperature and top_k balance coherence and engagement
3. Raw input aligns with BlenderBot's training, avoiding prompt-related confusion

### Experiment 3: Dual BlenderBot-400M Conversation (blenderbot_chat.py)

**Objective:** Simulate dynamic conversations between two BlenderBot-400M bots to showcase context management.

**Implementation:**
- Used Very Engaging Sampling:
  ```python
  {
      'top_k': 20,
      'top_p': 0.99,
      'temperature': 0.6,
      'max_new_tokens': 200,
      'repetition_penalty': 1.1
  }
  ```
- Started with specific prompts and injected new topics every 3 turns
- Each bot maintained separate chat histories with `max_history_tokens=512`
- Conversations logged to `conversation_log.txt`
- Tested for 1-10 turns (default 5)

**Findings:**
- Generic starters led to repetitive loops on beach/cruise topics
- Topic injection and `repetition_penalty=1.1` improved variety:
  ```
  Bot 1: "Movies are awesome! I'm into hiking. What's your favorite movie?"
  Bot 2: "My favorite movie is The Godfather Part II. You?"
  ```
- Still showed minor repetition and topic drift

**Lessons:**
1. Bot-to-bot conversations require topic injection for diversity
2. BlenderBot-400M excels in human-bot interactions
3. Careful tuning needed for autonomous dialogues
4. Logging and token count monitoring are essential

## Key Lessons Learned

### Model Selection
- **DialoGPT-small**
  - Lightweight but limited by meme-heavy outputs
  - Struggles with structured inputs
- **BlenderBot-400M**
  - Superior engagement for hobbies and Q&A
  - Better conversational flow due to larger size and training

### Parameter Tuning
- **Temperature**
  - High (1.3+): Noisy outputs
  - Low (0.4-0.6): More coherent responses
- **Top-k/Top-p**
  - `top_k=10-20` provides good balance
  - `top_p=0.99` allows for diverse but focused responses
- **Other Parameters**
  - `max_new_tokens=150-200` prevents truncation
  - `repetition_penalty=1.1` reduces redundancy

### Context Management
- `max_history_tokens=512` balances memory and context
- Raw input works better than system prompts
- Separate histories for multi-bot conversations

### Conversation Dynamics
- Human-bot interactions more engaging than bot-to-bot
- Topic injection prevents conversation loops
- Personalization remains challenging

## Technical Details

### Models
| Model | Parameters | Use Case |
|-------|------------|----------|
| DialoGPT-small | 137M | Basic chatbot functionality |
| BlenderBot-400M | 400M | Advanced conversations |

### Generation Parameters
```python
{
    'top_k': 10-20,
    'top_p': 0.99,
    'temperature': 0.4-0.6,
    'max_new_tokens': 200,
    'repetition_penalty': 1.1
}
```

### Performance
- Context window: 512 tokens
- Memory usage: Optimized for 16 GB RAM
- Model loading: Uses safetensors for efficiency
- Debugging: Comprehensive token counting and logging

## Future Experiments

### Model Improvements
- Experiment with DialoGPT-medium
- Fine-tune BlenderBot-400M for better personalization
- Implement LoRA or QLoRA for efficient fine-tuning

### Feature Additions
- Add profanity filtering
- Implement nonsense detection
- Develop topic modeling for better conversation flow
- Create relevance scoring for responses

### Testing
- Expand test cases with diverse inputs
- Benchmark performance across hardware
- A/B test different parameter sets

## Notes

This project demonstrates the evolution from basic to advanced conversational AI, highlighting the trade-offs between model size, response quality, and computational requirements. BlenderBot-400M shows impressive capabilities but still faces challenges in maintaining context and personalization in extended conversations.

## License
This project is for educational and personal use only. All models are subject to their respective licenses from Microsoft and Facebook AI.