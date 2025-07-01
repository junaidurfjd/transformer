Basic Chatbot
A conversational AI chatbot project exploring natural language models for engaging, context-aware conversations. The project includes multiple components:
- A basic chatbot using Microsoft's DialoGPT-small
- An enhanced conversational chatbot using Facebook's BlenderBot-400M
- An experiment with two BlenderBot agents conversing with each other

All components are optimized through extensive experimentation for portfolio-ready performance.
Features

Context-aware conversations with chat history management
Input validation and cleaning
Configurable response generation with sampling techniques
Resettable conversation history
Optimized for efficient memory usage on consumer hardware (e.g., 16 GB RAM)
Experimentation with model parameters to improve response quality

Requirements

Python 3.8+
transformers library
torch (PyTorch)
safetensors for efficient model loading
numpy<2.0, urllib3<2.0 for compatibility

Installation

Create a virtual environment (recommended):python3 -m venv basic-chat
source basic-chat/bin/activate


Install dependencies:pip install --upgrade pip
pip install torch==2.2.2 safetensors transformers "numpy<2.0" "urllib3<2.0"



## Usage

### Basic Chatbot
```bash
python chatbot.py
```

### Conversational Chatbot
```bash
python chatbot_experiment_happy_v11.py
```

### BlenderBot Conversation Experiment
Run two BlenderBot agents talking to each other:
```bash
python blenderbot_chat.py
```
When prompted:
- Enter the number of conversation turns (default: 5)
- Optionally provide an initial message to start the conversation



Start chatting! Commands:
Type anything to interact with the bot
Type reset to clear conversation history
Type quit to exit



## Experiment Analysis

### BlenderBot-to-BlenderBot Conversations

After multiple runs of the BlenderBot conversation experiment, we've observed several interesting patterns and areas for improvement:

#### Key Observations:

1. **Conversation Starters**
   - The model tends to start with simple, open-ended questions about common topics (pets, hobbies, movies)
   - Initial messages often follow the pattern "Do you have any X? I like Y."

2. **Consistency Issues**
   - Bots sometimes contradict themselves within the same conversation
   - Example: One bot said "I have a dog" then later "I don't have any pets"

3. **Topic Drift**
   - Conversations often drift to unrelated topics
   - Example: Beach conversation suddenly switched to cruises without clear transition

4. **Memory Limitations**
   - Bots sometimes forget or contradict earlier parts of the conversation
   - Repetition of similar phrases occurs in longer conversations

#### Areas for Improvement:

1. **Context Management**
   - Implement better conversation history tracking
   - Add topic consistency checks
   - Reduce repetition through better prompt engineering

2. **Diversity**
   - Encourage more varied conversation starters
   - Implement topic steering to explore different subjects
   - Add constraints to prevent repetitive responses

3. **Engagement**
   - Improve question-asking patterns
   - Add more detailed and specific responses
   - Implement better follow-up questions

### Next Steps

1. Experiment with different temperature and top-k settings
2. Implement conversation memory tracking
3. Add topic modeling to guide conversation flow
4. Develop metrics for evaluating conversation quality

## Project Architecture

### Base Utilities (base_utils.py)
A shared module containing common functionality used across different chatbot implementations:
- Model loading and initialization
- Input cleaning and preprocessing
- Response generation logic

### Chatbot Components

1. Basic Chatbot (chatbot.py)

Built with Microsoft's DialoGPT-small model.
Uses beam search (5 beams) for coherent responses.
Implements basic context management and input validation.
Focuses on lightweight, general-purpose conversations.

2. Conversational Chatbot (chatbot_experiment_happy_v11.py)

Built with Facebook's BlenderBot-400M model for enhanced conversational engagement.
Optimized for "happy case" inputs (e.g., hobbies, casual statements, simple Q&A).
Features a single, tuned configuration: Very Engaging Sampling (top_k=10, top_p=0.99, temperature=0.4, max_new_tokens=200).
Supports context-aware conversations with max_history_tokens=512 for efficient memory usage.
Includes debug logging for token counts to ensure proper response generation.

Experimentation and Lessons Learned
The project evolved through extensive experimentation to optimize BlenderBot-400M for engaging conversations, addressing challenges like truncation and personalization. Key iterations and findings:

Initial Setup (DialoGPT-small):

Used chatbot.py with beam search for stable, generic responses.
Limited by lack of parameter tuning and weaker engagement for hobby-related inputs.


Switch to BlenderBot-400M:

Adopted BlenderBot-400M for better conversational flow, especially for hobbies (e.g., “Mine is The Godfather Part II”).
Tested multiple configs (Balanced, Engaging, Very Engaging, Balanced Plus) with varying top_k, top_p, temperature, and max_new_tokens.


Truncation Issues:

Problem: Early runs had truncated outputs (e.g., “fi book?”, “you up to?”) due to low max_new_tokens (60–120) and incorrect slicing in batch_decode.
Solution: Increased max_new_tokens to 200 and removed slicing in batch_decode (reply_ids decoded fully with skip_special_tokens=True).


Config Tuning:

Problem: High temperature (1.3) and top_p (0.97) caused noisy, fragmented responses.
Solution: Settled on top_k=10, top_p=0.99, temperature=0.4 for focused, chatty outputs (e.g., “I love hiking as well, especially in the mountains”).


Prompting Missteps:

Problem: Adding prompts (e.g., “You are a friendly assistant”) worsened personalization (e.g., ignoring “Alex”/“Sam”).
Solution: Used raw input to align with BlenderBot’s training, improving responses.


Context Management:

Problem: High max_history_tokens (1024) risked memory issues on 16 GB RAM.
Solution: Set to 512 for stability, sufficient for single-turn experiments and interactive mode.


Key Findings:

BlenderBot-400M excels at hobby chats (e.g., movies, hiking) and simple Q&A (e.g., “dogs are my favorite”).
Struggles with personalization (e.g., using names like “Alex”).
Low temperature and top_k improve coherence without sacrificing engagement.
Proper tokenization (skip_special_tokens=True, clean_up_tokenization_spaces=True) is critical to avoid stray characters.



Technical Details

Models: DialoGPT-small (basic chatbot) and BlenderBot-400M (conversational chatbot).
Generation: Very Engaging Sampling uses top-k sampling (top_k=10), nucleus sampling (top_p=0.99), and low temperature (0.4) for coherent, engaging responses.
Context: Maintains history with max_history_tokens=512, trimming excess tokens dynamically.
Optimization: Uses safetensors for efficient model loading and runs on consumer hardware (16 GB RAM).
Debugging: Includes token count logging to verify input/output lengths.

Future Experiments

Add more configs (e.g., Balanced Sampling) for comparison.
Explore DialoGPT-medium or fine-tuning BlenderBot for better personalization.
Test additional inputs (e.g., “I love coding!”) to expand happy cases.
Implement profanity filtering or nonsense detection from the basic chatbot.

Notes

The conversational chatbot shines with hobby-related and casual inputs but may struggle with names or complex open-ended questions.
Optimized for portfolio demonstration, showcasing parameter tuning and debugging skills.
Responses are generated without external prompts for authenticity.

License
This project is for educational and personal use only.