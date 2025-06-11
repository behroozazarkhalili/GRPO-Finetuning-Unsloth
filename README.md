# Object-Oriented GRPO Trainer for Llama 3.2 3B

This repository contains a comprehensive object-oriented implementation of the GRPO (Group Relative Policy Optimization) trainer for fine-tuning Llama 3.2 3B on mathematical reasoning tasks using the GSM8K dataset.

## 🏗️ Architecture Overview
### Core Classes

#### 1. `ModelConfig`
- **Purpose**: Centralized configuration management for model parameters
- **Key Features**:
  - Model selection and loading parameters
  - LoRA configuration (rank, target modules)
  - Memory and performance settings
  - Type hints for all parameters

#### 2. `ReasoningPromptTemplate`
- **Purpose**: Handles prompt formatting and reasoning structure
- **Key Features**:
  - Manages special reasoning tokens (`<start_working_out>`, `<SOLUTION>`, etc.)
  - Creates system prompts for mathematical reasoning
  - Provides regex patterns for response validation
  - Formats questions into chat template format

#### 3. `DatasetProcessor`
- **Purpose**: Dataset loading, processing, and preparation
- **Key Features**:
  - Loads and processes GSM8K dataset
  - Extracts numerical answers from dataset format
  - Calculates maximum prompt lengths
  - Handles dataset mapping and transformation

#### 4. `RewardFunctions`
- **Purpose**: Collection of reward functions for GRPO training
- **Key Features**:
  - **Format Matching**: Rewards correct response structure
  - **Answer Checking**: Validates numerical correctness
  - **Approximate Matching**: Handles partial format compliance
  - **Number Extraction**: Extracts and validates numerical answers
  - Comprehensive logging and debugging support

#### 5. `GRPOModelTrainer` (Main Class)
- **Purpose**: Orchestrates the entire training pipeline
- **Key Features**:
  - Model setup with LoRA configuration
  - Complete training pipeline management
  - Inference capabilities (with/without LoRA)
  - Model saving in multiple formats
  - LoRA verification and validation

## 🚀 Key Advantages of The Repository

### 1. **Modularity & Reusability**
- Each component has a single responsibility
- Easy to extend or modify individual components
- Reusable across different projects

### 2. **Type Safety**
- Comprehensive type hints throughout
- Better IDE support and error catching
- Clear parameter expectations

### 3. **Documentation**
- Detailed docstrings for all classes and methods
- Clear parameter descriptions
- Usage examples and return value specifications

### 4. **Error Handling**
- Proper exception handling
- Validation of model states
- Graceful failure modes

### 5. **Extensibility**
- Easy to add custom reward functions
- Configurable training parameters
- Support for different model architectures

### 6. **Logging & Debugging**
- Structured logging throughout
- Progress tracking
- Model verification utilities

## 📖 Usage Examples

### Basic Training
```python
from grpo_trainer_oop import ModelConfig, GRPOModelTrainer

# Create configuration
config = ModelConfig(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    lora_rank=64
)

# Initialize and train
trainer = GRPOModelTrainer(config)
trainer.setup_model()
trainer.train()
trainer.save_lora("my_trained_lora")
```

### Inference Only
```python
# Setup for inference
trainer = GRPOModelTrainer(config)
trainer.setup_model()

# Generate response
response = trainer.generate_response(
    "What is 15 + 27?",
    use_lora=True,
    lora_path="my_trained_lora"
)
```

### Custom Configuration
```python
# Custom configuration for faster training
config = ModelConfig(
    max_seq_length=1024,  # Shorter sequences
    lora_rank=32,         # Smaller rank
    gpu_memory_utilization=0.8
)
```

## 🔧 Advanced Usage

### Custom Reward Functions
You can easily extend the trainer with custom reward functions:

```python
class CustomTrainer(GRPOModelTrainer):
    def _custom_reward(self, prompts, completions, answer, **kwargs):
        # Your custom reward logic here
        return scores
    
    def train(self):
        # Override to include custom rewards
        # ... (see example_usage.py for full implementation)
```

### Different Model Formats
```python
# Save in different formats
trainer.save_model("model_16bit", save_method="merged_16bit")
trainer.save_model("model_4bit", save_method="merged_4bit")
trainer.save_model("lora_only", save_method="lora")
```

### Pushing to Hugging Face Hub
You can easily push your trained models directly to Hugging Face Hub:

```python
# Push merged model to Hub
trainer.save_model(
    save_path="local_model",
    save_method="merged_16bit",
    push_to_hub=True,
    hub_model_id="your-username/llama-3.2-3b-grpo-math",
    token="your_hf_token_here"
)

# Push LoRA adapter only
trainer.save_model(
    save_path="local_lora",
    save_method="lora",
    push_to_hub=True,
    hub_model_id="your-username/llama-3.2-3b-grpo-lora",
    token="your_hf_token_here"
)
```

## 📁 File Structure

```
├── grpo_trainer_oop.py      # Main OOP implementation
├── example_usage.py         # Usage examples and demos
├── README_OOP.md           # This documentation
├── Advanced_Llama3_2_(3B)_GRPO_LoRA.py  # Original script
└── outputs/                # Training outputs and checkpoints
```

## 🎯 Benefits of This Refactor

1. **Maintainability**: Clear separation of concerns makes the code easier to maintain
2. **Testability**: Each component can be unit tested independently
3. **Flexibility**: Easy to swap out components or add new functionality
4. **Readability**: Well-documented code with clear interfaces
5. **Scalability**: Architecture supports adding new models, datasets, or reward functions
6. **Professional**: Production-ready code structure suitable for enterprise use

## 🔍 Key Methods Reference

### ModelConfig
- `__init__()`: Initialize configuration with sensible defaults

### ReasoningPromptTemplate
- `format_prompt(question)`: Format question for model input
- `_create_system_prompt()`: Generate system prompt for reasoning

### DatasetProcessor
- `load_gsm8k_dataset()`: Load and process GSM8K dataset
- `calculate_max_prompt_length()`: Determine optimal sequence length

### RewardFunctions
- `match_format_exactly()`: Reward perfect format compliance
- `check_answer()`: Validate answer correctness
- `check_numbers()`: Extract and verify numerical answers

### GRPOModelTrainer
- `setup_model()`: Initialize model with LoRA
- `train()`: Execute complete training pipeline
- `generate_response()`: Generate model responses
- `save_lora()`: Save trained LoRA adapter
- `save_model()`: Save model in various formats

## 🤗 Hugging Face Hub Integration

The OOP trainer provides seamless integration with Hugging Face Hub for sharing your trained models with the community.

### Getting Your Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** permissions
3. Copy the token for use in your scripts

### Pushing Models to Hub

#### Method 1: Using the `save_model()` method
```python
# Train your model
trainer = GRPOModelTrainer(config)
trainer.setup_model()
trainer.train()

# Push merged 16-bit model
trainer.save_model(
    save_path="./local_model",
    save_method="merged_16bit",
    push_to_hub=True,
    hub_model_id="your-username/llama-3.2-3b-math-reasoning",
    token="hf_your_token_here"
)

# Push 4-bit quantized model
trainer.save_model(
    save_path="./local_model_4bit",
    save_method="merged_4bit", 
    push_to_hub=True,
    hub_model_id="your-username/llama-3.2-3b-math-4bit",
    token="hf_your_token_here"
)

# Push LoRA adapter only (smaller upload)
trainer.save_model(
    save_path="./local_lora",
    save_method="lora",
    push_to_hub=True,
    hub_model_id="your-username/llama-3.2-3b-math-lora",
    token="hf_your_token_here"
)
```

#### Method 2: Using Unsloth's direct Hub methods
```python
# For merged models
if False:  # Set to True when ready to upload
    trainer.model.push_to_hub_merged(
        "your-username/llama-3.2-3b-math-reasoning",
        trainer.tokenizer,
        save_method="merged_16bit",
        token="hf_your_token_here"
    )

# For LoRA adapters
if False:  # Set to True when ready to upload
    trainer.model.push_to_hub(
        "your-username/llama-3.2-3b-math-lora",
        token="hf_your_token_here"
    )
    trainer.tokenizer.push_to_hub(
        "your-username/llama-3.2-3b-math-lora",
        token="hf_your_token_here"
    )
```

### GGUF Format for llama.cpp

For maximum compatibility with local inference tools like llama.cpp, Ollama, and Jan:

```python
# Save and push GGUF format (multiple quantizations)
if False:  # Set to True when ready
    trainer.model.push_to_hub_gguf(
        "your-username/llama-3.2-3b-math-gguf",
        trainer.tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
        token="hf_your_token_here"
    )

# Single GGUF quantization
if False:  # Set to True when ready
    trainer.model.push_to_hub_gguf(
        "your-username/llama-3.2-3b-math-q4",
        trainer.tokenizer,
        quantization_method="q4_k_m",
        token="hf_your_token_here"
    )
```

### Environment Variable for Token

For security, consider using environment variables:

```python
import os

# Set your token as an environment variable
# export HF_TOKEN="hf_your_token_here"

token = os.getenv("HF_TOKEN")
trainer.save_model(
    save_path="./model",
    save_method="merged_16bit",
    push_to_hub=True,
    hub_model_id="your-username/your-model-name",
    token=token
)
```

### Model Card and Documentation

When pushing to Hub, consider adding a model card:

```python
# Create a model card (model_card.md) in your local directory before pushing
model_card_content = """
---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- unsloth
- trl
- grpo
- mathematics
- reasoning
- gsm8k
datasets:
- openai/gsm8k
language:
- en
---

# Llama 3.2 3B Mathematical Reasoning Model

This model has been fine-tuned using GRPO (Group Relative Policy Optimization) on the GSM8K dataset for mathematical reasoning tasks.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name")

# Your inference code here
```

## Training Details

- Base Model: meta-llama/Llama-3.2-3B-Instruct
- Training Method: GRPO with LoRA
- Dataset: GSM8K
- LoRA Rank: 64
- Training Steps: 500
"""

# Save model card
with open("model_card.md", "w") as f:
    f.write(model_card_content)
```

### Best Practices for Hub Uploads

1. **Test Locally First**: Always test your model locally before uploading
2. **Use Descriptive Names**: Choose clear, descriptive repository names
3. **Include Model Cards**: Document your model's capabilities and limitations
4. **Version Control**: Use git tags or branch names for different versions
5. **License Compliance**: Ensure you comply with the base model's license
6. **Size Considerations**: LoRA adapters are much smaller than full models

## 🚦 Getting Started

1. **Install Dependencies**: Ensure you have unsloth, transformers, trl, and other required packages
2. **Run Examples**: Start with `python example_usage.py inference` to test the setup
3. **Train Model**: Use `python example_usage.py quick` for a quick training run
4. **Push to Hub**: Use the Hub integration to share your trained models
5. **Customize**: Modify the configuration or extend classes for your specific needs

This object-oriented implementation provides a solid foundation for mathematical reasoning model training while maintaining the flexibility to adapt to different use cases and requirements. 