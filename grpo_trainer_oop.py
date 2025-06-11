"""
Object-Oriented GRPO (Group Relative Policy Optimization) Trainer for Llama 3.2 3B

This module provides a comprehensive object-oriented implementation for training
language models using GRPO with LoRA (Low-Rank Adaptation) on mathematical
reasoning tasks using the GSM8K dataset.

Author: Generated from Unsloth GRPO notebook
"""

from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import re
import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for model parameters and training settings."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length: int = 2048,
        lora_rank: int = 64,
        load_in_4bit: bool = False,
        fast_inference: bool = True,
        gpu_memory_utilization: float = 0.6
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: HuggingFace model identifier
            max_seq_length: Maximum sequence length for training
            lora_rank: Rank for LoRA adaptation (higher = more parameters)
            load_in_4bit: Whether to load model in 4-bit precision
            fast_inference: Enable vLLM fast inference
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.load_in_4bit = load_in_4bit
        self.fast_inference = fast_inference
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # LoRA target modules for Llama architecture
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]


class ReasoningPromptTemplate:
    """Handles prompt formatting and reasoning structure for mathematical problems."""
    
    def __init__(self):
        """Initialize reasoning prompt template with special tokens."""
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end = "<end_working_out>"
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"
        
        self.system_prompt = self._create_system_prompt()
        self.match_format_regex = self._create_format_regex()
        self.match_numbers_regex = self._create_numbers_regex()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for mathematical reasoning."""
        return (
            f"You are given a problem.\n"
            f"Think about the problem and provide your working out.\n"
            f"Place it between {self.reasoning_start} and {self.reasoning_end}.\n"
            f"Then, provide your solution between {self.solution_start}{self.solution_end}"
        )
    
    def _create_format_regex(self) -> re.Pattern:
        """Create regex pattern to match the expected response format."""
        return re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.reasoning_start}.+?{self.reasoning_end}.*?"
            rf"{self.solution_start}(.+?){self.solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def _create_numbers_regex(self) -> re.Pattern:
        """Create regex pattern to extract numerical answers."""
        return re.compile(
            self.solution_start + r".*?([\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def format_prompt(self, question: str) -> List[Dict[str, str]]:
        """
        Format a question into the chat template format.
        
        Args:
            question: The mathematical problem to solve
            
        Returns:
            List of message dictionaries for chat template
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]


class DatasetProcessor:
    """Handles dataset loading, processing, and preparation for training."""
    
    def __init__(self, prompt_template: ReasoningPromptTemplate):
        """
        Initialize dataset processor.
        
        Args:
            prompt_template: Template for formatting prompts
        """
        self.prompt_template = prompt_template
    
    def load_gsm8k_dataset(self) -> Dataset:
        """
        Load and process the GSM8K dataset.
        
        Returns:
            Processed dataset with prompts and answers
        """
        logger.info("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        
        # Process dataset to extract answers and format prompts
        processed_dataset = dataset.map(self._process_example)
        
        logger.info(f"Loaded {len(processed_dataset)} examples from GSM8K")
        return processed_dataset
    
    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single dataset example.
        
        Args:
            example: Raw dataset example
            
        Returns:
            Processed example with prompt and extracted answer
        """
        question = example["question"]
        raw_answer = example["answer"]
        
        # Extract numerical answer from the solution
        extracted_answer = self._extract_hash_answer(raw_answer)
        
        # Format prompt using template
        prompt = self.prompt_template.format_prompt(question)
        
        return {
            "prompt": prompt,
            "answer": extracted_answer,
        }
    
    @staticmethod
    def _extract_hash_answer(text: str) -> Optional[str]:
        """
        Extract the numerical answer from GSM8K format (after ####).
        
        Args:
            text: Raw answer text from dataset
            
        Returns:
            Extracted numerical answer or None if not found
        """
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    def calculate_max_prompt_length(self, dataset: Dataset, tokenizer) -> int:
        """
        Calculate the maximum prompt length in the dataset.
        
        Args:
            dataset: Processed dataset
            tokenizer: Model tokenizer
            
        Returns:
            Maximum prompt length in tokens
        """
        logger.info("Calculating maximum prompt length...")
        
        def tokenize_prompt(example):
            tokens = tokenizer.apply_chat_template(
                example["prompt"], 
                add_generation_prompt=True, 
                tokenize=True
            )
            return {"tokens": tokens}
        
        tokenized_dataset = dataset.map(tokenize_prompt, batched=True)
        lengths = [len(tokens) for tokens in tokenized_dataset["tokens"]]
        max_length = max(lengths)
        
        logger.info(f"Maximum prompt length: {max_length} tokens")
        return max_length


class RewardFunctions:
    """Collection of reward functions for GRPO training."""
    
    def __init__(self, prompt_template: ReasoningPromptTemplate):
        """
        Initialize reward functions.
        
        Args:
            prompt_template: Template containing regex patterns
        """
        self.prompt_template = prompt_template
        self.printed_times = 0
        self.print_every_steps = 5
    
    def match_format_exactly(self, completions: List[List[Dict]], **kwargs) -> List[float]:
        """
        Reward function for exact format matching.
        
        Args:
            completions: List of model completions
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            
            # Reward 3 points for exact format match
            if self.prompt_template.match_format_regex.search(response) is not None:
                score += 3.0
            
            scores.append(score)
        return scores
    
    def match_format_approximately(self, completions: List[List[Dict]], **kwargs) -> List[float]:
        """
        Reward function for approximate format matching.
        
        Args:
            completions: List of model completions
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            
            # Count occurrences of each special token
            reasoning_start_count = response.count(self.prompt_template.reasoning_start)
            reasoning_end_count = response.count(self.prompt_template.reasoning_end)
            solution_start_count = response.count(self.prompt_template.solution_start)
            solution_end_count = response.count(self.prompt_template.solution_end)
            
            # Reward correct usage, penalize incorrect usage
            score += 0.5 if reasoning_start_count == 1 else -1.0
            score += 0.5 if reasoning_end_count == 1 else -1.0
            score += 0.5 if solution_start_count == 1 else -1.0
            score += 0.5 if solution_end_count == 1 else -1.0
            
            scores.append(score)
        return scores
    
    def check_answer(
        self, 
        prompts: List[List[Dict]], 
        completions: List[List[Dict]], 
        answer: List[str], 
        **kwargs
    ) -> List[float]:
        """
        Reward function for answer correctness.
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            answer: List of correct answers
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract answers using regex
        extracted_responses = [
            match.group(1) if (match := self.prompt_template.match_format_regex.search(r)) 
            is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0.0
            
            if guess is None:
                scores.append(0.0)
                continue
            
            # Exact match gets highest reward
            if guess == true_answer:
                score += 3.0
            # Match with whitespace differences gets lower reward
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # Reward based on numerical proximity
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 1.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 0.5
                    else:
                        score -= 1.5  # Penalize wrong answers
                except (ValueError, ZeroDivisionError):
                    score -= 1.5  # Penalize non-numeric answers
            
            scores.append(score)
        return scores
    
    def check_numbers(
        self, 
        prompts: List[List[Dict]], 
        completions: List[List[Dict]], 
        answer: List[str], 
        **kwargs
    ) -> List[float]:
        """
        Reward function for numerical answer extraction and checking.
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            answer: List of correct answers
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract numerical answers
        extracted_responses = [
            match.group(1) if (match := self.prompt_template.match_numbers_regex.search(r)) 
            is not None else None
            for r in responses
        ]
        
        # Print progress periodically
        if self.printed_times % self.print_every_steps == 0:
            logger.info(f"Question: {question}")
            logger.info(f"Expected Answer: {answer[0]}")
            logger.info(f"Model Response: {responses[0]}")
            logger.info(f"Extracted Answer: {extracted_responses[0]}")
            logger.info("*" * 50)
        
        self.printed_times += 1
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0.0)
                continue
            
            try:
                true_answer_float = float(true_answer.strip())
                # Remove commas from numbers like 123,456
                guess_float = float(guess.strip().replace(",", ""))
                
                # Exact match gets positive reward, wrong answer gets penalty
                score = 1.5 if guess_float == true_answer_float else -0.5
                scores.append(score)
            except (ValueError, AttributeError):
                scores.append(0.0)
        
        return scores


class GRPOModelTrainer:
    """Main trainer class for GRPO fine-tuning with LoRA."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the GRPO trainer.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompt_template = ReasoningPromptTemplate()
        self.dataset_processor = DatasetProcessor(self.prompt_template)
        self.reward_functions = RewardFunctions(self.prompt_template)
        
    def setup_model(self) -> Tuple[Any, Any]:
        """
        Load and configure the model with LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            fast_inference=self.config.fast_inference,
            max_lora_rank=self.config.lora_rank,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Model setup completed")
        return model, tokenizer
    
    def prepare_dataset(self) -> Dataset:
        """
        Load and prepare the training dataset.
        
        Returns:
            Processed dataset ready for training
        """
        return self.dataset_processor.load_gsm8k_dataset()
    
    def create_training_config(self, max_prompt_length: int) -> GRPOConfig:
        """
        Create GRPO training configuration.
        
        Args:
            max_prompt_length: Maximum prompt length in tokens
            
        Returns:
            GRPO training configuration
        """
        return GRPOConfig(
            learning_rate=5e-6,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=max_prompt_length,
            max_completion_length=self.config.max_seq_length - max_prompt_length,
            max_steps=500,
            save_steps=250,
            max_grad_norm=1.0,
            report_to="none",
            output_dir="outputs",
        )
    
    def train(self) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting GRPO training pipeline...")
        
        # Setup model
        if self.model is None or self.tokenizer is None:
            self.setup_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Calculate max prompt length
        max_prompt_length = self.dataset_processor.calculate_max_prompt_length(
            dataset, self.tokenizer
        ) + 1  # Add buffer
        
        # Create training configuration
        training_args = self.create_training_config(max_prompt_length)
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.reward_functions.match_format_exactly,
                self.reward_functions.match_format_approximately,
                self.reward_functions.check_answer,
                self.reward_functions.check_numbers,
            ],
            args=training_args,
            train_dataset=dataset,
        )
        
        # Start training
        logger.info("Beginning GRPO training...")
        trainer.train()
        logger.info("Training completed!")
    
    def generate_response(
        self, 
        question: str, 
        use_lora: bool = False,
        lora_path: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response to a mathematical question.
        
        Args:
            question: The mathematical problem to solve
            use_lora: Whether to use LoRA adapter
            lora_path: Path to LoRA adapter (if use_lora is True)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Format the prompt
        if use_lora:
            messages = self.prompt_template.format_prompt(question)
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text = self.tokenizer.apply_chat_template([
                {"role": "user", "content": question},
            ], tokenize=False, add_generation_prompt=True)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        # Generate response
        lora_request = None
        if use_lora and lora_path:
            lora_request = self.model.load_lora(lora_path)
        
        output = self.model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        
        return output
    
    def save_lora(self, save_path: str) -> None:
        """
        Save the trained LoRA adapter.
        
        Args:
            save_path: Path to save the LoRA adapter
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        logger.info(f"Saving LoRA adapter to: {save_path}")
        self.model.save_lora(save_path)
        
        # Verify the LoRA was actually trained
        self._verify_lora_training(save_path)
    
    def _verify_lora_training(self, lora_path: str) -> None:
        """
        Verify that the LoRA adapter contains non-zero weights.
        
        Args:
            lora_path: Path to the LoRA adapter
        """
        adapter_file = f"{lora_path}/adapter_model.safetensors"
        
        try:
            with safe_open(adapter_file, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    n_zeros = (tensor == 0).sum() / tensor.numel()
                    
                    # Ensure the tensor is not all zeros
                    if n_zeros.item() == 1.0:  # All zeros
                        logger.warning(f"LoRA tensor {key} contains all zeros!")
                    else:
                        logger.info(f"LoRA tensor {key} verified: {(1-n_zeros.item())*100:.2f}% non-zero")
            
            logger.info("LoRA verification completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to verify LoRA: {e}")
    
    def save_model(
        self, 
        save_path: str, 
        save_method: str = "merged_16bit",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        token: Optional[str] = None
    ) -> None:
        """
        Save the model in various formats.
        
        Args:
            save_path: Local path to save the model
            save_method: Save method ('merged_16bit', 'merged_4bit', 'lora')
            push_to_hub: Whether to push to Hugging Face Hub
            hub_model_id: Hub model identifier (required if push_to_hub=True)
            token: Hugging Face token (required if push_to_hub=True)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized.")
        
        logger.info(f"Saving model with method: {save_method}")
        
        if save_method in ["merged_16bit", "merged_4bit"]:
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method=save_method)
            
            if push_to_hub and hub_model_id and token:
                self.model.push_to_hub_merged(hub_model_id, self.tokenizer, save_method=save_method, token=token)
                
        elif save_method == "lora":
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            if push_to_hub and hub_model_id and token:
                self.model.push_to_hub(hub_model_id, token=token)
                self.tokenizer.push_to_hub(hub_model_id, token=token)
        
        logger.info(f"Model saved successfully to: {save_path}")


def main():
    """Main function demonstrating usage of the GRPO trainer."""
    # Create configuration
    config = ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        lora_rank=64,
        gpu_memory_utilization=0.6
    )
    
    # Initialize trainer
    trainer = GRPOModelTrainer(config)
    
    # Setup model
    trainer.setup_model()
    
    # Test inference before training
    logger.info("Testing model before training...")
    response_before = trainer.generate_response("What is the sqrt of 101?")
    logger.info(f"Response before training: {response_before}")
    
    # Train the model
    trainer.train()
    
    # Save LoRA adapter
    trainer.save_lora("grpo_saved_lora")
    
    # Test inference after training
    logger.info("Testing model after training...")
    response_after = trainer.generate_response(
        "What is the sqrt of 101?", 
        use_lora=True, 
        lora_path="grpo_saved_lora"
    )
    logger.info(f"Response after training: {response_after}")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 