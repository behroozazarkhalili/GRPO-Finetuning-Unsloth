"""
Example usage of the Object-Oriented GRPO Trainer

This script demonstrates how to use the new OOP structure for training
a mathematical reasoning model with GRPO and LoRA.
"""

from grpo_trainer_oop import ModelConfig, GRPOModelTrainer
import logging

# Configure logging to see training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_training_example():
    """Example of a quick training run with custom configuration."""
    
    # Create custom configuration
    config = ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length=1024,  # Smaller for faster training
        lora_rank=32,         # Smaller rank for faster training
        gpu_memory_utilization=0.7
    )
    
    # Initialize trainer
    trainer = GRPOModelTrainer(config)
    
    # Setup model
    logger.info("Setting up model...")
    trainer.setup_model()
    
    # Test before training
    logger.info("Testing model before training...")
    question = "If John has 5 apples and gives away 2, how many does he have left?"
    response_before = trainer.generate_response(question)
    logger.info(f"Response before training: {response_before}")
    
    # Train the model (this will take some time)
    logger.info("Starting training...")
    trainer.train()
    
    # Save the trained LoRA
    trainer.save_lora("quick_training_lora")
    
    # Test after training
    logger.info("Testing model after training...")
    response_after = trainer.generate_response(
        question, 
        use_lora=True, 
        lora_path="quick_training_lora"
    )
    logger.info(f"Response after training: {response_after}")


def inference_only_example():
    """Example of using a pre-trained model for inference only."""
    
    # Create configuration for inference
    config = ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        lora_rank=64
    )
    
    # Initialize trainer
    trainer = GRPOModelTrainer(config)
    
    # Setup model
    trainer.setup_model()
    
    # Test various mathematical problems
    test_questions = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "What is the square root of 144?",
        "If I buy 3 items at $4.50 each, how much do I spend in total?"
    ]
    
    logger.info("Testing mathematical reasoning...")
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\nQuestion {i}: {question}")
        
        # Generate response without LoRA (base model)
        response = trainer.generate_response(question, temperature=0.3)
        logger.info(f"Base model response: {response}")
        
        # If you have a trained LoRA, you can test with it:
        # response_lora = trainer.generate_response(
        #     question, 
        #     use_lora=True, 
        #     lora_path="path_to_your_lora",
        #     temperature=0.3
        # )
        # logger.info(f"LoRA model response: {response_lora}")


def custom_reward_training_example():
    """Example showing how to extend the trainer with custom reward functions."""
    
    class CustomGRPOTrainer(GRPOModelTrainer):
        """Extended trainer with custom reward function."""
        
        def __init__(self, config):
            super().__init__(config)
            # Add custom reward function to the existing ones
            self.reward_functions.custom_reward = self._custom_reward_function
        
        def _custom_reward_function(self, prompts, completions, answer, **kwargs):
            """Custom reward function that rewards longer explanations."""
            scores = []
            for completion in completions:
                response = completion[0]["content"]
                # Reward based on response length (encourage detailed explanations)
                length_score = min(len(response) / 500, 1.0)  # Max 1 point for 500+ chars
                scores.append(length_score)
            return scores
        
        def train(self):
            """Override train method to include custom reward function."""
            logger.info("Starting GRPO training with custom rewards...")
            
            # Setup model if not already done
            if self.model is None or self.tokenizer is None:
                self.setup_model()
            
            # Prepare dataset
            dataset = self.prepare_dataset()
            
            # Calculate max prompt length
            max_prompt_length = self.dataset_processor.calculate_max_prompt_length(
                dataset, self.tokenizer
            ) + 1
            
            # Create training configuration
            training_args = self.create_training_config(max_prompt_length)
            
            # Initialize trainer with custom reward functions
            from trl import GRPOTrainer
            trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=[
                    self.reward_functions.match_format_exactly,
                    self.reward_functions.match_format_approximately,
                    self.reward_functions.check_answer,
                    self.reward_functions.check_numbers,
                    self._custom_reward_function,  # Add custom reward
                ],
                args=training_args,
                train_dataset=dataset,
            )
            
            # Start training
            logger.info("Beginning GRPO training with custom rewards...")
            trainer.train()
            logger.info("Training completed!")
    
    # Use the custom trainer
    config = ModelConfig(lora_rank=32, max_seq_length=1024)
    custom_trainer = CustomGRPOTrainer(config)
    
    # Setup and train
    custom_trainer.setup_model()
    custom_trainer.train()
    custom_trainer.save_lora("custom_reward_lora")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "quick":
            logger.info("Running quick training example...")
            quick_training_example()
        elif mode == "inference":
            logger.info("Running inference-only example...")
            inference_only_example()
        elif mode == "custom":
            logger.info("Running custom reward training example...")
            custom_reward_training_example()
        else:
            logger.error(f"Unknown mode: {mode}")
            logger.info("Available modes: quick, inference, custom")
    else:
        logger.info("Available examples:")
        logger.info("  python example_usage.py quick     - Quick training example")
        logger.info("  python example_usage.py inference - Inference-only example")
        logger.info("  python example_usage.py custom    - Custom reward training")
        
        # Run inference example by default
        logger.info("\nRunning inference example by default...")
        inference_only_example() 