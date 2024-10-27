import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class FineTuner:
    def __init__(self, model, tokenizer, device=None):
        # Set device
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') if device is None else device

        # Assign the model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(self.device)

        # Resize model embeddings if additional special tokens are provided
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['{', '}']})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def fine_tune(self, training_data, epochs=5, batch_size=16, learning_rate=3e-4, weight_decay=0.01):
        train_df, eval_df = train_test_split(training_data, test_size=0.2, random_state=42)

        # Convert to `Dataset`
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)

        # Tokenize both train and evaluation datasets
        def tokenize_function(examples):
            model_inputs = self.tokenizer(examples["prompt"], max_length=64, padding="max_length", truncation=True)
            labels = self.tokenizer(examples["response"], max_length=64, padding="max_length", truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./param_results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            eval_accumulation_steps=10,
            logging_dir="./param_logs",
            logging_strategy="epoch"
        )

        # Use DataCollatorForSeq2Seq for sequence-to-sequence tasks
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding=True)

        # Use the default Trainer
        param_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # Train the model and measure training time and peak memory usage
        print("--- Starting Training ---\n")
        param_trainer.train()
        
    def send_message(self, test_prompts):
        self.model.eval()
        return_outputs = []
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return_outputs.append(f"{decoded_output}")
        return return_outputs
            
    def save(self, save_path):
        # Save the model and tokenizer
        model_path = os.path.join(save_path, 'model')
        tokenizer_path = os.path.join(save_path, 'tokenizer')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        
    @staticmethod
    def load(save_path, device=None):
        # Load the fine-tuned model and tokenizer
        model_path = os.path.join(save_path, 'model')
        tokenizer_path = os.path.join(save_path, 'tokenizer')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') if device is None else device
        model.to(device)

        # Create a fine tuner instance with loaded model and tokenizer
        fine_tuner = FineTuner(model, tokenizer, device=device)
        return fine_tuner
