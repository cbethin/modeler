import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, DataCollatorForSeq2Seq
from transformers import Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class FlanT5FineTuner:
    def __init__(self, model_name="google/flan-t5-small", device=None):
        # Set device
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') if device is None else device

        # Load the model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model.to(self.device)

        # Resize model embeddings
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

        # Function to monitor memory usage periodically
        max_memory = [0]
        process = psutil.Process()
        stop_event = threading.Event()

        def monitor_memory():
            while not stop_event.is_set():
                current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                max_memory[0] = max(max_memory[0], current_memory)
                time.sleep(1)  # Check memory every second

        # Train the model and measure training time and peak memory usage
        print("--- Starting Training ---\n")
        param_trainer.train()
        
    def send_message(self, test_prompts):
        self.model.eval()
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Q: {prompt}\nA: {decoded_output}\n")
            
    def save(self, save_path):
        # Save the model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
    @staticmethod
    def load(model_path, device=None):
        # Load the fine-tuned model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') if device is None else device
        model.to(device)

        # Create a fine tuner instance with loaded model and tokenizer
        fine_tuner = FlanT5FineTuner()
        fine_tuner.model = model
        fine_tuner.tokenizer = tokenizer
        fine_tuner.device = device

        return fine_tuner
