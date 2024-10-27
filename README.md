# Modeler
================

A library for fine-tuning and serving models (like Flan-T5, Bart, LLaMa), including integration with OpenAI GPT models.

## Table of Contents
-----------------

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Fine Tuning Models](#fine-tuning-models)
5. [Sending Messages with Fine Tuned Models](#sending-messages-with-fine-tuned-models)
6. [Flan-T5 Chat Server](#flant5-chat-server)

## Getting Started
---------------

To get started, install the repository by running `pip install .` in your terminal.

## Usage
-----

To use the repository, you can start the Flask chat server by running `flan_t5_fine_tuner chat_server.py` in your terminal.

### Fine Tuning Models

Fine tuning models involves training a model on a dataset of text prompts and responses. To fine tune a model, create an instance of the `FineTuner` class and call its `fine_tune` method with your dataset and other parameters.

```python
from flan_t5_fine_tuner import FineTuner

model = AutoModelForSeq2SeqLM.from_pretrained('flan-t5-small')
tokenizer = AutoTokenizer.from_pretrained('flan-t5-small')

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fine_tuner = FineTuner(model, tokenizer, device=device)

# Define your dataset and parameters
training_data = ...  # Your dataset of text prompts and responses
epochs = 5
batch_size = 16
learning_rate = 3e-4
weight_decay = 0.01

# Fine tune the model
fine_tuner.fine_tune(training_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
```

### Sending Messages with Fine Tuned Models

Once you have fine tuned a model, you can use it to send messages by calling its `send_message` method.

```python
# Define your text prompts
test_prompts = ['What is the meaning of life?', 'Can you tell me a joke?']

# Get the fine tuned model
fine_tuned_model = ...  # Your fine tuned FineTuner instance

# Send messages with the fine tuned model
messages = fine_tuned_model.send_message(test_prompts)
```

## Flan-T5 Chat Server
-------------------

The repository includes a Flask chat server that can be used to serve models like Flan-T5. To use this, create an instance of the `ChatServer` class and call its `start_server` method.

```python
from modeler.chat_server import ChatServer

# Define your fine tuned model
model = AutoModelForSeq2SeqLM.from_pretrained('flan-t5-small')
tokenizer = AutoTokenizer.from_pretrained('flan-t5-small')

# Create a chat server instance
chat_server = ChatServer(model, tokenizer)

# Start the server
chat_server.start_server(port=5042)
```

## Contributing
------------

If you would like to contribute to this repository, please fork it and submit a pull request.

## License
-------

This repository is licensed under the MIT license.