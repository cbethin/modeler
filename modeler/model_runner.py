import openai

class ModelRunner:
    def __init__(self, fine_tuner=None, base_url=None, api_key=None, model=None):
        if fine_tuner:
            self.model = fine_tuner.model
            self.tokenizer = fine_tuner.tokenizer
            self.device = fine_tuner.device
            self.is_local_model = True
        elif base_url and api_key:
            self.model_name = model
            self.client = openai.ChatCompletion(api_base=base_url, api_key=api_key)
            self.is_local_model = False
        else:
            raise ValueError("Either fine_tuner or base_url and api_key must be provided.")

    def chat_completion(self, messages):
        if self.is_local_model:
            prompt = ""
            for message in messages:
                if message['role'] == 'user':
                    prompt += message['content'] + "\n"

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {'choices': [{'message': {'role': 'assistant', 'content': decoded_output}}]}
        else:
            response = self.client.create(model=self.model_name, messages=messages)
            return response
