import openai

def create_choices(content: str) -> str:
    output = {
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': content.replace('system:', '').replace('assistant:', '').replace('user:', '')
                }
            }
        ]
    }
    
    print(output)
    return output

class ModelRunner():
    def __init__(self, fine_tuner=None, generate_from_prompt=None, base_url=None, api_key=None, model=None):
        if fine_tuner:
            self.model = fine_tuner.model
            self.tokenizer = fine_tuner.tokenizer
            self.device = fine_tuner.device
            self.is_local_model = True
            self.generate_from_prompt = None
        elif generate_from_prompt:
            self.generate_from_prompt = generate_from_prompt
            self.is_local_model = False
        elif base_url and api_key:
            self.model_name = model
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            self.is_local_model = False
            self.generate_from_prompt = None
        else:
            raise ValueError("Either fine_tuner, pipeline, or base_url and api_key must be provided.")

    def chat_completion(self, messages, **args):
        prompt = messages[-1]['content']
            
        if self.generate_from_prompt:
            content = self.generate_from_prompt(prompt)
            return create_choices(content)
            
        if self.is_local_model:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return create_choices(decoded_output)
        else:
            response = self.client.chat.completions.create(model=self.model_name, messages=messages)
            return response