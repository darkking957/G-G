import time
import os
import dotenv
import tiktoken
from .base_language_model import BaseLanguageModel

dotenv.load_dotenv()
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

# Define model lists
DEEPSEEK_MODEL = ['deepseek-chat']
OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

def get_token_limit(model='deepseek-chat'):
    """Returns the token limitation of provided model"""
    if model in ['deepseek-chat']:
        num_tokens_limit = 8192  # Assumption - replace with actual DeepSeek limit
    elif model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
    
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150 
        
        # Initialize appropriate client based on model
        if self.model_name in DEEPSEEK_MODEL:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key="sk-347835eac1c048209a0a27a62e5b2341",
                    base_url="https://api.deepseek.com"
                )
                self.use_deepseek = True
            except ImportError:
                raise ImportError("OpenAI package is required for DeepSeek API. Please install it with 'pip install openai'.")
        else:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            self.use_deepseek = False
    
    def tokenize(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            if self.use_deepseek:
                # Simple approximation for DeepSeek models
                num_tokens = int(len(text.split()) * 1.3)
            else:
                encoding = tiktoken.encoding_for_model(self.model_name)
                num_tokens = len(encoding.encode(text))
        except KeyError:
            raise KeyError(f"Warning: model {self.model_name} not found.")
        return num_tokens + self.redundant_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        '''
        Model does not need to prepare for inference
        '''
        pass
    
    def generate_sentence(self, llm_input):
        cur_retry = 0
        num_retry = self.retry
        
        # Check if the input is too long
        input_length = self.tokenize(llm_input)
        if input_length > self.maximun_token:
            print(f"Input length {input_length} is too long. The maximum token is {self.maximun_token}.\n Right truncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        
        while cur_retry <= num_retry:
            try:
                if self.use_deepseek:
                    # DeepSeek implementation
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": llm_input}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False
                    )
                    result = response.choices[0].message.content.strip()
                else:
                    # Original OpenAI implementation
                    import openai
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": llm_input}],
                        request_timeout=30,
                    )
                    result = response["choices"][0]["message"]["content"].strip() # type: ignore
                
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.tokenize(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None