import os
import json
import requests
from abc import ABC, abstractmethod

import openai
from superagi.llms.base_llm import BaseLlm
from superagi.config.config import get_config
from superagi.lib.logger import logger

class KoboldCpp(BaseLlm):
    def __init__(self, api_key, image_model=None, model="", temperature=0.6, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT"), top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0, number_of_results=80):
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.top_p = top_p
        self.model = model
        self.endpoint = get_config("CUSTOM_ENDPOINT")


    def get_model(self):
        """
        Returns:
            str: The model.
        """
        return self.model


    def get_image_model(self):
        """
        Returns:
            str: The image model.
        """
        return self.image_model

    def chat_completion(self, messages, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT")):
        """
        Call the OpenAI chat completion API.

        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            dict: The response.
        """
        prompt = '### Instruction:\n\n'
        for p in messages:
            prompt += p['content']
        prompt += '### Response:\n\n'
        try:
            url = 'http://172.17.0.1:5001'
            api_endpoint = self.endpoint
            data = dict(prompt = prompt,
            max_context_length = max_tokens,
            max_length = get_config("MAX_TOOL_TOKEN_LIMIT"),
            temperature = self.temperature,
            top_k = 120,
            top_a = 0.0,
            top_p = self.top_p,
            typical_p = 1.0,
            tfs = 1.0,
            rep_pen = 1.1,
            rep_pen_range = 128,
            seed = -1,
            stop_sequence = [],
            stream_sse = 0)

            response = requests.post(url + api_endpoint, data=json.dumps(data))
            if response.status_code == 200:
                return {"response": response, "content": response.json()['results'][0]['text']}
            return {"response": response, "content": ''}
                
        except Exception as exception:
            logger.info("Exception:", exception)
            return {"error": exception}

    def generate_image(self, prompt: str, size: int = 512, num: int = 2):
        return {}
