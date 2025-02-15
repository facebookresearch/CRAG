import os
from typing import Any, Dict, List
import httpx
import json

# Load environment variables from .env file
from dotenv import dotenv_values
config = dotenv_values('.env')

# GigaChat API Configuration
GIGACHAT_API_KEY = config["GIGACHAT_API_KEY"]
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
CA_CERT_PATH = "cert/russiantrustedca.pem"  # Path to the CA certificate file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

######################################################################################################
######################################################################################################
###
### Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

#### CONFIG PARAMETERS END---

class InstructModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.initialize_models()

    def initialize_models(self):
        # Initialize HTTP client for GigaChat API
        self.client = httpx.Client(verify=CA_CERT_PATH)

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query.
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        queries = batch["query"]
        formatted_prompts = self.format_prompts(queries)

        # Generate responses via GigaChat API
        headers = {
            "Authorization": f"Bearer {GIGACHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        responses = []
        for prompt in formatted_prompts:
            response = self.client.post(
                f"{GIGACHAT_API_URL}/chat/completions",
                headers=headers,
                data=json.dumps({"model": "GigaChat", "messages": prompt})
            )
            response_data = response.json()
            responses.append(response_data["choices"][0]["message"]["content"])

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            answers.append(response[:75])

        return answers

    def format_prompts(self, queries):
        """
        Formats queries using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        system_prompt = {
            "role": "system", 
            "content": "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'."
        }
        formatted_prompts = []

        for query in queries:
            user_message = {
                "role": "user", 
                "content": f"Question: {query}\n"
            }

            formatted_prompts.append(
                [system_prompt, user_message]
            )

        return formatted_prompts