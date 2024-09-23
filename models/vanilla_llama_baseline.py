# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List

import numpy as np
import torch
import vllm
from models.utils import trim_predictions_to_max_token_length

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

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

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
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "models/meta-llama/Meta-Llama-3-8B-Instruct"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            """
            )

        # initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            worker_use_ray=True,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: update the dtype based on the available GPU
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()

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
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        formatted_prompts = self.format_prommpts(queries, query_times)

        # Generate responses via vllm
        breakpoint()
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                # Note: We are using 50 max new tokens instead of 75,
                # because the 75 max token limit is checked using the Llama2 tokenizer.
                # The Llama3 model instead uses a differet tokenizer with a larger vocabulary
                # This allows it to represent the same content more efficiently, using fewer tokens.
            ),
            use_tqdm = False
        )

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            answers.append(response.outputs[0].text)

        return answers

    def format_prommpts(self, queries, query_times):
        """
        Formats queries and corresponding query_times using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
        - query_times (list of str): A list of query_time strings corresponding to each query.
            
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            user_message = ""
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts
