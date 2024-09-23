# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from collections import defaultdict
from json import JSONDecoder
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger
from sentence_transformers import SentenceTransformer
from utils.cragapi_wrapper import CRAG

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

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
SUBMISSION_BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

# entity extraction template
Entity_Extract_TEMPLATE = """
You are given a Query and Query Time. Do the following: 

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

For `encyclopedia` and `other` queries, these are possible keys:
-  `main_entity`: extract the main entity of the query. 

For `finance` queries, these are possible keys:
- `market_identifier`: stock identifiers including individual company names, stock symbols.
- `metric`: financial metrics that the query is asking about. This must be one of the following: `price`, `dividend`, `P/E ratio`, `EPS`, `marketCap`, and `other`.
- `datetime`: time frame that query asks about. When datetime is not explicitly mentioned, use `Query Time` as default. 

For `movie` queries, these are possible keys:
- `movie_name`: name of the movie
- `movie_aspect`: if the query is about a movie, which movie aspect the query asks. This must be one of the following: `budget`, `genres`, `original_language`, `original_title`, `release_date`, `revenue`, `title`, `cast`, `crew`, `rating`, `length`.
- `person`: person name related to moves
- `person_aspect`: if the query is about a person, which person aspect the query asks. This must be one of the following: `acted_movies`, `directed_movies`, `oscar_awards`, `birthday`.
- `year`: if the query is about movies released in a specific year, extract the year

For `music` queries, these are possible keys:
- `artist_name`: name of the artist
- `artist_aspect`: if the query is about an artist, extract the aspect of the artist. This must be one of the following: `member`, `birth place`, `birth date`, `lifespan`, `artist work`, `grammy award count`, `grammy award date`.
- `song_name`: name of the song
- `song_aspect`: if the query is about a song, extract the aspect of the song. This must be one of the following: `auther`, `grammy award count`, `release country`, `release date`.

For `sports` queries, these are possible keys:
- `sport_type`: one of `basketball`, `soccer`, `other`
- `tournament`: such as NBA, World Cup, Olympic.
- `team`: teams that user interested in.
- `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

Return the results in a FLAT json. 

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
"""

#### CONFIG PARAMETERS END---

class ChunkExtractor:

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results

class RAG_KG_Model:
    """
    An example RAGModel
    """
    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "models/meta-llama/Meta-Llama-3-8B-Instruct"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://github.com/facebookresearch/CRAG/blob/main/docs/download_baseline_model_weights.md
            """
            )

        # Initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            worker_use_ray=True,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype= "half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "models/sentence-transformers/all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding to a query.
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

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
            
        # Retrieve knowledge graph results
        entities = self.extract_entity(batch)
        batch_kg_results = self.get_kg_results(entities)
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results, batch_kg_results)
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                
                # Note: We are using 50 max new tokens instead of 75,
                # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                # Llama3 instead uses a different tokenizer with a larger vocabulary
                # This allows the Llama3 tokenizer to represent the same content more efficiently, 
                # while using fewer tokens.
            ),
            use_tqdm=False # you might consider setting this to True during local development
        )

        # Aggregate answers into List[str]
        answers = []
        for response in responses: 
            answers.append(response.outputs[0].text)
        
        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[], batch_kg_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        - batch_kg_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            kg_results = batch_kg_results[_idx]

            user_message = ""
            retrieval_references = ""
            if len(retrieval_results) > 0:
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    retrieval_references += f"- {snippet.strip()}\n"
            # Limit the length of references to fit the model's input size.
            retrieval_references = retrieval_references[: int(MAX_CONTEXT_REFERENCES_LENGTH / 2)]
            kg_results = kg_results[: int(MAX_CONTEXT_REFERENCES_LENGTH / 2)]
            
            references = "### References\n" + \
                "# Web\n" + \
                retrieval_references + \
                "# Knowledge Graph\n" + \
                kg_results

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using only the references listed above, answer the following question: \n"
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

    def extract_entity(self, batch):
        queries = batch["query"]
        query_times = batch["query_time"]
        formatted_prompts = self.format_prompts_for_entity_extraction(queries, query_times)
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=4096,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False # you might consider setting this to True during local development
        )

        entities = []
        for response in responses:
            res = response.outputs[0].text
            try:
                res = json.loads(res)
            except:
                res = extract_json_objects(res)
            entities.append(res)
        return entities
    
    def get_kg_results(self, entities):
        # examples for "open" (encyclopedia),  "movie" or "other" domains
        api = CRAG(server=CRAG_MOCK_API_URL)
        batch_kg_results = []
        for entity in entities:
            kg_results = []
            res = ""
            if "domain" in entity.keys():
                domain = entity["domain"]
                if domain in ["encyclopedia", "other"]:
                    if "main_entity" in entity.keys():
                        try:
                            top_entity_name = api.open_search_entity_by_name(entity["main_entity"])["result"][0]
                            res = api.open_get_entity(top_entity_name)["result"]
                            kg_results.append({top_entity_name: res})
                        except Exception as e:
                            logger.warning(f"Error in open_get_entity: {e}")
                            pass
                if domain == "movie":
                    if "movie_name" in entity.keys() and entity["movie_name"] is not None:
                        if isinstance(entity["movie_name"], str):
                            movie_names = entity["movie_name"].split(",")
                        else:
                            movie_names = entity["movie_name"]
                        for movie_name in movie_names:
                            try:
                                res = api.movie_get_movie_info(movie_name)["result"][0]
                                res = res[entity["movie_aspect"]]
                                kg_results.append({movie_name + "_" + entity["movie_aspect"]: res})
                            except Exception as e:
                                logger.warning(f"Error in movie_get_movie_info: {e}")
                                pass
                    if "person" in entity.keys() and entity["person"] is not None:
                        if isinstance(entity["person"], str):
                            person_list = entity["person"].split(",")
                        else:
                            person_list = entity["person"]
                        for person in person_list:
                            try:
                                res = api.movie_get_person_info(person)["result"][0]
                                aspect = entity["person_aspect"]
                                if aspect in ["oscar_awards", "birthday"]:
                                    res = res[aspect]
                                    kg_results.append({person + "_" + aspect: res})
                                if aspect in ["acted_movies", "directed_movies"]:
                                    movie_info = []
                                    for movie_id in res[aspect]:
                                        movie_info.append(api.movie_get_movie_info_by_id(movie_id))
                                    kg_results.append({person + "_" + aspect: movie_info})
                            except Exception as e:
                                logger.warning(f"Error in movie_get_person_info: {e}")
                                pass
                    if "year" in entity.keys() and entity["year"] is not None:
                        if isinstance(entity["year"], str) or isinstance(entity["year"], int):
                            years = str(entity["year"]).split(",")
                        else:
                            years = entity["year"]
                        for year in years:
                            try:
                                res = api.movie_get_year_info(year)["result"]
                                all_movies = []
                                oscar_movies = []
                                for movie_id in res["movie_list"]:
                                    all_movies.append(api.movie_get_movie_info_by_id(movie_id)["result"]["title"])
                                for movie_id in res["oscar_awards"]:
                                    oscar_movies.append(api.movie_get_movie_info_by_id(movie_id)["result"]["title"])   
                                kg_results.append({year + "_all_movies": all_movies})
                                kg_results.append({year + "_oscar_movies": oscar_movies})
                            except Exception as e:
                                logger.warning(f"Error in movie_get_year_info: {e}")
                                pass
            batch_kg_results.append("<DOC>\n".join([str(res) for res in kg_results]) if len(kg_results) > 0 else "")
        return batch_kg_results
 
    def format_prompts_for_entity_extraction(self, queries, query_times):     
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            user_message = ""
            user_message += f"Query: {query}\n"
            user_message += f"Query Time: {query_time}\n"
            
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": Entity_Extract_TEMPLATE},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts

