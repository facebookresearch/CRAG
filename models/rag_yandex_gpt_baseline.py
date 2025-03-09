# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import requests
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from yandex_cloud_ml_sdk import YCloudML  # Yandex Cloud ML SDK for embeddings

from dotenv import dotenv_values

config = dotenv_values(".env")

# YandexGPT API Configuration
YANDEX_API_KEY = config["YCLOUD_API_TOKEN"]
YANDEX_FOLDER_ID = config["YCLOUD_FOLDER_ID"]
YANDEXGPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
SUBMISSION_BATCH_SIZE = 8


class ChunkExtractor:
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        soup = BeautifulSoup(html_source, "html.parser")
        text = soup.get_text(" ", strip=True)

        if not text:
            return interaction_id, [""]

        _, offsets = text_to_sentences_and_offsets(text)
        chunks = [
            text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH] for start, end in offsets
        ]

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        chunk_dict = defaultdict(list)
        for ref in ray_response_refs:
            iid, chunks = ray.get(ref)
            chunk_dict[iid].extend(chunks)

        return self._flatten_chunks(chunk_dict)

    def _flatten_chunks(self, chunk_dict):
        chunks, ids = [], []
        for iid, texts in chunk_dict.items():
            unique = list(set(texts))
            chunks.extend(unique)
            ids.extend([iid] * len(unique))
        return np.array(chunks), np.array(ids)


class RAGModel:
    def __init__(self):
        self.sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
        self.chunk_extractor = ChunkExtractor()
        self.embedding_model = {
            "query": self.sdk.models.text_embeddings("query"),
            "doc": self.sdk.models.text_embeddings("doc"),
        }
        self.llm = self.sdk.models.completions("yandexgpt").configure(
            temperature=0.1, max_tokens=75
        )

    def get_batch_size(self) -> int:
        return SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        chunks, chunk_ids = self.chunk_extractor.extract_chunks(
            batch["interaction_id"], batch["search_results"]
        )

        # Calculate embeddings
        query_embs = np.array(
            [self._get_embedding(text, "query") for text in batch["query"]]
        )
        chunk_embs = np.array([self._get_embedding(text, "doc") for text in chunks])

        # Retrieve context
        contexts = []
        for i, q_emb in enumerate(query_embs):
            mask = chunk_ids == batch["interaction_id"][i]
            scores = chunk_embs[mask] @ q_emb
            top_indices = (-scores).argsort()[:NUM_CONTEXT_SENTENCES]
            contexts.append("\n".join(chunks[mask][top_indices]))

        # Generate answers
        return [
            self._generate_answer(
                query=batch["query"][i],
                context=contexts[i],
                timestamp=batch["query_time"][i],
            )
            for i in range(len(batch["query"]))
        ]

    def _get_embedding(self, text: str, emb_type: str) -> np.ndarray:
        try:
            return self.embedding_model[emb_type].run(text)
        except Exception as e:
            print(f"Embedding Error: {str(e)}")
            return np.zeros(768)

    def _generate_answer(self, query: str, context: str, timestamp: str) -> str:
        messages = [
            {
                "role": "system",
                "text": "Answer using ONLY the provided references. If unsure, say 'I don't know'.",
            },
            {
                "role": "user",
                "text": f"References:\n{context}\n\nCurrent Time: {timestamp}\nQuestion: {query}",
            },
        ]

        try:
            result = self.llm.run(messages)
            return result[0].text[:75]
        except Exception as e:
            print(f"Generation Error: {str(e)}")
            return "I don't know"
