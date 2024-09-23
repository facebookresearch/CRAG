# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from transformers import LlamaTokenizerFast

tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer")
tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction
