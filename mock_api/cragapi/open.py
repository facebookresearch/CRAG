# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import json
import os
import string

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")


class OpenKG(object):
    def __init__(self):
        self.kg = {}
        for i in range(2):
            open_kg_file = os.path.join(KG_BASE_DIRECTORY, "open", "kg."+str(i)+".jsonl.bz2")
            logger.info(f"Reading open_kg file from: {open_kg_file}")
            with bz2.open(open_kg_file, "rt", encoding='utf8') as f:
                l = f.readline()
                while l:
                    l = json.loads(l)
                    self.kg[l[0]] = l[1]
                    l = f.readline()
        self.key_map = {}
        self.corpus = []
        for e in self.kg:
            ne = self.normalize(e)
            if ne not in self.key_map:
                self.key_map[ne] = []
            self.key_map[ne].append(e)
            self.corpus.append(ne)
        self.corpus = list(set(self.corpus))
        self.corpus.sort()
        self.corpus = [ne.split() for ne in self.corpus]
        self.bm25 = BM25Okapi(self.corpus)
        
        logger.info("Open KG initialized ✅")

        
    def normalize(self, x):
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())

    def search_entity_by_name(self, query):
        n = 10
        query = self.normalize(query)
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:n]
        top_ne = [" ".join(self.corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne:
            assert(ne in self.key_map)
            top_e += self.key_map[ne]
        return top_e[:n]

    def get_entity(self, entity):
        return self.kg[entity] if entity in self.kg else None        
