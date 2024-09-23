# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import string
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")

class MovieKG:
    '''Knowledge Graph API for movie domain

    Mock KG API for movie domain. Supports getting information of movies and of persons including cast and crew.
    '''
    def __init__(self, top_n: int=10) -> None:
        '''Initialize API and load data. Loads 3 dbs from json

        Args:
            top_n: max number of entities to return in entity search
        '''
        # Reading the year database
        year_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "year_db.json")
        logger.info(f"Reading year database from: {year_db_path}")
        with open(year_db_path) as f:
            self._year_db = json.load(f)

        # Reading the person database
        person_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "person_db.json")
        logger.info(f"Reading person database from: {person_db_path}")
        with open(person_db_path) as f:
            self._person_db = json.load(f)

        # Reading the movie database
        movie_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "movie_db.json")
        logger.info(f"Reading movie database from: {movie_db_path}")
        with open(movie_db_path) as f:
            self._movie_db = json.load(f)

        self._top_n = top_n
        self._person_db_lookup = self._get_direct_lookup_db(self._person_db)
        self._movie_db_lookup = self._get_direct_lookup_db(self._movie_db)
        self._movie_corpus, self._movie_bm25 = self._get_ranking_db(self._movie_db)
        self._person_corpus, self._person_bm25 = self._get_ranking_db(self._person_db)
        
        logger.info("Movie KG initialized ✅")

    def _normalize(self, x: str) -> str:
        '''Helper function for normalizing text

        Args:
            x: string to be normalized

        Returns:
            normalized string
        '''
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())

    def _get_ranking_db(self, db: Dict[str, Any]) -> Tuple[List[str], BM25Okapi]:
        '''Helper function to get BM25 index

        Args:
            db: dictionary of entities keyed by entity name

        Returns:
            corpus: list of entity names corresponding to BM25 index position
            bm25: BM25 index
        '''
        corpus = [i.split() for i in db.keys()]
        bm25 = BM25Okapi(corpus)
        return corpus, bm25

    def _get_direct_lookup_db(self, db: Dict[str, Any]) -> Dict[int, Any]:
        '''Converts name-indexed db to id-indexed db for latency optimization

        Args:
            db: dictionary of entities keyed by normalized entity name

        Returns:
            dictionary of entities keyed by unique entity id
        '''
        temp_db = {}
        for key, value in db.items():
            if 'id' in value:
                temp_db[value['id']] = value
        return temp_db

    def _search_entity_by_name(self, query: str, bm25: BM25Okapi, corpus: List[str], map_db: Dict[str, Any]) -> List[Dict[str, Any]]:
        '''BM25 search for top n=10 matching entities

        Args:
            query: string to be searched
            bm25: BM25 index
            corpus: list of entity names corresponding to BM25 index position
            map_db: dictionary of entities keyed by normalized entity name

        Returns:
            list of top n matching entities. Each entity is a tuple of (normalized entity name, entity info)
        '''
        n = self._top_n
        query = self._normalize(query)
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:n]
        top_ne = [" ".join(corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne[:n]:
            assert(ne in map_db)
            top_e.append(map_db[ne])
        return top_e[:n]

    def get_person_info(self, person_name: str) -> List[Dict[str, Any]]:
        '''Gets person info in database through BM25.

        Gets person info through BM25 Search. The returned entities MAY contain the following fields:
            - name (string): name of person
            - id (int): unique id of person
            - acted_movies (list[int]): list of movie ids in which person acted
            - directed_movies (list[int]): list of movie ids in which person directed
            - birthday (string): string of person's birthday, in the format of "YYYY-MM-DD"
            - oscar_awards: list of oscar awards (dict), win or nominated, in which the person was the entity. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award

        Args:
            person_name: string to be searched

        Returns:
            list of top n matching entities. Entities are ranked by BM25 score.
        '''
        res = self._search_entity_by_name(person_name, self._person_bm25, self._person_corpus, self._person_db)
        return res

    def get_movie_info(self, person_name: str) -> List[Dict[str, Any]]:
        '''Gets movie info in database through BM25.

        Gets movie info through BM25 Search. The returned entities MAY contain the following fields:
            - title (string): title of movie
            - id (int): unique id of movie
            - release_date (string): string of movie's release date, in the format of "YYYY-MM-DD"
            - original_title (string): original title of movie, if in another language other than english
            - original_language (string): original language of movie. Example: 'en', 'fr'
            - budget (int): budget of movie, in USD
            - revenue (int): revenue of movie, in USD
            - rating (float): rating of movie, in range [0, 10]
            - genres (list[dict]): list of genres of movie. Sample genre object is {'id': 123, 'name': 'action'}
            - oscar_awards: list of oscar awards (dict), win or nominated, in which the movie was the entity. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award
            - cast (list [dict]): list of cast members of the movie and their roles. The format of the cast member entity is:
                'name' (string): name of the cast member,
                'id' (int): unique id of the cast member,
                'character' (string): character played by the cast member in the movie,
                'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
                'order' (int): order of the cast member in the movie. For example, the actress with the lowest order is the main actress,
            - crew' (list [dict]): list of crew members of the movie and their roles. The format of the crew member entity is:
                'name' (string): name of the crew member,
                'id' (int): unique id of the crew member,
                'job' (string): job of the crew member,

        Args:
            movie_name: string to be searched

        Returns:
            list of top n matching entities. Entities are ranked by BM25 score.
        '''
        res = self._search_entity_by_name(person_name, self._movie_bm25, self._movie_corpus, self._movie_db)
        return res

    def get_year_info(self, year: str) -> Dict[str, Any]:
        '''Gets info of a specific year

        Gets year info. The returned entity MAY contain the following fields:
            - movie_list: list of movie ids in the year. This field can be very long to a few thousand films
            - oscar_awards: list of oscar awards (dict), held in that particular year. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award

        Args:
            year: string of year. Note that we only support years between 1990 and 2021

        Returns:
            an entity representing year information
        '''
        if int(year) not in range(1990, 2022):
            raise ValueError("Year must be between 1990 and 2021")
        return self._year_db.get(str(year), None)

    def get_movie_info_by_id(self, movie_id: int) -> Dict[str, Any]:
        '''Helper fast lookup function to get movie info directly by id

        Return a movie entity with same format as the entity in get_movie_info.

        Args:
            movie_id: unique id of movie

        Returns:
            an entity representing movie information
        '''
        return self._movie_db_lookup.get(movie_id, None)

    def get_person_info_by_id(self, person_id: int) -> Dict[str, Any]:
        '''Helper fast lookup function to get person info directly by id

        Return a person entity with same format as the entity in get_person_info.

        Args:
            person_id: unique id of person

        Returns:
            an entity representing person information
        '''
        return self._person_db_lookup.get(person_id, None)
