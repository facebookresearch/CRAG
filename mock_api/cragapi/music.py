# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
import pickle
import string
import time
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .fast_bm25 import BM25

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")


class MusicKG(object):
    def __init__(self):
        # Reading the artist dictionary
        artist_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_dict_simplified.pickle")
        logger.info(f"Reading artist dictionary from: {artist_dict_path}")
        with open(artist_dict_path, 'rb') as file:
            self.artist_dict = pickle.load(file)

        # Reading the song dictionary
        song_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_simplified.pickle")
        logger.info(f"Reading song dictionary from: {song_dict_path}")
        with open(song_dict_path, 'rb') as file:
            self.song_dict = pickle.load(file)

        # Reading the Grammy DataFrame
        grammy_df_path = os.path.join(KG_BASE_DIRECTORY, "music", "grammy_df.pickle")
        logger.info(f"Reading Grammy DataFrame from: {grammy_df_path}")
        with open(grammy_df_path, 'rb') as file:
            self.grammy_df = pickle.load(file)

        # Reading the rank dictionary for Hot 100
        rank_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "rank_dict_hot100.pickle")
        logger.info(f"Reading rank dictionary for Hot 100 from: {rank_dict_hot_path}")
        with open(rank_dict_hot_path, 'rb') as file:
            self.rank_dict_hot = pickle.load(file)

        # Reading the song dictionary for Hot 100
        song_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_hot100.pickle")
        logger.info(f"Reading song dictionary for Hot 100 from: {song_dict_hot_path}")
        with open(song_dict_hot_path, 'rb') as file:
            self.song_dict_hot = pickle.load(file)

        # Reading the artist work dictionary
        artist_work_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_work_dict.pickle")
        logger.info(f"Reading artist work dictionary from: {artist_work_dict_path}")
        with open(artist_work_dict_path, 'rb') as file:
            self.artist_work_dict = pickle.load(file)
        
        self.key_map_artist = {}
        self.corpus_artist = []
        for e in self.artist_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_artist:
                self.key_map_artist[ne_split] = []
            self.key_map_artist[ne_split].append(e)
            self.corpus_artist.append(ne)
        self.corpus_artist = list(set(self.corpus_artist))
        self.corpus_artist.sort()
        self.corpus_artist = [ne.split() for ne in self.corpus_artist]
        self.bm25_artist = BM25(self.corpus_artist)

        self.key_map_song = {}
        self.corpus_song = []
        for e in self.song_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_song:
                self.key_map_song[ne_split] = []
            self.key_map_song[ne_split].append(e)
            self.corpus_song.append(ne)
        self.corpus_song = list(set(self.corpus_song))
        self.corpus_song.sort()
        self.corpus_song = [ne.split() for ne in self.corpus_song]
        self.bm25_song = BM25(self.corpus_song)
        
        logger.info("Music KG initialized ✅")

    
    def normalize(self, x):
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())
    
    def search_artist_entity_by_name(self, query):
        """ Return the fuzzy matching results of the query (artist name); we only return the top-10 similar results from our KB

        Args:
            query (str): artist name

        Returns:
            Top-10 similar entity name in a list
        
        """
        n = 10
        query = self.normalize(query)
        results = self.bm25_artist.get_top_n(query.split(), self.corpus_artist, n=n)
        top_e = []
        for cur_ne_str in results:
            assert(str(cur_ne_str) in self.key_map_artist.keys())
            top_e += self.key_map_artist[str(cur_ne_str)]
        return top_e[:n]
    
    def search_song_entity_by_name(self, query):
        """ Return the fuzzy matching results of the query (song name); we only return the top-10 similar results from our KB

        Args:
            query (str): song name

        Returns:
            Top-10 similar entity name in a list
        
        """
        n = 10
        query = self.normalize(query)
        results = self.bm25_song.get_top_n(query.split(), self.corpus_song, n=n)
        top_e = []
        for cur_ne_str in results:
            assert(str(cur_ne_str) in self.key_map_song.keys())
            top_e += self.key_map_song[str(cur_ne_str)]
        return top_e[:n]

    def get_billboard_rank_date(self, rank, date=None):
        """ Return the song name(s) and the artist name(s) of a certain rank on a certain date; 
            If no date is given, return the list of of a certain rank of all dates. 

        Args:
            rank (int): the interested rank in billboard; from 1 to 100.
            date (Optional, str, in YYYY-MM-DD format): the interested date; leave it blank if do not want to specify the date.
        
        Returns:
            rank_list (list): a list of song names of a certain rank (on a certain date).
            artist_list (list): a list of author names corresponding to the song names returned.
        """

        rank_list = []
        artist_list = []        
        if not str(rank) in self.rank_dict_hot.keys():
            return None, None
        else:
            if date:
                for item in self.rank_dict_hot[str(rank)]:
                    if item['Date'] == date:
                        return [item['Song']], [item['Artist']]
            else:
                for item in self.rank_dict_hot[str(rank)]:
                    rank_list.append(item['Song'])
                    artist_list.append(item['Artist'])
        return rank_list, artist_list
    
    def get_billboard_attributes(self, date, attribute, song_name):
        """ Return the attributes of a certain song on a certain date
        
        Args:
            date (str, in YYYY-MM-DD format): the interested date of the song
            attribute (str): attributes from ['rank_last_week', 'weeks_in_chart', 'top_position', 'rank']
            song_name (str): the interested song name
        
        Returns:
            cur_value (str): the value of the interested attribute of a song on a certain date
        """
        if not song_name in self.song_dict_hot:
            return None
        else:
            cur_dict = self.song_dict_hot[song_name]
            if not date in cur_dict.keys():
                return None
            else:
                row = cur_dict[date]
                if row[6] == '-':
                    if attribute == 'rank_last_week':
                        cur_value = row[6]
                    elif attribute == 'weeks_in_chart':
                        cur_value = row[5]
                    elif attribute == 'top_position':
                        cur_value = row[4]
                    else:
                        cur_value = row[3]
                else:
                    if attribute == 'rank_last_week':
                        cur_value = row[4]
                    elif attribute == 'weeks_in_chart':
                        cur_value = row[6]
                    elif attribute == 'top_position':
                        cur_value = row[5]
                    else:
                        cur_value = row[3]
                return cur_value
        
    def grammy_get_best_artist_by_year(self, year):
        """ Return the Best New Artist of a certain year in between 1958 and 2019

        Args:
            year (int, in YYYY format): the interested year
        
        Returns:
            artist_list (list): the list of artists who win the award
        """
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Best New Artist') & (self.grammy_df['year'] == year)]
            artist_list = filtered_df['nominee'].tolist()
            return artist_list
    
    def grammy_get_award_count_by_artist(self, artist_name):
        """ Return the number of awards won by a certain artist between 1958 and 2019

        Args:
            artist_name (str): the name of the artist
        
        Returns:
            the number of total awards (int)
        """
        total_unique_rows_artist = 0
        total_unique_rows_nominee = 0
        total_unique_rows_worker = 0
        for value in self.grammy_df['nominee']:
            if artist_name in str(value):
                total_unique_rows_nominee += 1
        for value in self.grammy_df['artist']:
            if artist_name in str(value):
                total_unique_rows_artist += 1
        for value in self.grammy_df['workers']:
            if artist_name in str(value):
                total_unique_rows_worker += 1
        return total_unique_rows_nominee + total_unique_rows_artist + total_unique_rows_worker

    def grammy_get_award_count_by_song(self, song_name):
        """ Return the number of awards won by a certain song between 1958 and 2019

        Args:
            song_name (str): the name of the song
        
        Returns:
            the number of total awards (int)
        """
        total_unique_rows_nominee = len(self.grammy_df[self.grammy_df['nominee']==song_name])
        return total_unique_rows_nominee
    
    def grammy_get_best_song_by_year(self, year):
        """ Return the Song Of The Year in a certain year between 1958 and 2019
        
        Args:
            year (int, in YYYY format): the interested year
        
        Returns:
            song_list (list): the list of the song names that win the Song Of The Year in a certain year
        """
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Song Of The Year') & (self.grammy_df['year'] == year)]
            song_list = filtered_df['nominee'].tolist()
            return song_list
    
    def grammy_get_award_date_by_artist(self, artist_name):
        """ Return the award winning years of a certain artist

        Args:
            artist_name (str): the name of the artist

        Returns:
            selected_years (list): the list of years the artist is awarded
        """
        idx = []
        for i, value in enumerate(self.grammy_df['nominee']):
            if artist_name in str(value):
                idx.append(i)
        for i, value in enumerate(self.grammy_df['artist']):
            if artist_name in str(value):
                idx.append(i)
        for i, value in enumerate(self.grammy_df['workers']):
            if artist_name in str(value):
                idx.append(i)
        selected_idx = list(set(idx))
        selected_years = []
        for cur_idx in selected_idx:
            selected_years.append(self.grammy_df['year'][cur_idx])
        selected_years = list(set(selected_years))
        selected_years = [int(x) for x in selected_years]
        return selected_years

    def grammy_get_best_album_by_year(self, year):
        """ Return the Album Of The Year of a certain year between 1958 and 2019

        Args:
            year (int, in YYYY format): the interested year
        
        Returns:
            song_list (list): the list of albums that won the Album Of The Year in a certain year
        """
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Album Of The Year') & (self.grammy_df['year'] == year)]
            song_list = filtered_df['nominee'].tolist()
            return song_list

    def grammy_get_all_awarded_artists(self):
        """Return all the artists ever awarded Grammy Best New Artist between 1958 and 2019
        
        Args:
            None
        
        Returns:
            nominee_values (list): the list of artist ever awarded Grammy Best New Artist

        """
        nominee_values = self.grammy_df[self.grammy_df['category'] == 'Best New Artist']['nominee'].dropna().unique().tolist()
        return nominee_values
    
    def get_artist_birth_place(self, artist_name):
        """ Return the birth place country code (2-digit) for the input artist

        Args:
            artist_name (str): the name of the artist
        
        Returns:
            country (str): the two-digit country code following ISO-3166
        """
        try:
            d = self.artist_dict[artist_name]
            country = d['country']
            if country:
                return country
            else:
                return None
        except:
            return None
        
    def get_artist_birth_date(self, artist_name):
        """ Return the birth date of the artist

        Args:
            artist_name (str): the name of the artist
        
        Returns:
            life_span_begin (str, in YYYY-MM-DD format if possible): the birth date of the person or the begin date of a band
        
        """
        try:
            d = self.artist_dict[artist_name]
            life_span_begin = d['birth_date']
            if life_span_begin:
                return life_span_begin
            else:
                return None
        except:
            return None

    def get_members(self, band_name):
        """ Return the member list of a band

        Args:
            band_name (str): the name of the band
        
        Returns:
            the list of members' names.
        """
        try:
            d = self.artist_dict[band_name]
            members = d['members']
            return list(set(members))
        except:
            return None

    def get_lifespan(self, artist_name):
        """ Return the lifespan of the artist

        Args:
            artist_name (str): the name of the artist
        
        Returns:
            the birth and death dates in a list
        
        """
        try:
            d = self.artist_dict[artist_name]
            life_span_begin = d['birth_date']
            life_span_end = d['end_date']
            life = [life_span_begin, life_span_end]
            return life
        except:
            return [None, None]
    
    def get_song_author(self, song_name):
        """ Return the author of the song

        Args:
            song_name (str): the name of the song
        
        Returns:
            author (str): the author of the song
        """
        try:
            d = self.song_dict[song_name]
            author = d['author']
            if author:
                return author
            else:
                return None
        except:
            return None

    def get_song_release_country(self, song_name):
        """ Return the release country of the song

        Args:
            song_name (str): the name of the song
        
        Returns:
            country (str): the two-digit country code following ISO-3166
        """
        try:
            d = self.song_dict[song_name]
            country = d['country']
            if country:
                return country
            else:
                return None
        except:
            return None

    def get_song_release_date(self, song_name):
        """ Return the release date of the song

        Args:
            song_name (str): the name of the song
        
        Returns:
            date (str in YYYY-MM-DD format): the date of the song
        """
        try:
            d = self.song_dict[song_name]
            date = d['date']
            if date:
                return date
            else:
                return None
        except:
            return None

    def get_artist_all_works(self, artist_name):
        """ Return the list of all works of a certain artist

        Args:
            artist_name (str): the name of the artist
        
        Returns:
            work_list (list): the list of all work names
        
        """
        if artist_name in self.artist_work_dict.keys():
            work_list = self.artist_work_dict[artist_name]
            return work_list
        else:
            return None

