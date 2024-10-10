# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import random
import sqlite3 as sql

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from loguru import logger

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")

class SoccerKG:
    def __init__(self, file_name='soccer_team_match_stats.pkl'):
        """
            Load soccer KG at different time stamp for public and private set
        """
        soccer_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", file_name)
        logger.info(f"Reading soccer KG from: {soccer_kg_file}")
        team_match_stats = pd.read_pickle(os.path.join(KG_BASE_DIRECTORY, "sports", file_name))
        self.team_match_stats = team_match_stats[team_match_stats.index.get_level_values('league').notna()]
        logger.info("Soccer KG initialized ✅")

    # ==================== APIs for competitors ====================

    def get_games_on_date(self, date_str, soccer_team_name=None):
        """ 
            Description: Get all soccer game rows given date_str
            Input: 
                - soccer_team_name: soccer team name, if None, get results for all teams
                - date_str: in format of %Y-%m-%d, %Y-%m or %Y, e.g. 2024-03-01, 2024-03, 2024
            Output: a json contains info of the games
        """
        parts = date_str.split('-')
        if soccer_team_name is None:
            filtered_df = self.team_match_stats
        else:
            filtered_df = self.team_match_stats.loc[(slice(None), slice(None), soccer_team_name, slice(None)), :]
        if len(parts) == 3:
            # date
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y-%m-%d') == date_str]    
        elif len(parts) == 2: 
            # month year
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y-%m') == date_str]
        elif len(parts) == 1:
            # year
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y') == date_str]
        else:
            filtered_df = None
        if filtered_df is not None and len(filtered_df) > 0:
            return filtered_df.to_json(date_format='iso')

class NBAKG:
    # ==================== Helper funcs ====================
    def __init__(self):
        nba_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", 'nba.sqlite')
        logger.info(f"Reading NBA KG from: {nba_kg_file}")
        self.conn = sql.connect(nba_kg_file) # create connection object to database
        logger.info("NBA KG initialized ✅")

    def get_time_cond(self, date_str):
        """Helper funcs"""
        parts = date_str.split('-')
        if len(parts) == 3:
            # date
            return f"strftime('%Y-%m-%d',game_date) = '{date_str}'"
        elif len(parts) == 2: 
            # month year
            return f"strftime('%Y-%m',game_date) = '{date_str}'"
        elif len(parts) == 1:
            # year
            return f"strftime('%Y',game_date) = '{date_str}'"
        else:
            return "1"
    
    def team_in_game_cond(self, basketball_team_name):
        """Helper funcs"""
        return f"(team_name_home = '{basketball_team_name}' or team_name_away = '{basketball_team_name}')"

    # ==================== API for competitors ====================

    def get_games_on_date(self, date_str, basketball_team_name=None):
        """ 
            Description: Get all nba game rows given date_str
            Input: date_str in format of %Y-%m-%d, %Y-%m, or %Y, e.g. 2023-01-01, 2023-01, 2023, basketball_team_name (Optional)
            Output: a json contains info of the game
        """
        if basketball_team_name is not None:
            team_cond = self.team_in_game_cond(basketball_team_name)
            time_cond = self.get_time_cond(date_str)
            df_game_by_team = pd.read_sql(f"select * from game where {team_cond} and {time_cond}", self.conn)
            if len(df_game_by_team) > 0:
                return df_game_by_team.to_json(date_format='iso')
        else:
            time_cond = self.get_time_cond(date_str)
            df_game_by_team = pd.read_sql(f"select * from game where {time_cond}", self.conn)
            if len(df_game_by_team) > 0:
                return df_game_by_team.to_json(date_format='iso')

    def get_play_by_play_data_by_game_ids(self, game_ids):
        """
        Description: Get all nba play by play rows given game ids
        Input: list of nba game ids, e.g., ["0022200547", "0029600027"]
        Output: info of the play by play events of given game id
        """
        game_ids_str = ', '.join(f"'{game_id}'" for game_id in game_ids)
        df_play_by_play_by_gameids = pd.read_sql(f"select * from play_by_play where game_id in ({game_ids_str})", self.conn)
        if len(df_play_by_play_by_gameids) > 0:
            return df_play_by_play_by_gameids.to_json(date_format='iso')   
