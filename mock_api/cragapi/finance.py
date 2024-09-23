# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import string

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi
from sqlitedict import SqliteDict

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")
##########################################################################################
# The following are the mock API functions needed for the Finance domain.
##########################################################################################

class FinanceKG():
    def __init__(self):
        self.fuzzy_n = 10
        company_dict_file_path = os.path.join(KG_BASE_DIRECTORY, "finance", 'company_name.dict')
        logger.info(f"Reading {company_dict_file_path}")
        df = pd.read_csv(company_dict_file_path)[["Name", "Symbol"]]
        self.name_dict = dict(df.values)

        self.key_map = dict()
        self.corpus = []
        for e in self.name_dict:
            ne = self.normalize(e)
            if ne not in self.key_map:
                self.key_map[ne] = []
            self.key_map[ne].append(e)
            self.corpus.append(ne.split())
        self.bm25 = BM25Okapi(self.corpus)
        self._load_db()
        logger.info("Finance KG initialized ✅")
    
    def _load_db(self):
        # Price history
        price_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_price.sqlite")
        logger.info(f"Reading price history from: {price_history_path}")
        self.price_history = SqliteDict(price_history_path)

        # Detailed price history
        detailed_price_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_detailed_price.sqlite")
        logger.info(f"Reading detailed price history from: {detailed_price_history_path}")
        self.detailed_price_history = SqliteDict(detailed_price_history_path)

        # Dividend history
        dividend_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_dividend.sqlite")
        logger.info(f"Reading dividend history from: {dividend_history_path}")
        self.dividend_history = SqliteDict(dividend_history_path)

        # Market cap
        market_cap_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_marketcap.sqlite")
        logger.info(f"Reading market capitalization from: {market_cap_path}")
        self.market_cap = SqliteDict(market_cap_path)

        # Financial info
        financial_info_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_info.sqlite")
        logger.info(f"Reading financial information from: {financial_info_path}")
        self.financial_info = SqliteDict(financial_info_path)
        
    
    def normalize(self, x:str) -> str:
        """
        Normalize a given string.
        arg:
            x: str
        output:
            normalized string value: str
        """
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())

    def get_company_name(self, query:str) -> list[str]:
        """
        Given a query, return top matched company names.
        arg:
            query: str
        output:
            top matched company names: list[str]
        """
        
        query = self.normalize(query)
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:self.fuzzy_n]
        top_ne = [" ".join(self.corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne:
            assert(ne in self.key_map)
            top_e += self.key_map[ne]
        return top_e[:self.fuzzy_n]

    def get_ticker_by_name(self, company_name:str) -> str:
        """
        Return ticker name by company name.
        arg:
            company_name: the company name: str
        output:
            the ticker name of the company: str
        """
        return self.name_dict.get(company_name, None)

    def get_price_history(self, ticker_name:str):
        """
        Return 1 year history of daily Open price, Close price, High price, Low price and trading Volume.
        arg: 
            ticker_name: str
        output:
            1 year daily price history: json 
        example:
            {'2023-02-28 00:00:00 EST': {'Open': 17.258894515434886,
                                         'High': 17.371392171233836,
                                         'Low': 17.09014892578125,
                                         'Close': 17.09014892578125,
                                         'Volume': 45100},
             '2023-03-01 00:00:00 EST': {'Open': 17.090151299382544,
                                         'High': 17.094839670907174,
                                         'Low': 16.443295499989794,
                                         'Close': 16.87453269958496,
                                         'Volume': 104300},
             ...
             }
        """
        db = self.price_history
        if ticker_name in db:
            return db[ticker_name]

    def get_detailed_price_history(self, ticker_name:str):
        """ 
        Return the past 5 days' history of 1 minute Open price, Close price, High price, Low price and trading Volume, starting from 09:30:00 EST to 15:59:00 EST. Note that the Open, Close, High, Low, Volume are the data for the 1 min duration. However, the Open at 9:30:00 EST may not be equal to the daily Open price, and Close at 15:59:00 EST may not be equal to the daily Close price, due to handling of the paper trade. The sum of the 1 minute Volume may not be equal to the daily Volume.
        arg: 
            ticker_name: str
        output:
            past 5 days' 1 min price history: json  
        example:
            {'2024-02-22 09:30:00 EST': {'Open': 15.920000076293945,
                                         'High': 15.920000076293945,
                                         'Low': 15.920000076293945,
                                         'Close': 15.920000076293945,
                                         'Volume': 629},
             '2024-02-22 09:31:00 EST': {'Open': 15.989999771118164,
                                         'High': 15.989999771118164,
                                         'Low': 15.989999771118164,
                                         'Close': 15.989999771118164,
                                         'Volume': 108},
              ...
            }
        """
        db = self.detailed_price_history
        if ticker_name in db:
            return db[ticker_name]

    def get_dividends_history(self, ticker_name:str):
        """
        Return dividend history of a ticker.
        arg: 
            ticker_name: str
        output:
            dividend distribution history: json
        example:
            {'2019-12-19 00:00:00 EST': 0.058,
             '2020-03-19 00:00:00 EST': 0.2,
             '2020-06-12 00:00:00 EST': 0.2,
             ...
             }
        """
        db = self.dividend_history
        if ticker_name in db:
            return db[ticker_name]

    def get_market_capitalization(self, ticker_name: str) -> float:
        """
        Return the market capitalization of a ticker.
        arg: 
            ticker_name: str
        output:
            market capitalization: float
        """
        db = self.market_cap
        if ticker_name in db:
            return db[ticker_name]

    def get_eps(self, ticker_name:str) -> float:
        """
        Return earnings per share of a ticker.
        arg: 
            ticker_name: str
        output:
            earnings per share: float
        """
        db = self.financial_info
        if ticker_name in db and 'forwardEps' in db[ticker_name]:
            return db[ticker_name]['forwardEps']

    def get_pe_ratio(self, ticker_name:str) -> float:
        """
        Return price-to-earnings ratio of a ticker.
        arg: 
            ticker_name: str
        output:
            price-to-earnings ratio: float
        """
        db = self.financial_info
        if ticker_name in db and 'forwardPE' in db[ticker_name]:
            return db[ticker_name]['forwardPE']
    
    def get_info(self, ticker_name:str):
        """
        Return meta data of a ticker.
        arg: 
            ticker_name: str
        output:
            meta information: json
        """
        db = self.financial_info
        if ticker_name in db:
            return db[ticker_name]
