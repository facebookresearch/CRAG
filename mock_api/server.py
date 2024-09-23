# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Annotated
from typing import Optional
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel
from cragapi.open import OpenKG
from cragapi.movie import MovieKG
from cragapi.finance import FinanceKG
from cragapi.music import MusicKG
from cragapi.sports import SoccerKG, NBAKG

API = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    API["open"] = OpenKG()
    API["movie"] = MovieKG()
    API["finance"] = FinanceKG()
    API["music"] = MusicKG()
    API["sports"] = {"soccer": SoccerKG(), "nba": NBAKG()}
    yield
    # shutdown


app = FastAPI(
    swagger_ui_parameters={"tryItOutEnabled": True},
    title="CRAG API",
    description="API for Meta KDD Cup '24 CRAG: Comprehensive RAG Benchmark",
    lifespan=lifespan
)



@app.get("/")
async def root():
    return {"message": "The CRAG API service is running"}


class QueryOpenName(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "florida",
                }
            ]
        }
    }
@app.post("/open/search_entity_by_name", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def open_search_entity_by_name(query: QueryOpenName):
    """
    Get a list of entities that best match the query. It returns at most 10 entities at a time.

    Args:

    - query (str): the query 

    Returns:

    - A list of entities (List[str])
    
    """
    result = API["open"].search_entity_by_name(query.query)
    return {"result": result}


class QueryOpenEntity(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Florida City, Florida",
                }
            ]
        }
    }
@app.post("/open/get_entity", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def open_get_entity(entity: QueryOpenEntity):
    """
    Get the details of the entity.

    Args:
    
    - query (str): the entity 

    Returns:

    - None if the entity is not found. Otherwise, there are three fields in the returned Dict:
        * "summary_text": It is in a similar form as the lead section of an article from Wikipedia without structured information.
        * "summary_structured": It is a Dict[str, str], whose function is similar to the infobox in a Wikipedia article, but in a key-value Dict parsed by mwparserfromhell.
        * "raw_mediawiki": It is in the same format as the source code of a whole article from Wikpedia.
            
    """
    result = API["open"].get_entity(entity.query)
    return {"result": result}


class QueryMoviePerson(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Jackson",
                }
            ]
        }
    }
@app.post("/movie/get_person_info", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def movie_get_person_info(person_name: QueryMoviePerson):
    """
    Gets person info in database through BM25.

    Args:

    - query (str): person name to be searched

    Returns:

    - list of top n matching entities (List[Dict[str, Any]]). Entities are ranked by BM25 score. The returned entities MAY contain the following fields:
        - name (string): name of person
        - id (int): unique id of person
        - acted_movies (list[int]): list of movie ids in which person acted
        - directed_movies (list[int]): list of movie ids in which person directed
        - birthday (string): string of person's birthday, in the format of "YYYY-MM-DD"
        - oscar_awards: list of oscar awards (dict), win or nominated, in which the person was the entity. The format for oscar award entity are:
            - 'year_ceremony' (int): year of the oscar ceremony,
            - 'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            - 'category' (string): category of this oscar award,
            - 'name' (string): name of the nominee,
            - 'film' (string): name of the film,
            - 'winner' (bool): whether the person won the award
        
    """
    result = API["movie"].get_person_info(person_name.query)
    return {"result": result}


class QueryMovieMovie(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Lord of the Rings",
                }
            ]
        }
    }
@app.post("/movie/get_movie_info", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def movie_get_movie_info(person_name: QueryMovieMovie):
    """
    Gets movie info in database through BM25.

    Args:

    - query (str): movie name to be searched

    Returns:

    - list of top n matching entities (List[Dict[str, Any]]). Entities are ranked by BM25 score. The returned entities MAY contain the following fields:
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
            - 'year_ceremony' (int): year of the oscar ceremony,
            - 'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            - 'category' (string): category of this oscar award,
            - 'name' (string): name of the nominee,
            - 'film' (string): name of the film,
            - 'winner' (bool): whether the person won the award
        - cast (list [dict]): list of cast members of the movie and their roles. The format of the cast member entity is:
            - 'name' (string): name of the cast member,
            - 'id' (int): unique id of the cast member,
            - 'character' (string): character played by the cast member in the movie,
            - 'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
            - 'order' (int): order of the cast member in the movie. For example, the actress with the lowest order is the main actress,
        - crew' (list [dict]): list of crew members of the movie and their roles. The format of the crew member entity is:
            - 'name' (string): name of the crew member,
            - 'id' (int): unique id of the crew member,
            - 'job' (string): job of the crew member,

    """
    result = API["movie"].get_movie_info(person_name.query)
    return {"result": result}


class QueryMovieYear(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "1992",
                }
            ]
        }
    }
@app.post("/movie/get_year_info", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def movie_get_year_info(year: QueryMovieYear):
    """
    Gets info of a specific year

    Args:

    - query (str): string of year. Note that we only support years between 1990 and 2021

    Returns:

    - An entity representing year information (Dict[str, Any]). The returned entity MAY contain the following fields:
        - movie_list: list of movie ids in the year. This field can be very long to a few thousand films
        - oscar_awards: list of oscar awards (dict), held in that particular year. The format for oscar award entity are:
            - 'year_ceremony' (int): year of the oscar ceremony,
            - 'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
            - 'category' (string): category of this oscar award,
            - 'name' (string): name of the nominee,
            - 'film' (string): name of the film,
            - 'winner' (bool): whether the person won the award

    """
    result = API["movie"].get_year_info(year.query)
    return {"result": result}


class QueryMovieID(BaseModel):
    query: int
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": 100,
                }
            ]
        }
    }
@app.post("/movie/get_movie_info_by_id", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def movie_get_movie_info_by_id(movie_id: QueryMovieID):
    """
    Helper fast lookup function to get movie info directly by id

    Args:

    - query (int): unique id of movie

    Returns:

    - A movie entity (Dict[str, Any]) with same format as the entity in get_movie_info.
    
    """
    result = API["movie"].get_movie_info_by_id(movie_id.query)
    return {"result": result}

@app.post("/movie/get_person_info_by_id", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def movie_get_person_info_by_id(person_id: QueryMovieID):
    """
    Helper fast lookup function to get person info directly by id

    Args:

    - query (int): unique id of person

    Returns:

    -  A person entity (Dict[str, Any]) with same format as the entity in get_person_info.
    
    """
    result = API["movie"].get_person_info_by_id(person_id.query)
    return {"result": result}


class QueryFinanceName(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Apple",
                }
            ]
        }
    }
@app.post("/finance/get_company_name", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_company_name(query: QueryFinanceName):
    """
    Given a query, return top matched company names.

    Args:

    - query (str): the query

    Returns:

    - Top matched company names (list[str]).
    
    """
    result = API["finance"].get_company_name(query.query)
    return {"result": result}


class QueryFinanceCompanyName(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Inflection Point Acquisition Corp. II Unit",
                }
            ]
        }
    }
@app.post("/finance/get_ticker_by_name", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_ticker_by_name(company_name: QueryFinanceCompanyName):
    """
    Return ticker name by company name.

    Args:
    
    - query (str): the company name
        
    Returns:

    - The ticker name of the company (str).
    
    """
    result = API["finance"].get_ticker_by_name(company_name.query)
    return {"result": result}

class QueryFinanceTicker(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "META",
                }
            ]
        }
    }
@app.post("/finance/get_price_history", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_price_history(ticker_name: QueryFinanceTicker):
    """
    Return 1 year history of daily Open price, Close price, High price, Low price and trading Volume.

    Args: 

    - query (str): ticker_name
    
    Returns:

    - 1 year daily price history whose format follows the below example:
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
    result = API["finance"].get_price_history(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_detailed_price_history", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_detailed_price_history(ticker_name: QueryFinanceTicker):
    """
    Return the past 5 days' history of 1 minute Open price, Close price, High price, Low price and trading Volume, starting from 09:30:00 EST to 15:59:00 EST. Note that the Open, Close, High, Low, Volume are the data for the 1 min duration. However, the Open at 9:30:00 EST may not be equal to the daily Open price, and Close at 15:59:00 EST may not be equal to the daily Close price, due to handling of the paper trade. The sum of the 1 minute Volume may not be equal to the daily Volume.

    Args: 

    - query (str): ticker_name
    
    Returns:
    
    - Past 5 days' 1 min price history whose format follows the below example:
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
    result = API["finance"].get_detailed_price_history(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_dividends_history", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_dividends_history(ticker_name: QueryFinanceTicker):
    """
    Return dividend history of a ticker.

    Args: 

    - query (str): ticker_name
    
    Returns:

    - Dividend distribution history whose format follows the below example:
        {'2019-12-19 00:00:00 EST': 0.058,
         '2020-03-19 00:00:00 EST': 0.2,
         '2020-06-12 00:00:00 EST': 0.2,
         ...
         }
         
    """
    result = API["finance"].get_dividends_history(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_market_capitalization", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_market_capitalization(ticker_name: QueryFinanceTicker):
    """
    Return the market capitalization of a ticker.

    Args: 

    - query (str): ticker_name

    Returns:

    - Market capitalization (float)
    
    """
    result = API["finance"].get_market_capitalization(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_eps", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_eps(ticker_name: QueryFinanceTicker):
    """
    Return earnings per share of a ticker.

    Args: 

    - query (str): ticker_name
        
    Returns:
    
    - Earnings per share (float)
    
    """
    result = API["finance"].get_eps(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_pe_ratio", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_pe_ratio(ticker_name: QueryFinanceTicker):
    """
    Return price-to-earnings ratio of a ticker.

    Args: 

    - query (str): ticker_name

    Returns:

    - Price-to-earnings ratio (float)
    
    """
    result = API["finance"].get_pe_ratio(ticker_name.query)
    return {"result": result}

@app.post("/finance/get_info", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def finance_get_info(ticker_name: QueryFinanceTicker):
    """
    Return meta data of a ticker.

    Args: 

    - query (str): ticker_name:

    Returns:

    - Meta information
    
    """
    result = API["finance"].get_info(ticker_name.query)
    return {"result": result}


class QueryMusicArtistName(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "lady gaga",
                }
            ]
        }
    }
@app.post("/music/search_artist_entity_by_name", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_search_artist_entity_by_name(query: QueryMusicArtistName):
    """
    Return the fuzzy matching results of the query (artist name).

    Args:

    - query (str): artist name

    Returns:

    - Top-10 similar entity name in a list
    
    """
    result = API["music"].search_artist_entity_by_name(query.query)
    return {"result": result}


class QueryMusicSongName(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I can feel IT",
                }
            ]
        }
    }
@app.post("/music/search_song_entity_by_name", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_search_song_entity_by_name(query: QueryMusicSongName):
    """
    Return the fuzzy matching results of the query (song name).

    Args:

    - query (str): song name

    Returns:

    - Top-10 similar entity name in a list
    
    """
    result = API["music"].search_song_entity_by_name(query.query)
    return {"result": result}

class QueryMusicRank(BaseModel):
    rank: int
    date: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rank": 1,
                    "date": '2024-02-28'
                }
            ]
        }
    }
@app.post("/music/get_billboard_rank_date", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_billboard_rank_date(query: QueryMusicRank):
    """
    Return the song name(s) and the artist name(s) of a certain rank on a certain date; If no date is given, return the list of of a certain rank of all dates.

    Args:

    - rank (int): the interested rank in billboard; from 1 to 100.
    - date (Optional, str, in YYYY-MM-DD format): the interested date; leave it blank if do not want to specify the date.
    
    Returns:

    - rank_list (list): a list of song names of a certain rank (on a certain date).
    - artist_list (list): a list of author names corresponding to the song names returned.
    
    """
    result = API["music"].get_billboard_rank_date(query.rank, query.date)
    return {"result": result}


class QueryMusicAttribute(BaseModel):
    date: str
    attribute: str
    song_name: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": '2024-02-28',
                    "attribute": 'weeks_in_chart',
                    "song_name": 'Texas Hold \'Em'
                }
            ]
        }
    }
@app.post("/music/get_billboard_attributes", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_billboard_attributes(query: QueryMusicAttribute):
    """
    Return the attributes of a certain song on a certain date
        
    Args:

    - date (str, in YYYY-MM-DD format): the interested date of the song
    - attribute (str): attributes from ['rank_last_week', 'weeks_in_chart', 'top_position', 'rank']
    - song_name (str): the interested song name
    
    Returns:

    - the value (str) of the interested attribute of a song on a certain date
    
    """
    result = API["music"].get_billboard_attributes(query.date, query.attribute, query.song_name)
    return {"result": result}


class QueryMusicYear(BaseModel):
    query: int
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": 1998,
                }
            ]
        }
    }
@app.post("/music/grammy_get_best_artist_by_year", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_best_artist_by_year(year: QueryMusicYear):
    """
    Return the Best New Artist of a certain year in between 1958 and 2019

    Args:

    - query (int, in YYYY format): the interested year
    
    Returns:

    - the list of artists who win the award

    """
    result = API["music"].grammy_get_best_artist_by_year(year.query)
    return {"result": result}

class QueryMusicArtist(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Adele",
                }
            ]
        }
    }
@app.post("/music/grammy_get_award_count_by_artist", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_award_count_by_artist(artist_name: QueryMusicArtist):
    """
    Return the number of awards won by a certain artist between 1958 and 2019

    Args:

    - query (str): the name of the artist
    
    Returns:

    - the number of total awards (int)
    
    """
    result = API["music"].grammy_get_award_count_by_artist(artist_name.query)
    return {"result": result}


class QueryMusicSong(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I Can Feel It",
                }
            ]
        }
    }
@app.post("/music/grammy_get_award_count_by_song", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_award_count_by_song(song_name: QueryMusicSong):
    """
    Return the number of awards won by a certain artist between 1958 and 2019

    Args:

    - query (str): the name of the song
    
    Returns:

    - the number of total awards (int)
    """
    result = API["music"].grammy_get_award_count_by_song(song_name.query)
    return {"result": result}


@app.post("/music/grammy_get_best_song_by_year", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_best_song_by_year(year: QueryMusicYear):
    """
    Return the Song Of The Year in a certain year between 1958 and 2019
        
    Args:

    - query (int, in YYYY format): the interested year
    
    Returns:

    - the list of the song names that win the Song Of The Year in a certain year
        
    """
    result = API["music"].grammy_get_best_song_by_year(year.query)
    return {"result": result}

@app.post("/music/grammy_get_award_date_by_artist", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_award_date_by_artist(artist_name: QueryMusicArtist):
    """
    Return the award winning years of a certain artist

    Args:

    - query (str): the name of the artist

    Returns:

    - the list of years the artist is awarded

    """
    result = API["music"].grammy_get_award_date_by_artist(artist_name.query)
    return {"result": result}

@app.post("/music/grammy_get_best_album_by_year", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_best_album_by_year(year: QueryMusicYear):
    """
    Return the Album Of The Year of a certain year between 1958 and 2019

    Args:

    - query (int, in YYYY format): the interested year
    
    Returns:

    - the list of albums that won the Album Of The Year in a certain year

    """
    result = API["music"].grammy_get_best_album_by_year(year.query)
    return {"result": result}

@app.post("/music/grammy_get_all_awarded_artists", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_grammy_get_all_awarded_artists():
    """
    Return all the artists ever awarded Grammy Best New Artist between 1958 and 2019
    
    Returns:

    - the list of artist ever awarded Grammy Best New Artist (list)
    
    """
    result = API["music"].grammy_get_all_awarded_artists()
    return {"result": result}

@app.post("/music/get_artist_birth_place", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_artist_birth_place(artist_name: QueryMusicArtist):
    """
    Return the birth place country code (2-digit) for the input artist

    Args:

    - query (str): the name of the artist
    
    Returns:

    - the two-digit country code following ISO-3166 (str)
    
    """
    result = API["music"].get_artist_birth_place(artist_name.query)
    return {"result": result}

@app.post("/music/get_artist_birth_date", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_artist_birth_date(artist_name: QueryMusicArtist):
    """
    Return the birth date of the artist

    Args:

    - query (str): the name of the artist
    
    Returns:

    - life_span_begin (str, in YYYY-MM-DD format if possible): the birth date of the person or the begin date of a band
    
    """
    result = API["music"].get_artist_birth_date(artist_name.query)
    return {"result": result}


class QueryMusicBand(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Maroon 5",
                }
            ]
        }
    }
@app.post("/music/get_members", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_members(band_name: QueryMusicBand):
    """
    Return the member list of a band

    Args:

    - query (str): the name of the band
    
    Returns:

    - the list of members' names.
    
    """
    result = API["music"].get_members(band_name.query)
    return {"result": result}

@app.post("/music/get_lifespan", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_lifespan(artist_name: QueryMusicArtist):
    """
    Return the lifespan of the artist

    Args:

    - query (str): the name of the artist
    
    Returns:

    - the birth and death dates in a list
    
    """
    result = API["music"].get_lifespan(artist_name.query)
    return {"result": result}

@app.post("/music/get_song_author", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_song_author(song_name: QueryMusicSong):
    """
    Return the author of the song

    Args:

    - query (str): the name of the song
    
    Returns:

    - the author of the song (str)
    
    """
    result = API["music"].get_song_author(song_name.query)
    return {"result": result}

@app.post("/music/get_song_release_country", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_song_release_country(song_name: QueryMusicSong):
    """
    Return the release country of the song

    Args:

    - query (str): the name of the song
    
    Returns:

    - the two-digit country code following ISO-3166 (str)
    
    """
    result = API["music"].get_song_release_country(song_name.query)
    return {"result": result}

@app.post("/music/get_song_release_date", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_song_release_date(song_name: QueryMusicSong):
    """
    Return the release date of the song

    Args:

    - query (str): the name of the song
    
    Returns:

    - the date of the song (str in YYYY-MM-DD format)
    """
    result = API["music"].get_song_release_date(song_name.query)
    return {"result": result}

@app.post("/music/get_artist_all_works", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def music_get_artist_all_works(artist_name: QueryMusicArtist):
    """
    Return the list of all works of a certain artist

    Args:

    - query (str): the name of the artist
    
    Returns:

    - the list of all work names
    
    """
    result = API["music"].get_artist_all_works(artist_name.query)
    return {"result": result}

class QuerySportsSoccer(BaseModel):
    date: str
    team_name: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": "2024-03-09",
                    "team_name": "Everton"
                }
            ]
        }
    }
@app.post("/sports/soccer/get_games_on_date", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def sports_soccer_get_games_on_date(query: QuerySportsSoccer):
    """
    Get soccer games given date

    Args:

    - date (str, in YYYY-MM-DD/YYYY-MM/YYYY format): e.g., 2024-03-01, 2024-03, 2024
    - team_name (Optional, str)

    Returns:

    - info of the games, such as 
        - venue: whether the team is home or away in game
        - result: win lose result of the game
        - GF: goals of the team in game
        - opponent: opponent of the team
        - Captain: Captain of the team
    """
    try:
        result = json.loads(API["sports"]["soccer"].get_games_on_date(query.date, query.team_name))
    except:
        result = None
    return {"result": result}


class QuerySportsNBA(BaseModel):
    date: str
    team_name: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": "2022-10-11",
                    "team_name": "Chicago Bulls"
                }
            ]
        }
    }
@app.post("/sports/nba/get_games_on_date", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def sports_nba_get_games_on_date(query: QuerySportsNBA):
    """
    Get all nba game rows given date_str

    Args: 

    - date (str, in YYYY-MM-DD/YYYY-MM/YYYY format): the time of the games, e.g. 2023-01-01, 2023-01, 2023
    - team_name (Optional, str):  basketball team name, like Los Angeles Lakers

    Returns:

    - info of the games found, such as
        - game_id: id of the game
        - team_name_home: home team name
        - team_name_away: away team name
        - wl_home: win lose result of home team
        - wl_away: win lose result of away team
        - pts_home: home team points in the game
        - pts_away: away team points in the game
    """
    try:
        result = json.loads(API["sports"]["nba"].get_games_on_date(query.date, query.team_name))
    except:
        result = None
    return {"result": result}

class QuerySportsNBAGameIds(BaseModel):
    game_ids: List[str]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "game_ids": ["0022200547", "0029600027"],
                }
            ]
        }
    }
@app.post("/sports/nba/get_play_by_play_data_by_game_ids", responses={
    200: {
        "content": {
            "application/json": {
                "example": "[[Click 'Execute' to get the response]]"
            }
        }
    }
})
async def sports_nba_get_play_by_play_data_by_game_ids(query: QuerySportsNBAGameIds):
    """
    Get all nba play by play rows given game ids
    
    Args: 

    - game_ids (List[str]):  nba game ids, e.g., ["0022200547", "0029600027"]

    Returns:

    - info of the play by play events of given game id, such as
        - game_id: A unique identifier for each game.
        - eventmsgtype: A code representing the type of event.
        - eventmsgactiontype: A code representing the action type of event message.
        - period: The period of the game in which the event occurred.
        - wctimestring: Wall clock time when the event occurred.
        - pctimestring: Game clock time when the event occurred.
        - homedescription: A description of the event from the perspective of the home team.
        - neutraldescription: A neutral description of the event.
        - visitordescription: A description of the event from the perspective of the visiting team.
        - player1_id: A unique identifier for the first player involved in the event.
        - player1_name: The name of the first player involved in the event.
    """
    try:
        result = json.loads(API["sports"]["nba"].get_play_by_play_data_by_game_ids(query.game_ids))
    except:
        result = None
    return {"result": result}