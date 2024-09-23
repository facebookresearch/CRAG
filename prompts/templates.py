# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0."""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction is not sure about the answer."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"score": 0, "explanation": "The prediction, 151 m, does not match the ground truth, 279 m."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the grount truth."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"score": 0, "explanation": "The prediction does not say Amazon River is longer than 4000 miles."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"score": 0, "explanation": "The prediction, 3.3 M, does not match the number, 3.6 M, in ground truth."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."}

Question: "what is xxx's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""
