# -*- coding: utf-8 -*-
"""
First attempt using the NRC Emotional Lexicon suggested from Dr. Villanes.  The final result is a dataframe containing the total emotional weights for each character.
Still unsure on the exact interpretation, but all rows add to 1 so I believe it could be some kind of percentage.

"""

from nrclex import NRCLex
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd

script = pd.read_csv("script.csv")  #Read script csv
script = script.drop(columns=['Unnamed: 0'])    #Removes the unnecessary 'Unnamed: 0' column
#print(script['character'].unique())
#print(script['character'].value_counts())

##Creates separate dataframes for each character's dialogue
joyce = script[script['character'] == 'Joyce']
sandra = script[script['character'] == 'Sandra']
felipe = script[script['character'] == 'Felipe']
diane = script[script['character'] == 'Diane']
angelo = script[script['character'] == 'Angelo']
riley = script[script['character'].isin(['YOUNG RILEY', 'RILEY VOICE', 'RILEY'])]   #Could be interesting to see the difference between young Riley and Riley.

##Function used to convert the dataframe name to a string
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

##For each character, pass all of their dialogue through NRCLex and get back an 'affect frequency' for each word.  Then sums up the weighted frequencies of all words.
characters = ['Joyce', 'Sandra', 'Felipe', 'Diane', 'Angelo', 'Riley']
overall = []
for character in (joyce, sandra, felipe, diane, angelo, riley):
    data = []
    for line in character['quote']:
        words = word_tokenize(line)     #Breaks the string into a list of its words
        for term in words:
            emotion = NRCLex(term)
            if len(emotion.affect_list) > 0:    #If the selected word has no emotion association, then the word will be ignored
                temp = emotion.affect_frequencies
                temp['term'] = term     #Adds the selected word with its affect frequency
                data.append(temp)
    df_nrc = pd.DataFrame.from_dict(data)
    df_nrc = df_nrc.fillna(0)  
    temp2 = {}  #Temporary dictionary to add to the final dataframe.  Definitely could be a more efficient way to do this.
    temp2['character'] = get_df_name(character)     #Character's name
    temp2['Anger'] = sum(df_nrc['anger']) / len(df_nrc)     #The sum of all anger-weighted words for a character and divide by the number of quotes since characters have different numbers of lines.  Another possibility could be to divide by total words spoken for each character.
    temp2['Disgust'] = sum(df_nrc['disgust']) / len(df_nrc)
    temp2['Fear'] = sum(df_nrc['fear']) / len(df_nrc)
    temp2['Joy'] = sum(df_nrc['joy']) / len(df_nrc)
    temp2['Sadness'] = sum(df_nrc['sadness']) / len(df_nrc)
    temp2['Positive'] = sum(df_nrc['positive']) / len(df_nrc)
    temp2['Negative'] = sum(df_nrc['negative']) / len(df_nrc)
    temp2['Trust'] = sum(df_nrc['trust']) / len(df_nrc)
    temp2['Surprise'] = sum(df_nrc['surprise']) / len(df_nrc)
    temp2['Anticipation'] = (sum(df_nrc['anticipation']) + sum(df_nrc['anticip'])) / len(df_nrc)    #For some reason NRCLex created an 'anticip' value for some words
    overall.append(temp2)
df = pd.DataFrame.from_dict(overall)
print(df.head(6))   #Final dataframe has the emotional breakdown for each character based on the 10 possible values.  All rows should add to 1.


##Example of how to use NRCLex.  For more options, go to https://pypi.org/project/NRCLex/
# emotion = NRCLex('horrible')
# print(emotion.affect_frequencies)

##Example using the NLTK word tokenizer.  This takes a string and creates a list of comma-separated words.
# text = diane['quote'].str.cat(sep=' ')
# words = word_tokenize(text)
# emotion = NRCLex(words)
# print(emotion.top_emotions)

##Example of how to use the NLTK stop words parser.  Removes irrevelant words from a string based on NLTK's dictionary. Could be useful to include; NRCLex somewhat does this.
# stop_words = set(stopwords.words('english'))
# term_vec = []
# for i in riley['quote']:
#     t = word_tokenize(i)
#     term_list = []
#     for term in t:
#         if term not in stop_words:
#             term_list.append(term)
#     term_vec.append(term_list)
# for vec in term_vec:
#     print(vec)