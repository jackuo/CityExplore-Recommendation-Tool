"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
from sklearn.externals import joblib
import re

# Load the models 
# model_dict is the collection of extra tree models 

# This line doesn't work, joblib only loads locally. File is too big to upload to heroku though
# model_dict = joblib.load('https://drive.google.com/open?id=1h20N5Cooti2e5CDkmKY5LOzRuLksyR5e')
# model_dict = joblib.load('./static/models/models_compressed.p')
# word_vectorizer = joblib.load('static/models/word_vectorizer.p')

model_dict = joblib.load('./static/models/log_models.p')
word_vectorizer = joblib.load('static/models/log_word_vectorizer.p')

cl_path = 'static/cleaning/clean_letters.txt'

clean_word_dict = {}
with open(cl_path, 'r', encoding='utf-8') as cl:
    for line in cl:
        line = line.strip('\n')
        typo, correct = line.split(',')
        clean_word_dict[typo] = correct

def clean_word(text):
    # Removes different characters, symbols, numbers, some stop words
    replace_numbers = re.compile(r'\d+', re.IGNORECASE)
    special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)

    for typo, correct in clean_word_dict.items():
        text = re.sub(typo, " " + correct + " ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i’m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_numbers.sub('', text)
    return text



def raw_chat_to_model_input(raw_input_string):
    # Converts string into cleaned text
    cleaned_text = []
    for text in [raw_input_string]:
        cleaned_text.append(clean_word(text))
    return word_vectorizer.transform(cleaned_text)

def predict_toxicity(raw_input_string):
    ''' Given any input string, predict the toxicity levels'''
    model_input = raw_chat_to_model_input(raw_input_string)
    results = []
    for key,model in model_dict.items():
        results.append(round(model.predict_proba(model_input)[0,1],3))
    return results

def make_prediction(input_chat):
    """
    Given string to classify, returns the input argument and the dictionary of 
    model classifications in a dict so that it may be passed back to the HTML page.

    Input:
    Raw string input

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    if not input_chat:
        input_chat = ' '
    if len(input_chat) > 500:
        input_chat = input_chat[:500]
    pred_probs = predict_toxicity(input_chat)

    probs = [{'name': list(model_dict.keys())[index], 'prob': pred_probs[index]}
             for index in np.argsort(pred_probs)[::-1]]

    return (input_chat, probs)

# This section checks that the prediction code runs properly
# To test, use "python predictor_api.py" in the terminal.

# if __name__='__main__' section only runs
# when running this file; it doesn't run when importing





## Later, bring in Star Rating, Number of reviews, distance
from numpy import dot 
from numpy.linalg import norm
import pandas as pd
import numpy as nump
import pickle
import copy
import folium
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

def unpickle_file(filename):
    with open(filename, 'rb') as picklefile:
        return(pickle.load(picklefile))

# Import details file
df_urls = unpickle_file('/final_models/all_listing_details.pkl')
details_filename = '/final_models/final_details_dataframe.pkl'
df_details = unpickle_file(details_filename)
# keep only listings that have 5 reviews or more
df_details['number_reviews'] = pd.to_numeric(df_details['number_reviews'])
df_details_filter = df_details[df_details['number_reviews'] > 4]
df_details_filter = pd.merge(df_details_filter, df_urls[['name', 'main_url']], on = 'name', how = 'left')
# keep only the columns we need
df_details_filter = df_details_filter[['name', 'star_rating', 'main_url', 'neighborhood', 'borough', 'latitude', 'longitude']]

# Import H Files
# Activity
H_activity_filename = '/final_models/activity/activity_H_dataframe.pkl'
H_activity = unpickle_file(H_activity_filename).reset_index().rename(columns = {'index': 'name'})

# Food
H_food_filename = '/final_models/food/food_H_dataframe.pkl'
H_food = unpickle_file(H_food_filename).reset_index().rename(columns = {'index': 'name'})

# Other
H_drinks_filename = '/final_models/drinks/drinks_H_dataframe.pkl'
H_drinks = unpickle_file(H_drinks_filename).reset_index().rename(columns = {'index': 'name'})



def unpickle_file(filename):
    with open(filename, 'rb') as picklefile:
        return(pickle.load(picklefile))

def input_to_user_vector(input_list, rec_type):
    
    category_dict = {'Activity': 14, 'Meal': 14, 'Second Meal': 11}
    # make list the len of the number of categories in each rec type
    user_vector = [0]*category_dict[rec_type] 
    # convert every value from string to integer
    input_list = [int(index) for index in input_list]
    # use input_list to create a user vector
    for i in input_list:
        user_vector[i] = 1
    
    return(user_vector)

def cosine_similarity(H, user_vector): #find cosine similarity of these listings
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    # Create Cosine Similarity Column
    #H['cosine_sim'] = pd.Series([cosine(user_vector, val) for index, val in H.iloc[:, 1:].iterrows()])
    df = copy.deepcopy(H)
    df['cosine_sim'] = pd.Series([cosine(user_vector, val) for index, val in df.iloc[:, 1:].iterrows()]) 
    return(df)

def details_dataframe(H):
    # merge with details dataframe
    H_details = pd.merge(H, df_details_filter, on = 'name', how = 'left').drop_duplicates().dropna()
    # scale the star rating values
    H_details['scaled_star_rating'] = pd.to_numeric(H_details['star_rating'])/5 #scale star rating
    # generate recommendation score
    H_details['recommendation_score'] = H_details['cosine_sim'] + 0.5*H_details['scaled_star_rating']
    return(H_details)

def neighborhood_generator(H, neighborhood):
    return(H[H['neighborhood'] == neighborhood])
    
def recommendation_details(H, num_recs, rec_type):
    H_details = H.sort_values(by = 'recommendation_score', ascending = False)[:num_recs]
    H_details = H_details[['name', 'star_rating', 'neighborhood', 'borough', 'latitude', 'longitude', 'recommendation_score']]
    H_details['Type'] = rec_type
    return(H_details)

def make_recommendations(H, user_vector, neighborhood, num_recs, rec_type):
    H_cosine = cosine_similarity(H, user_vector)
    H_details = details_dataframe(H_cosine)
    H_neigh = neighborhood_generator(H_details, neighborhood)
    H_recs = recommendation_details(H_neigh, num_recs, rec_type)
    return(H_recs)




if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    chat_in = 'bob'
    pprint(chat_in)

    x_input, probs = make_prediction(chat_in)
    print(f'Input values: {x_input}')
    print('Output probabilities')
    pprint(probs)
