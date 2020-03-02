import flask
from flask import request
import copy
from numpy import dot 
from numpy.linalg import norm
import pandas as pd
import numpy as nump
import pickle
import folium
from flask import jsonify

from predictor_api import make_prediction, input_to_user_vector, cosine_similarity
from predictor_api import details_dataframe, neighborhood_generator, recommendation_details, make_recommendations

def unpickle_file(filename):
    with open(filename, 'rb') as picklefile:
        return(pickle.load(picklefile))


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


# Initialize the app

app = flask.Flask(__name__)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!


@app.route("/", methods=["POST"])
def print_piped():
    if request.form.getlist('Activity_Checklist'):
        input_vector = request.form.getlist('Activity_Checklist')
        print(msg)
        x_input, predictions = make_prediction(str(msg))
        flask.render_template('predictor.html',
                                chat_in=mycheckbox,
                                prediction=predictions)
    return jsonify(predictions)

@app.route("/", methods=["GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    print("Activity: ", request.args.getlist('Activity_Checklist'))
    print("Meal: ", request.args.getlist('Meal_Checklist'))
    print("Second Meal: ", request.args.getlist('Second_Meal_Checklist'))
    print("REQUEST ARGS: ", request.args.get('neighborhood_list'))
    args_dict = request.args.to_dict()
    
    if(request.args):
        # get the indicies from the checkboxes
        activity_vector = request.args.getlist('Activity_Checklist')
        meal_vector = request.args.getlist('Meal_Checklist')
        second_meal_vector = request.args.getlist('Second_Meal_Checklist')
        location = args_dict['neighborhood_list']

        # generate vectors to feed into recommendation
        activity_model_vector = input_to_user_vector(activity_vector, 'Activity')
        meal_model_vector = input_to_user_vector(meal_vector, 'Meal')
        second_meal_model_vector = input_to_user_vector(second_meal_vector, 'Second Meal')

        df_all_recs = pd.DataFrame()
        df_all_recs = df_all_recs.append(make_recommendations(H_activity, activity_model_vector, location, 3, 'Activity'))
        df_all_recs = df_all_recs.append(make_recommendations(H_food, meal_model_vector, location, 3, 'Meal'))
        df_all_recs = df_all_recs.append(make_recommendations(H_drinks, second_meal_model_vector, location, 3, 'Second Meal'))

        #create folium map
        colors = {'Activity' : 'red', 'Meal': 'blue', 'Second Meal': 'green'}
        map_osm = folium.Map(location=[df_all_recs['latitude'].mean(), df_all_recs['longitude'].mean()], zoom_start=13, tiles='CartoDB positron')
        (df_all_recs.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], 
                                                    radius=10, fill_color=colors[row['Type']],
                                                    popup = folium.Popup(row['name'], row['star_rating'])).add_to(map_osm), axis=1))  

        map_osm.save('templates/map.html')
        #foilum_map = map_osm._repr_html_()

        #for the location filter
        neighborhood_list = ['High Bridge and Morrisania', 'Bronx Park and Fordham', 'Southeast Bronx', 'Central Bronx',
        'Kingsbridge and Riverdale', 'Hunts Point and Mott Haven', 'Northeast Bronx', 'Sunset Park', 'Greenpoint',
        'Northwest Brooklyn', 'East New York and New Lots', 'Central Brooklyn', 'Bushwick and Williamsburg', 'Flatbush',
        'Southwest Brooklyn', 'East Harlem', 'Greenwich Village and Soho', 'Lower Manhattan', 'Lower East Side',
        'Chelsea and Clinton', 'Central Harlem', 'Upper West Side', 'Inwood and Washington Heights', 'Upper East Side',
        'Gramercy Park and Murray Hill', 'Central Queens', 'West Central Queens', 'Rockaways', 'Southwest Queens',
        'Northwest Queens', 'West Queens', 'Northeast Queens', 'Jamaica', 'North Queens',
        'South Shore', 'Stapleton and St. George', 'Mid-Island', 'Port Richmond']

        return flask.render_template('predictor.html', neighborhood_list=neighborhood_list)
        
    else: 
        #For first load, request.args will be an empty ImmutableDict type. If this is the case,
        # we need to pass an empty string into make_prediction function so no errors are thrown.
        x_input, predictions = make_prediction('')

        #for the location filter
        neighborhood_list = ['High Bridge and Morrisania', 'Bronx Park and Fordham', 'Southeast Bronx', 'Central Bronx',
        'Kingsbridge and Riverdale', 'Hunts Point and Mott Haven', 'Northeast Bronx', 'Sunset Park', 'Greenpoint',
        'Northwest Brooklyn', 'East New York and New Lots', 'Central Brooklyn', 'Bushwick and Williamsburg', 'Flatbush',
        'Southwest Brooklyn', 'East Harlem', 'Greenwich Village and Soho', 'Lower Manhattan', 'Lower East Side',
        'Chelsea and Clinton', 'Central Harlem', 'Upper West Side', 'Inwood and Washington Heights', 'Upper East Side',
        'Gramercy Park and Murray Hill', 'Central Queens', 'West Central Queens', 'Rockaways', 'Southwest Queens',
        'Northwest Queens', 'West Queens', 'Northeast Queens', 'Jamaica', 'North Queens',
        'South Shore', 'Stapleton and St. George', 'Mid-Island', 'Port Richmond']

        return flask.render_template('predictor.html',neighborhood_list=neighborhood_list)


@app.route('/map.html')
def main():
        return flask.render_template('map.html')

# Start the server, continuously listen to requests.

if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()

