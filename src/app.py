'''from flask import Flask, request, render_template
import util

app = Flask(__name__)

@app.route("/")
def home():
    team = util.get_team()
    venue = util.get_venue()
    return render_template("index.html", team = team, venue = venue)

@app.route("/predictedScore", methods = ["POST"])
def predicted_score():
    try:
        venue = request.form["venue"]
        bat_team = request.form["bat_team"]
        bowl_team = request.form["bowl_team"]
        overs = float(request.form["overs"])
        runs = int(request.form["runs"])
        wickets = int(request.form["wickets"])

        result = int(util.predict_score(runs, wickets, overs, bat_team, bowl_team, venue))
        if result == 1:
            return "Wrong inputs have been entered. Please try again"
        else:
            #return "Predicted score range of " + bat_team + " is " + str(result)
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(result))
    except :
        return "Wrong inputs have been entered. Please try again"

if __name__ == "__main__":
    app.run()'''
import util
print(util.get_team())
#exit()
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        venue = request.form["venue"]
        bat_team = request.form["bat_team"]
        bowl_team = request.form["bowl_team"]
        overs = float(request.form["overs"])
        runs = int(request.form["runs"])
        wickets = int(request.form["wickets"])
        #venue = venue.to_lower()
        #print(venue)
        result = int(util.predict_score(runs, wickets, overs, bat_team, bowl_team, venue))
        if result == 1:
            return "Wrong inputs have been entered. Please try again"
        else:
            #return render_template('index.html',prediction_text = "Predicted score of " + bat_team + " is " + str(result))
            return "Predicted score range of " + bat_team + " is " + str(result)

if __name__=="__main__":
    app.run(debug=True)