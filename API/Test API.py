


from flask import Flask, request, jsonify, render_template   # Importing libraries
import pickle
from flask_cors import CORS

app = Flask(__name__)      #creating an instance of the Flask application and __name__ is  the  name of current module




CORS(app) # enabling Cross-Origin Resource Sharing


model = pickle.load(open('final.pkl','rb'))  #Loading the trained ML model from 'final.pkl' file

@app.route('/')


def index():
    return render_template('index.html')


#fetching the inputs and displaying of the predicted  output 

@app.route('/predict', methods=['POST'])  
def predict():


    data=request.form # fetching data which is entered in the form

    # segregating  the inputs and  then typecasting  it since they are in string and model takes integers
    
    p1 = int(data['age'])
    p2 = int(data['annual_income'])
    p3 = int(data['spending_score'])

    if p3 > 100:                                           # checking that spending score should not greater than 100 and if it is return error
        return "Error: Spending score should not be greater than 100"

    #  Making a prediction  of which cluster does the particular customer belongs using the loaded model

    result = model.predict([[p1, p2, p3]]) 


#Mapping the predicted cluster into thier respective names and labels

    category = {              # this dictonary maps the cluster number to descriptive group name
        0: "Sensible group",
        1: "Target group",
        2: "Careful group",
        3: "Careless group",
        4: "Standard group"
    }
   #labels dictionary maps the cluster number to a detailed label that describes the income and spending behavior of that cluster

    labels = {                     
        0: " Low-income, low spenders; prioritize savings over excessive spending.",
        1: " Middle-to-high-income, high spenders; ideal target for the mall.",
        2: " High-income, low spenders; cautious with their expenses.",
        3: " Low-income, high spenders; potential credit risk, should be approached with caution.",
        4: " Middle-income, moderate spenders; reliable and stable customer base."
    }    
         # Create a dictionary that contains information about the predicted cluster based on the machine learning model's output.
    prediction = {                   
        'cluster': int(result),
        'label': labels[int(result)],
        'category' : category[int(result)]
    }
   
    return render_template('prediction.html', prediction=prediction)   # Rendering the 'prediction.html' template with the prediction data
    


if __name__ == '__main__':              # Running the Flask application in debug mode on port 9000
     app.run(debug=True, port=9000)
    









