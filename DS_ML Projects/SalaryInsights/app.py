import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    Test on Server
    '''
    exp = request.form.get('Experience' , type=int)
    gender = request.form.get('Gender' , type=int)
    jobtitle = request.form.get('JobTitle' , type=int)
    int_features = [exp, gender, jobtitle]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated Salary (USD): ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
