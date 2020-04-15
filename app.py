from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import math
app = Flask(__name__)
model = pickle.load(open('taxi.pk1', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    listof_Values = [int(x) for x in request.form.values()]
    arrayof_Values = [np.array(listof_Values)]
    predictions = model.predict(arrayof_Values)

    return render_template('index.html', predictions=math.floor(predictions[0]))


if __name__ == '__main__':
    app.run(host='0,0,0,0',post=8080)
