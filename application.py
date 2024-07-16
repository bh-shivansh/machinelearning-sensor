from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            footfall = float(request.form.get('footfall')),
            tempMode = float(request.form.get('tempMode')),
            AQ = float(request.form.get('AQ')),
            USS = float(request.form.get('USS')),
            CS = float(request.form.get('CS')),
            VOC = float(request.form.get('VOC')),
            RP = request.form.get('RP'),
            IP= request.form.get('IP'),
            Temperature = request.form.get('Temperature')
            
        )

        pred_df = data.get_data_as_dataframe()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0],2)
        return render_template('index.html',results=results,pred_df = pred_df)
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            footfall = float(request.json['footfall']),
            tempMode = float(request.json['tempMode']),
            AQ = float(request.json['AQ']),
            USS = float(request.json['USS']),
            CS = float(request.json['CS']),
            VOC = float(request.json['VOC']),
            RP = request.json['RP'],
            IP = request.json['IP'],
            Temperature = request.json['Temperature']
            
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        dct = {'price':round(pred[0],2)}
        return jsonify(dct)

if __name__ == '__main__':
    app.run(debug=True)

    app.run(host='0.0.0.0', port=8000)