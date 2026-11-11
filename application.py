from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app=application

'''Creating route for home page'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')

    try:
        reading_score = int(request.form.get('reading_score', 0))
        writing_score = int(request.form.get('writing_score', 0))

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=reading_score,
            writing_score=writing_score,
        )

        features = data.get_data_as_frame()
        pred_pipeline = PredictPipeline()
        
        prediction = pred_pipeline.predict(features=features)[0]

        return render_template(
            'result.html',
            prediction=round(float(prediction), 2),
            gender=data.gender,
            race_ethnicity=data.race_ethnicity,
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=data.reading_score,
            writing_score=data.writing_score,
        )
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0') #debug=True will be ok while deploying
    
    '''we will use elastic beanstalk in AWS to deploy our application . first create .ebextensions folder for it '''