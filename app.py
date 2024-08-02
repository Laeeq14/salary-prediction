import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('salary_model.pkl')

# Load the dataset
data = pd.read_csv('Salary_Data.csv')

# Extract unique values
job_titles = data['Job Title'].unique().tolist()
education_levels = data['Education Level'].unique().tolist()
genders = data['Gender'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', job_titles=job_titles, education_levels=education_levels, genders=genders)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    years_of_experience = float(data['Years_Of_Experience'])
    job_title = data['Job_Title']
    education_level = data['Education_Level']
    age = float(data['Age'])
    gender = data['Gender']
    
    features_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])
    
    # Predict using the model
    prediction = model.predict(features_df)[0]
    
    return jsonify({'predicted_salary': prediction})

@app.route('/categories', methods=['GET'])
def categories():
    return jsonify({
        'Job_Titles': job_titles,
        'Education_Levels': education_levels
    })

if __name__ == '__main__':
    app.run(debug=True)
