from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sns

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for visualization
df = pd.read_csv('water_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            population = float(request.form['population'])
            industries = float(request.form['industries'])
            temperature = float(request.form['temperature'])
            irrigation = float(request.form['irrigation'])
            rainfall = float(request.form['rainfall'])
            
            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'Population': [population],
                'Number_of_Industries': [industries],
                'Average_Temperature': [temperature],
                'Irrigation_Area': [irrigation],
                'Rainfall': [rainfall]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Generate analysis charts
            generate_comparison_chart(population, industries, temperature, irrigation, rainfall, prediction)
            
            return render_template('predict.html', 
                                  prediction=round(prediction, 2),
                                  input_data={
                                      'Population': population,
                                      'Number of Industries': industries,
                                      'Average Temperature': temperature,
                                      'Irrigation Area': irrigation,
                                      'Rainfall': rainfall
                                  })
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

def generate_comparison_chart(population, industries, temperature, irrigation, rainfall, prediction):
    # Create a comparison bar chart
    features = ['Population', 'Industries', 'Temperature', 'Irrigation', 'Rainfall']
    
    # Normalize the values for comparison
    avg_values = [df['Population'].mean()/1000000, 
                 df['Number_of_Industries'].mean(), 
                 df['Average_Temperature'].mean(), 
                 df['Irrigation_Area'].mean()/1000, 
                 df['Rainfall'].mean()/100]
    
    input_values = [population/1000000, 
                   industries, 
                   temperature, 
                   irrigation/1000, 
                   rainfall/100]
    
    # Create the comparison chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, avg_values, width, label='Historical Average')
    plt.bar(x + width/2, input_values, width, label='Current Input')
    
    plt.xlabel('Features')
    plt.ylabel('Normalized Values')
    plt.title('Comparison of Input Values with Historical Average')
    plt.xticks(x, features)
    plt.legend()
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('static/images/comparison.png')

if __name__ == '__main__':
    # Ensure the static/images directory exists
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True)