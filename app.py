from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# PREVENT GUI ERRORS 
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if not os.path.exists('static/images'):
    os.makedirs('static/images')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get data
    rating = float(request.form['rating'])
    distance = float(request.form['distance'])
    weather = float(request.form['weather'])
    traffic = float(request.form['traffic'])
    vehicle = int(request.form['vehicle']) 

    # predict
    features = np.array([[rating, distance, weather, traffic, vehicle]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # logic for suggestion
    suggestion = "No specific action needed."
    result_text = "Low Risk - Supply Chain Stable"
    css_class = "success"

    if prediction[0] == 1:
        result_text = "High Risk of Disruption"
        css_class = "danger"
        if weather == 1:
             suggestion = "⚠️ Weather Alert: Heavy rain detected. Delay shipment or use waterproof packaging."
        elif traffic > 7:
             suggestion = "🚦 Traffic Jam: Route is congested. Reroute driver via alternate highway."
        elif rating < 3:
             suggestion = "⭐ Supplier Risk: Vendor reliability is low. Request 'Proof of Dispatch'."
        else:
             suggestion = "General Alert: Monitor this shipment closely."
    else:
        suggestion = "✅ Green Light: Shipment is on track. Proceed with standard schedule."

    # grapgh
    rating_risk = (5 - rating) * 2
    distance_risk = min((distance / 2000) * 10, 10)
    weather_risk = 9 if weather == 1 else 2
    traffic_risk = traffic

    # Create a DataFrame for Seaborn
    plot_data = pd.DataFrame({
        'Factor': ['Supplier Risk', 'Distance', 'Weather', 'Traffic'],
        'Severity': [rating_risk, distance_risk, weather_risk, traffic_risk]
    })

    # Set the style
    plt.figure(figsize=(8, 4))
    sns.set_theme(style="whitegrid")
    
    # Create the Barplot
    colors = ["#1e3a8a" if x < 6 else "#dc2626" for x in plot_data['Severity']]
    ax = sns.barplot(x='Factor', y='Severity', data=plot_data, palette=colors)
    
    # Add a limit line (Risk Threshold)
    plt.axhline(y=5, color='gray', linestyle='--', label='Risk Threshold')

    # Style tweaks
    plt.title('Real-Time Risk Factor Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 10)
    plt.ylabel('Severity Score (0-10)')
    plt.xlabel('')
    

    timestamp = int(time.time())
    image_path = f'static/images/prediction_plot.png'
    plt.savefig(image_path, bbox_inches='tight', dpi=100)
    plt.close() # Close plot to free memory


    return render_template('index.html', 
                           prediction_text=result_text, 
                           prediction_class=css_class, 
                           prob=round(probability*100, 2),
                           suggestion_text=suggestion,
                           plot_url=image_path,
                           timestamp=timestamp) 

@app.route('/dashboard')
def dashboard():
    importances = model.feature_importances_
    data_list = list(importances)
    return render_template('dashboard.html', data=data_list)

if __name__ == '__main__':
    app.run(debug=True)