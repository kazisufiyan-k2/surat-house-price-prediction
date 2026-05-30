import json
import os
import io
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_deployment_model.pkl')
model = joblib.load(MODEL_PATH)


def handler(event, context):
    if event.get('httpMethod') != 'POST':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Method Not Allowed'})
        }

    try:
        body = event.get('body', '{}') or '{}'
        data = json.loads(body)

        area = data.get('area', '')
        locality = data.get('locality', '')
        bhk = int(data.get('bhk', 0))
        sqft = float(data.get('sqft', 0))
        floor = int(data.get('floor', 0))
        age = int(data.get('age', 0))
        parking = data.get('parking', 'No')
        near_metro = data.get('near_metro', 'No')

        features = pd.DataFrame([{
            'Area': area,
            'Locality': locality,
            'BHK': bhk,
            'TotalSqFt': sqft,
            'Floor': floor,
            'AgeOfProperty': age,
            'Parking': parking,
            'NearMetro': near_metro
        }])

        prediction = model.predict(features)[0]
        price_fmt = f'₹ {round(float(prediction)):,}'
        price_lakhs = f'≈ ₹ {round(float(prediction) / 100000, 2)} Lakhs'

        graph = None
        try:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            labels = ['BHK', 'TotalSqFt', 'Floor', 'Age(yrs)']
            values = [bhk, sqft, floor, age]
            colors = ['#00c6fb', '#43e97b', '#f9ca24', '#f0932b']
            ax.bar(labels, values, color=colors, edgecolor='none', width=0.5)
            ax.set_title('Feature Input Summary', color='white', fontsize=12, pad=10)
            ax.tick_params(colors='#a0aec0')
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=120, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            img.seek(0)
            graph = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
            img.close()
        except Exception as chart_err:
            plt.close('all')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction_text': price_fmt,
                'price_lakhs': price_lakhs,
                'area': area,
                'locality': locality,
                'graph': graph
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
