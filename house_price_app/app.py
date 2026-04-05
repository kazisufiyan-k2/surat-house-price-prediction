from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# ── Flask App ──
app = Flask(__name__)

# ── Load Model ──
model = joblib.load("final_deployment_model.pkl")
print("✅ Model loaded! Features:", list(model.feature_names_in_))

# ── Dropdown Options (from dataset) ──
AREAS = ['Adajan', 'Athwa', 'CityLight', 'Dindoli', 'Katargam',
         'Pal', 'Piplod', 'Udhna', 'Varachha', 'Vesu']

LOCALITIES = ['Adajan Gam', 'Athwa Lines', 'Canal Road', 'City Light Road',
              'Dindoli Area', 'Katargam Road', 'LP Savani Road', 'Pal Gam',
              'Pal RTO', 'Piplod Area', 'Udhna Road', 'Varachha Road', 'Vesu Main Road']

# ── Home Route ──
@app.route("/")
def home():
    return render_template("index.html", areas=AREAS, localities=LOCALITIES)

# ── Prediction Route ──
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs — exact model column names
        area       = request.form.get("area", "")          # Adajan, Vesu etc.
        locality   = request.form.get("locality", "")      # Adajan Gam etc.
        bhk        = int(request.form.get("bhk", 0))
        sqft       = float(request.form.get("sqft", 0))
        floor      = int(request.form.get("floor", 0))
        age        = int(request.form.get("age", 0))
        parking    = request.form.get("parking", "No")       # Yes / No
        near_metro = request.form.get("near_metro", "No")    # Yes / No

        # ── Exact column names as model was trained ──
        features = pd.DataFrame([{
            "Area"          : area,
            "Locality"      : locality,
            "BHK"           : bhk,
            "TotalSqFt"     : sqft,
            "Floor"         : floor,
            "AgeOfProperty" : age,
            "Parking"       : parking,
            "NearMetro"     : near_metro
        }])

        print("Input:\n", features)

        # ── Predict ──
        prediction  = model.predict(features)[0]
        price_fmt   = f"₹ {round(float(prediction)):,}"
        price_lakhs = f"≈ ₹ {round(float(prediction) / 100000, 2)} Lakhs"

        # ── Chart ──
        graph = None
        try:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            labels = ["BHK", "TotalSqFt", "Floor", "Age(yrs)"]
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
            graph = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
            img.close()
        except Exception as ce:
            print(f"Chart error: {ce}")
            plt.close('all')

        return render_template(
            "index.html",
            areas=AREAS,
            localities=LOCALITIES,
            prediction_text=price_fmt,
            price_lakhs=price_lakhs,
            selected_area=area,
            selected_locality=locality,
            graph=graph
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template(
            "index.html",
            areas=AREAS,
            localities=LOCALITIES,
            error_text=f"⚠️ Error: {str(e)}"
        )

# ── Run ──
if __name__ == '__main__':
    app.run(debug=False, port=5000)
    