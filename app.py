from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Explicitly expose the app for Vercel
app = app 

@app.route('/')
def home():
    return "Backend Running ✅"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        df = pd.read_csv(file)

        # 1. Clean Week column
        df['Week'] = df['Week'].astype(str).str.extract('(\d+)').astype(int)

        # 2. Create Behaviour (Handling the first row NaN)
        df['FocusDiff'] = df['FocusScore'].diff()
        # We fill the first NaN with 0 so the first row isn't immediately deleted
        df['FocusDiff'] = df['FocusDiff'].fillna(0) 
        
        df['Behaviour'] = df['FocusDiff'].apply(
            lambda x: "Improving" if x > 0 else ("Declining" if x < 0 else "Stable")
        )

        # 3. ML Preparation
        X = df[['FocusScore', 'Verbal', 'Visual', 'Physical', 'Written']]
        y = df['Behaviour']

        # Ensure we have enough data to split
        if len(df) < 5:
            return jsonify({"error": "Not enough data rows (minimum 5) for analytics"}), 400

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # Use zero_division to avoid errors if a class is missing in small datasets
        acc = accuracy_score(y_test, y_pred)

        # 4. Data Grouping for Frontend
        weekly_data = {}
        modality_data = {}
        for week in sorted(df['Week'].unique()):
            temp = df[df['Week'] == week]
            weekly_data[f"Week {week}"] = temp['FocusScore'].tolist()
            modality_data[f"Week {week}"] = [
                round(temp['Verbal'].mean(), 2),
                round(temp['Visual'].mean(), 2),
                round(temp['Physical'].mean(), 2),
                round(temp['Written'].mean(), 2)
            ]

        # 5. Final Metrics
        overall_focus = round(df['FocusScore'].mean(), 2)
        modalities = ['Verbal', 'Visual', 'Physical', 'Written']
        dominant = max(modalities, key=lambda x: df[x].mean())

        weekly_avg = df.groupby('Week')['FocusScore'].mean()
        latest = weekly_avg.iloc[-1]
        prev = weekly_avg.iloc[-2] if len(weekly_avg) > 1 else latest

        status = "Stable"
        if latest > prev: status = "Improving"
        elif latest < prev: status = "Declining"

        week_change = round(((latest - prev) / prev) * 100, 2) if prev != 0 else 0

        return jsonify({
            "accuracy": round(acc, 2),
            "status": status,
            "overall_focus": overall_focus,
            "week_change": week_change,
            "dominant": dominant,
            "weekly_data": weekly_data,
            "modality_data": modality_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# DO NOT use app.run() for Vercel. 
# Vercel uses the 'app' object directly.