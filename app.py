from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return "Backend Running ✅"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"error": "No file"}), 400

        df = pd.read_csv(file)

        # Convert Week → number
        df['Week'] = df['Week'].str.extract('(\d+)').astype(int)

        # Create Behaviour
        df['Behaviour'] = df['FocusScore'].diff().apply(
            lambda x: "Improving" if x > 0 else ("Declining" if x < 0 else "Stable")
        )

        df = df.dropna()

        # ML
        X = df[['FocusScore', 'Verbal', 'Visual', 'Physical', 'Written']]
        y = df['Behaviour']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Weekly data
        weekly_data = {}
        for week in sorted(df['Week'].unique()):
            weekly_data[f"Week {week}"] = df[df['Week']==week]['FocusScore'].tolist()

        # Modality data
        modality_data = {}
        for week in sorted(df['Week'].unique()):
            temp = df[df['Week']==week]
            modality_data[f"Week {week}"] = [
                temp['Verbal'].mean(),
                temp['Visual'].mean(),
                temp['Physical'].mean(),
                temp['Written'].mean()
            ]

        # Metrics
        overall_focus = round(df['FocusScore'].mean(), 2)

        modalities = ['Verbal','Visual','Physical','Written']
        dominant = max(modalities, key=lambda x: df[x].mean())

        weekly_avg = df.groupby('Week')['FocusScore'].mean()
        latest = weekly_avg.iloc[-1]
        prev = weekly_avg.iloc[-2] if len(weekly_avg) > 1 else latest

        if latest > prev:
            status = "Improving"
        elif latest < prev:
            status = "Declining"
        else:
            status = "Stable"

        week_change = round(((latest - prev) / prev) * 100, 2) if prev != 0 else 0

        return jsonify({
            "accuracy": round(acc,2),
            "status": status,
            "overall_focus": overall_focus,
            "week_change": week_change,
            "dominant": dominant,
            "weekly_data": weekly_data,
            "modality_data": modality_data
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})

# -----------------------------
# Deployment configuration for Vercel
# -----------------------------
# Vercel sets the PORT as environment variable
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)