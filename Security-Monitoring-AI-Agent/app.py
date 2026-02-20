from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Home page just shows the upload form
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the uploaded CSV
        file = request.files['file']
        df = pd.read_csv(file)

        # Keep only numeric columns for anomaly detection
        numeric_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_columns)

        if not df.empty:
            # Isolation Forest model
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(df[numeric_columns])
            df['anomaly'] = model.predict(df[numeric_columns])
            anomalies = df[df['anomaly'] == -1]
        else:
            anomalies = pd.DataFrame()

        # Render the HTML with anomaly table
        return render_template(
            "index.html",
            tables=[anomalies.to_html(classes='data')],
            titles=anomalies.columns.values
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)