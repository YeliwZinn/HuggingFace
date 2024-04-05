import transformers
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis model (replace with your chosen model)
model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
nlp = pipeline("sentiment-analysis", model=model_name)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
  text = request.form["text"]
  results = nlp(text)
  return render_template("results.html", results=results)

if __name__ == "__main__":
  app.run(debug=True)
