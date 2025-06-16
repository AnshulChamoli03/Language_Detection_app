import numpy as np
from flask import Flask, request, render_template, flash
import pickle
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flask_app = Flask(__name__)
flask_app.secret_key = os.urandom(24)  # Required for flash messages

# Get the absolute path to the model files
current_dir = Path(__file__).parent
model_path = current_dir / "model.pkl"
cv_path = current_dir / "cv.pkl"

# Load model and vectorizer with error handling
try:
    logger.info(f"Attempting to load model from: {model_path}")
    logger.info(f"Attempting to load vectorizer from: {cv_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not cv_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at: {cv_path}")
    
    model = pickle.load(open(model_path, 'rb'))
    logger.info("Model loaded successfully")
    
    cv = pickle.load(open(cv_path, 'rb'))
    logger.info("Vectorizer loaded successfully")
    
except FileNotFoundError as e:
    logger.error(f"File not found error: {str(e)}")
    raise
except pickle.UnpicklingError as e:
    logger.error(f"Error unpickling the model files: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading model files: {str(e)}")
    raise

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=['POST'])
def Predict():
    try:
        # Get input text
        feature = [str(x) for x in request.form.values()]
        feature = "".join(feature)
        
        # Input validation
        if not feature.strip():
            flash("Please enter some text for language detection")
            return render_template("index.html")
        
        # Transform input
        data = cv.transform([feature]).toarray()
        
        # Make prediction
        prediction = model.predict(data)
        predicted_language = prediction[0]
        
        # Log prediction
        logger.info(f"Input text: {feature[:50]}... | Predicted language: {predicted_language}")
        
        return render_template(
            "index.html",
            prediction_text=f"The predicted language is {predicted_language}"
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        flash("An error occurred during language detection. Please try again.")
        return render_template("index.html")

if __name__ == "__main__":
    flask_app.run(debug=True)
