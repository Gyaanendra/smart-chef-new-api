from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import spacy
import warnings
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database connection function
def get_db_connection():
    try:
        conn = psycopg2.connect(
            os.getenv('DATABASE_URL'),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Load models and data
nlp = spacy.load("en_core_web_sm")
tfidf_vectorizer = joblib.load('tfidf_vectorizer_model.joblib')
tfidf_matrix = joblib.load('tfidf_matrix_model.joblib')

# Load recipe data
rdf = pd.read_csv('https://raw.githubusercontent.com/Gyaanendra/gfg-hackfest/refs/heads/main/data/postgres_data.csv')
rdf['CleanedIngredients'] = rdf['CleanedIngredients'].fillna('')

# Helper functions
def preprocess_text(text):
    text = str(text).lower()
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def calculate_similarity(user_ingredients, user_prep_time, user_cook_time):
    user_ingredients_text = preprocess_text(', '.join(user_ingredients))
    user_tfidf = tfidf_vectorizer.transform([user_ingredients_text])
    
    ingredient_similarity = np.asarray(cosine_similarity(user_tfidf, tfidf_matrix)[0])
    prep_time_similarity = 1 - abs(rdf['PrepTimeInMins'] - user_prep_time) / rdf['PrepTimeInMins'].max()
    cook_time_similarity = 1 - abs(rdf['CookTimeInMins'] - user_cook_time) / rdf['CookTimeInMins'].max()
    
    min_length = min(len(ingredient_similarity), len(prep_time_similarity), len(cook_time_similarity))
    ingredient_similarity = ingredient_similarity[:min_length]
    prep_time_similarity = prep_time_similarity[:min_length]
    cook_time_similarity = cook_time_similarity[:min_length]
    
    return (ingredient_similarity + prep_time_similarity + cook_time_similarity) / 3

def recommend_recipes(user_ingredients, user_prep_time, user_cook_time, top_n=10):
    combined_similarity = calculate_similarity(user_ingredients, user_prep_time, user_cook_time)
    sorted_indices = combined_similarity.argsort()[::-1][:top_n]
    return rdf.iloc[sorted_indices].copy()

# API endpoints
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200

@app.route("/api/display_recipe", methods=["GET"])
def display_recipe():
    try:
        cleaned_rdf = rdf.fillna('')  # Replace NaN values with empty strings
        recipes = cleaned_rdf[[
            'TranslatedRecipeName', 'TranslatedIngredients', 'PrepTimeInMins',
            'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine',
            'Course', 'Diet', 'TranslatedInstructions', 'image_src', 'unique_id'
        ]].to_dict(orient='records')
        
        return jsonify(recipes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recipeById", methods=["POST"])
def get_recipe_by_id():
    try:
        request_data = request.get_json()
        unique_id = request_data.get('unique_id')
        if not unique_id:
            return jsonify({"error": "unique_id is required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection error"}), 500

        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM recipes_data WHERE unique_id = %s LIMIT 1
            """, (unique_id,))
            recipe = cur.fetchone()
            
            if not recipe:
                return jsonify({"error": "Recipe not found"}), 404

            return jsonify({k: ('' if v is None else v) for k, v in recipe.items()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendation", methods=["POST"])
def recommendation():
    try:
        request_data = request.get_json()
        user_ingredients = request_data.get('user_ingredients', [])
        user_prep_time = request_data.get('user_prep_time', 0)
        user_cook_time = request_data.get('user_cook_time', 0)
        n_recipes = request_data.get('n_recipes', 20)

        if not isinstance(user_ingredients, list) or not user_ingredients:
            return jsonify({"error": "user_ingredients must be a non-empty list"}), 400
        
        user_prep_time = int(user_prep_time)
        user_cook_time = int(user_cook_time)
        n_recipes = int(n_recipes)

        if user_prep_time <= 0 or user_cook_time <= 0:
            return jsonify({"error": "Time values must be positive"}), 400

        recommendations = recommend_recipes(
            user_ingredients,
            user_prep_time,
            user_cook_time,
            top_n=n_recipes
        ).fillna('')  # Replace NaN values in the recommended recipes

        return jsonify(recommendations[[
            'TranslatedRecipeName', 'TranslatedIngredients', 'PrepTimeInMins',
            'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine',
            'Course', 'Diet', 'TranslatedInstructions', 'image_src', 'unique_id'
        ]].to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
