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
import requests
import json
import google.generativeai as genai
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure the API key for the generative AI model
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

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

# Load recipe data from the new URL
rdf = pd.read_csv('https://raw.githubusercontent.com/Gyaanendra/smart-chef-new-data/refs/heads/main/recipes_data.csv')
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
        ]].sample(frac=1).to_dict(orient='records')  # Shuffle the dataframe
        
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

@app.route("/api/more_recipes", methods=["POST"])
def more_recipes():
    try:
        # Get the input data from the request
        request_data = request.get_json()
        
        # Extract the required fields
        prep_time = request_data.get('PrepTimeInMins', 0)
        cook_time = request_data.get('CookTimeInMins', 0)
        ingredients = request_data.get('ingredients', [])  # Now expecting an array
        
        # Validate inputs
        if not isinstance(ingredients, list):
            return jsonify({
                "error": "ingredients must be an array"
            }), 400
            
        if not all([prep_time, cook_time, ingredients]):
            return jsonify({
                "error": "Missing required fields. Please provide PrepTimeInMins, CookTimeInMins, and ingredients"
            }), 400

        # Clean ingredients array - remove empty strings and strip whitespace
        ingredient_list = [ing.strip() for ing in ingredients if ing and ing.strip()]

        # Use the existing recommendation function
        recommendations = recommend_recipes(
            user_ingredients=ingredient_list,
            user_prep_time=int(prep_time),
            user_cook_time=int(cook_time),
            top_n=15  # Get 15 similar recipes
        ).fillna('')  # Replace NaN values


        return jsonify(recommendations[[
            'TranslatedRecipeName', 'TranslatedIngredients', 'PrepTimeInMins',
            'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine',
            'Course', 'Diet', 'TranslatedInstructions', 'image_src', 'unique_id'
        ]].to_dict(orient='records')),200

    except Exception as e:
        print(f"Error in more_recipes: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/like", methods=["POST"])
def like_recipe():
    try:
        request_data = request.get_json()
        user_id = request_data.get('user_id')
        recipe_id = request_data.get('recipe_id')

        if not user_id or not recipe_id:
            return jsonify({"error": "user_id and recipe_id are required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection error"}), 500

        with conn.cursor() as cur:
            # Check if the user has already liked the recipe
            cur.execute("""
                SELECT 1 FROM recipes_liked_byuser
                WHERE user_id = %s AND recipe_id = %s
            """, (user_id, recipe_id))
            if cur.fetchone():
                return jsonify({"message": "Recipe already liked"}), 200

            # Insert the like if it doesn't exist
            cur.execute("""
                INSERT INTO recipes_liked_byuser (user_id, recipe_id)
                VALUES (%s, %s)
            """, (user_id, recipe_id))
            conn.commit()

        conn.close()
        return jsonify({"message": "Recipe liked successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/liked", methods=["POST"])
def get_liked_recipes():
    try:
        request_data = request.get_json()
        user_id = request_data.get('user_id')

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection error"}), 500

        with conn.cursor() as cur:
            cur.execute("""
                SELECT recipe_id FROM recipes_liked_byuser
                WHERE user_id = %s
            """, (user_id,))
            liked_recipes = [row['recipe_id'] for row in cur.fetchall()]

        conn.close()
        return jsonify({"recipes_liked": liked_recipes}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generated_recommendation", methods=["POST"])
def generated_recommendation():
    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Check if prompt exists in the request
        if "prompt" not in input_data:
            return jsonify({"error": "Missing prompt in request"}), 400

        # Log the incoming prompt for debugging
        print(f"Received Prompt: {input_data['prompt']}")

        # Call the Gemini API to generate new recipes
        generated_recipes = call_gemini_api(input_data["prompt"])

        # Return the generated recipes
        return jsonify(generated_recipes), 200

    except Exception as e:
        print(f"Error in generated_recommendation: {str(e)}")
        return jsonify({"error": str(e)}), 500

def call_gemini_api(prompt_text):
    try:
        # Create an instance of the GenerativeModel
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Create a structured prompt that specifies the exact format required
        structured_prompt = f"""Generate 15 recipes based on this prompt:
{prompt_text}

Each recipe must be returned as a JSON object with exactly these fields:
- CookTimeInMins (number between 10-120)
- Course (string: Breakfast, Lunch, Dinner, or Snack)
- Cuisine (string indicating the cuisine type)
- Diet (string: "Vegetarian" or "Non Vegetarian")
- PrepTimeInMins (number between 5-60)
- Servings (number between 1-6)
- TotalTimeInMins (sum of PrepTimeInMins and CookTimeInMins)
- TranslatedIngredients (detailed list of ingredients with quantities)
- TranslatedInstructions (detailed, step-by-step cooking instructions)
- TranslatedRecipeName (descriptive name of the recipe)

Return ONLY a JSON array of recipes with these exact fields. No additional fields or explanatory text.
Each recipe should be practical, realistic, and include detailed measurements and clear instructions.
"""

        # Generate content using the model
        response = model.generate_content(structured_prompt)

        if response and hasattr(response, 'text'):
            # Log the raw response for debugging
            print(f"Raw API Response: {response.text}")

            # Clean the response text
            clean_text = response.text.replace("```json", "").replace("```", "").strip()

            try:
                # Parse the JSON response
                recipes = json.loads(clean_text)

                # Ensure the response is a list
                if not isinstance(recipes, list):
                    print("Response is not a list. Returning empty list.")
                    return []

                # Validate and format each recipe
                formatted_recipes = []
                for recipe in recipes:
                    if validate_recipe(recipe):
                        formatted_recipes.append({
                            "CookTimeInMins": recipe["CookTimeInMins"],
                            "Course": recipe["Course"],
                            "Cuisine": recipe["Cuisine"],
                            "Diet": recipe["Diet"],
                            "PrepTimeInMins": recipe["PrepTimeInMins"],
                            "Servings": recipe["Servings"],
                            "TotalTimeInMins": recipe["TotalTimeInMins"],
                            "TranslatedIngredients": recipe["TranslatedIngredients"],
                            "TranslatedInstructions": recipe["TranslatedInstructions"],
                            "TranslatedRecipeName": recipe["TranslatedRecipeName"]
                        })

                return formatted_recipes

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}\nResponse text: {clean_text}")
                return []

        print("No valid response from the Gemini API.")
        return []

    except Exception as e:
        print(f"Error in call_gemini_api: {str(e)}")
        return []

def validate_recipe(recipe):
    """Validate that a recipe contains all required fields with appropriate values."""
    required_fields = [
        "CookTimeInMins", "Course", "Cuisine", "Diet", "PrepTimeInMins",
        "Servings", "TotalTimeInMins", "TranslatedIngredients",
        "TranslatedInstructions", "TranslatedRecipeName"
    ]
    
    # Check all required fields exist
    if not all(field in recipe for field in required_fields):
        return False
    
    # Validate numeric fields
    try:
        if not (10 <= recipe["CookTimeInMins"] <= 120):
            return False
        if not (5 <= recipe["PrepTimeInMins"] <= 60):
            return False
        if not (1 <= recipe["Servings"] <= 6):
            return False
        if recipe["TotalTimeInMins"] != recipe["PrepTimeInMins"] + recipe["CookTimeInMins"]:
            return False
    except (TypeError, ValueError):
        return False
    
    # Validate Diet field
    if recipe["Diet"] not in ["Vegetarian", "Non Vegetarian"]:
        return False
    
    # Validate Course field
    if recipe["Course"] not in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        return False
    
    return True

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        "error": "Bad Request",
        "message": "The server could not process this request. Please ensure you're using the correct protocol and content type."
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500

if __name__ == "__main__":
    app.run(debug=True)
