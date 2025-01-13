from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(docs_url=None, redoc_url=None)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        raise HTTPException(status_code=500, detail="Database connection error")

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
@app.get("/")
async def d():
    return "Server is running ",200


@app.get("/api/display_recipe")
async def display_recipe():
    try:
        cleaned_rdf = rdf.fillna('')  # Replace NaN values with empty strings
        recipes = cleaned_rdf[[
            'TranslatedRecipeName', 'TranslatedIngredients', 'PrepTimeInMins',
            'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine',
            'Course', 'Diet', 'TranslatedInstructions', 'image_src', 'unique_id'
        ]].to_dict(orient='records')
        
        return recipes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

     
     
db_params = os.getenv('DATABASE_URL')

@app.post("/api/recipeById")
async def get_recipe_by_id(request: dict):
    try:
        unique_id = request.get('unique_id')
        if not unique_id:
            return {"error": "unique_id is required"}, 400

        with psycopg2.connect(db_params, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM recipes_data WHERE unique_id = %s LIMIT 1
                """, (unique_id,))
                recipe = cur.fetchone()
                
                if not recipe:
                    return {"error": "Recipe not found"}, 404

                # Handle None values
                return {k: ('' if v is None else v) for k, v in recipe.items()}
                
    except Exception as e:
        return {"error": str(e)}, 500
            
            
            
@app.post("/api/recommendation")
async def recommendation(request: dict):
    # Extract and validate input
    user_ingredients = request.get('user_ingredients', [])
    user_prep_time = request.get('user_prep_time', 0)
    user_cook_time = request.get('user_cook_time', 0)
    n_recipes = request.get('n_recipes', 20)

    # Basic validation
    if not isinstance(user_ingredients, list) or not user_ingredients:
        raise HTTPException(status_code=400, detail="user_ingredients must be a non-empty list")
    
    try:
        user_prep_time = int(user_prep_time)
        user_cook_time = int(user_cook_time)
        n_recipes = int(n_recipes)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Time values must be valid integers")

    if user_prep_time <= 0 or user_cook_time <= 0:
        raise HTTPException(status_code=400, detail="Time values must be positive")

    recommendations = recommend_recipes(
        user_ingredients,
        user_prep_time,
        user_cook_time,
        top_n=n_recipes
    ).fillna('')  # Replace NaN values in the recommended recipes

    return recommendations[[
        'TranslatedRecipeName', 'TranslatedIngredients', 'PrepTimeInMins',
        'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine',
        'Course', 'Diet', 'TranslatedInstructions', 'image_src', 'unique_id'
    ]].to_dict(orient='records')


# @app.get("/api/options")
# async def get_options():
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
        
#         # Get unique diets
#         cur.execute("SELECT DISTINCT diet FROM recipe_data WHERE diet IS NOT NULL ORDER BY diet")
#         diets = [None] + [row['diet'] for row in cur.fetchall()]
        
#         # Get unique cuisines
#         cur.execute("SELECT DISTINCT cuisine FROM recipe_data WHERE cuisine IS NOT NULL ORDER BY cuisine")
#         cuisines = [None] + [row['cuisine'] for row in cur.fetchall()]
        
#         # Get unique courses
#         cur.execute("SELECT DISTINCT course FROM recipe_data WHERE course IS NOT NULL ORDER BY course")
#         courses = [None] + [row['course'] for row in cur.fetchall()]
        
#         cur.close()
#         conn.close()
        
#         return {
#             "diets": diets,
#             "cuisines": cuisines,
#             "courses": courses
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app"
    )