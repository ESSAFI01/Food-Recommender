import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017" 
MONGO_DB = "openfoodfacts"
MONGO_COLLECTION = "products"

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# Load data
@st.cache_resource
def load_data():
    # Load from MongoDB instead of pickle
    docs = list(collection.find({}, {
        'product_name': 1, 'brands': 1, 'categories': 1, 'countries': 1, 'ingredients_text': 1,
        'allergens': 1, 'nutriscore_score': 1, 'nutriscore_grade': 1, 'energy_100g': 1,
        'fat_100g': 1, 'saturated-fat_100g': 1, 'carbohydrates_100g': 1, 'sugars_100g': 1,
        'fiber_100g': 1, 'proteins_100g': 1, 'salt_100g': 1, '_id': 0,'image_url':1
    }))
    df = pd.DataFrame(docs)
    assets = joblib.load("model_assets.pkl")
    return df, assets['vocab'], assets['synonyms']

df, vocab, SYNONYMS = load_data()

# Initialize vectorizer
vectorizer = TfidfVectorizer(vocabulary=vocab)
tfidf_matrix = vectorizer.fit_transform(df['ingredients_text'])

# Recommendation function (same as before)
def get_recs(ingredients, country=None, nutriscore=None, allergens=None, top_n=5):
    input_text = " ".join([SYNONYMS.get(ing.lower().strip(), ing.lower().strip()) 
                    for ing in ingredients])
    input_vec = vectorizer.transform([input_text])
    
    mask = np.ones(len(df), dtype=bool)
    if country:
        mask &= df['countries'].str.contains(country, case=False, na=False)
    if nutriscore:
        mask &= (df['nutriscore_grade'].str.lower() == nutriscore.lower())
    if allergens:
        for allergen in allergens:
            mask &= (~df['allergens'].str.contains(allergen, case=False, na=True))
    
    filtered_df = df[mask]
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    sim_scores = cosine_similarity(input_vec, vectorizer.transform(filtered_df['ingredients_text']))[0]
    filtered_df = filtered_df.copy()
    filtered_df['similarity'] = sim_scores
    return filtered_df.nlargest(top_n, 'similarity')

# --- NEW: IMPROVED DISPLAY FUNCTIONS ---
def display_nutriscore(grade):
    color_map = {
        'A': '🟢', 'B': '🟢', 
        'C': '🟡', 'D': '🟠', 
        'E': '🔴'
    }
    return f"{color_map.get(grade, '⚪')} {grade}"

def display_product_card(product):
    cols = st.columns([1, 4])
    
    with cols[0]:
        image_url = str(product.get('image_url', '') or '').strip()
        if not image_url or image_url.lower() == 'nan':
            image_url = r"C:\Users\essaf\Desktop\big-data-front\1046857.png"

        st.image(image_url, width=100)
    
    with cols[1]:
        st.subheader(product.get('product_name', 'Unnamed Product'))
        st.caption(f"**Brand:** {product.get('brands', 'N/A')}")
        
        nut_cols = st.columns(4)
        nut_cols[0].metric("Score", display_nutriscore(product.get('nutriscore_grade', 'N/A')))
        nut_cols[1].metric("Energy", f"{product.get('energy_100g', 0):.0f}kcal")
        nut_cols[2].metric("Protein", f"{product.get('proteins_100g', 0):.1f}g")
        nut_cols[3].metric("Match", f"{product.get('similarity', 0):.0%}")
        
        with st.expander("Details"):
            st.write(f"**Ingredients:** {product.get('ingredients_text', 'Not specified')}")
            st.write(f"**Allergens:** {product.get('allergens', 'None detected')}")
            st.write(f"**Countries:** {product.get('countries', 'N/A')}")

# Streamlit UI
st.title("🍏 Smart Food Recommender")
st.markdown("### Discover products matching your dietary needs")

with st.form("recommendation_form"):
    ingredients = st.text_input("What ingredients are you using?", "tomato, cheese, flour")
    country = st.selectbox("Country availability", ["Any"] + sorted(df['countries'].dropna().str.split(',').explode().str.strip().unique().tolist()[:20]))
    nutriscore = st.selectbox("Minimum nutrition quality", ["Any", "A", "B", "C", "D", "E"])
    allergens = st.multiselect("Exclude allergens", ["gluten", "milk", "nuts", "soy", "eggs"])
    submitted = st.form_submit_button("Find Matching Products")

if submitted:
    ingredients_list = [x.strip() for x in ingredients.split(",") if x.strip()]
    
    if not ingredients_list:
        st.error("Please enter at least one ingredient")
    else:
        with st.spinner("🔍 Searching our food database..."):
            recommendations = get_recs(
                ingredients=ingredients_list,
                country=None if country == "Any" else country,
                nutriscore=None if nutriscore == "Any" else nutriscore,
                allergens=allergens
            )
        
        if not recommendations.empty:
            st.success(f"✨ Found {len(recommendations)} great matches!")
            st.divider()
            
            for _, product in recommendations.iterrows():
                display_product_card(product)
                st.divider()
            
            # Download button
            st.download_button(
                label="📥 Download Recommendations",
                data=recommendations.to_csv(index=False),
                file_name="food_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.warning("No products found. Try relaxing your filters.")
            if st.button("Show me similar products anyway"):
                with st.spinner("Finding alternatives..."):
                    relaxed_recs = get_recs(
                        ingredients=ingredients_list,
                        country=None,
                        nutriscore=None,
                        allergens=None
                    )
                if not relaxed_recs.empty:
                    st.info("Here are some similar products without filters:")
                    for _, product in relaxed_recs.head(3).iterrows():
                        display_product_card(product)