import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample Vendor Data
vendors = [
    {"name": "Ramesh Plumber", "skills": "pipe fixing, leakage repair", "price": "low", "location": "Dehradun", "rating": 4.5},
    {"name": "Suresh Electrician", "skills": "wiring, fan repair", "price": "medium", "location": "Dehradun", "rating": 4.2},
    {"name": "Amit Cleaner", "skills": "home cleaning, bathroom cleaning", "price": "low", "location": "Dehradun", "rating": 4.0},
    {"name": "Rahul Plumber", "skills": "bathroom fitting, leakage repair", "price": "high", "location": "Dehradun", "rating": 4.8}
]

# Convert vendor to text
def vendor_to_text(v):
    return f"{v['name']} offers {v['skills']} with {v['price']} pricing in {v['location']} and rating {v['rating']}"

# Create embeddings
vendor_texts = [vendor_to_text(v) for v in vendors]
vendor_embeddings = model.encode(vendor_texts)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# UI
st.set_page_config(page_title="SmartSeva AI", page_icon="🏠")
st.title("🏠 SmartSeva AI - Intelligent Vendor Finder")

query = st.text_input("🔍 Describe your need (e.g. cheap plumber for leakage)")

if query:
    query_embedding = model.encode(query)

    scores = []
    for i, emb in enumerate(vendor_embeddings):
        similarity = cosine_similarity(query_embedding, emb)
        final_score = similarity * 0.7 + (vendors[i]['rating'] / 5) * 0.3
        scores.append((final_score, vendors[i]))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)

    st.subheader("✨ Best Matches for You")

    for score, v in scores[:3]:
        st.markdown(f"### 👨‍🔧 {v['name']}")
        st.write(f"🛠 Skills: {v['skills']}")
        st.write(f"💰 Price: {v['price']}")
        st.write(f"⭐ Rating: {v['rating']}")
        st.write(f"📍 Location: {v['location']}")
        st.write(f"🔗 Match Score: {round(score, 2)}")
        st.write("---")
