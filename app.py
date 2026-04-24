import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="SmartSeva AI", page_icon="🏠")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Vendor data (FIXED)

vendors = [
    {
        "name": "Ramesh Plumber",
        "skills": "pipe fixing, leakage repair",
        "price": "low",
        "location": "Dehradun",
        "rating": 4.5,
        "phone": "9876543210",
        "experience": "5 years"
    },
    {
        "name": "Rahul Plumber",
        "skills": "bathroom fitting, leakage repair",
        "price": "high",
        "location": "Dehradun",
        "rating": 4.7,
        "phone": "9012345678",
        "experience": "8 years"
    },
    {
        "name": "Deepak Plumber",
        "skills": "tap fixing, water tank repair",
        "price": "medium",
        "location": "Delhi",
        "rating": 4.3,
        "phone": "9898989898",
        "experience": "6 years"
    },
    {
        "name": "Suresh Electrician",
        "skills": "wiring, fan repair",
        "price": "medium",
        "location": "Delhi",
        "rating": 4.2,
        "phone": "9123456780",
        "experience": "7 years"
    },
    {
        "name": "Vikas Electrician",
        "skills": "AC wiring, switch repair",
        "price": "high",
        "location": "Noida",
        "rating": 4.6,
        "phone": "9345678901",
        "experience": "9 years"
    },
    {
        "name": "Anil Electrician",
        "skills": "lighting, inverter setup",
        "price": "low",
        "location": "Ghaziabad",
        "rating": 4.1,
        "phone": "9234567890",
        "experience": "4 years"
    },
    {
        "name": "Amit Cleaner",
        "skills": "home cleaning, bathroom cleaning",
        "price": "low",
        "location": "Noida",
        "rating": 4.0,
        "phone": "9988776655",
        "experience": "3 years"
    },
    {
        "name": "Rohit Cleaner",
        "skills": "deep cleaning, sofa cleaning",
        "price": "medium",
        "location": "Delhi",
        "rating": 4.4,
        "phone": "9871234567",
        "experience": "5 years"
    },
    {
        "name": "Sunil Painter",
        "skills": "wall painting, waterproofing",
        "price": "medium",
        "location": "Dehradun",
        "rating": 4.3,
        "phone": "9765432109",
        "experience": "6 years"
    },
    {
        "name": "Karan Painter",
        "skills": "interior painting, texture design",
        "price": "high",
        "location": "Delhi",
        "rating": 4.7,
        "phone": "9654321098",
        "experience": "10 years"
    },
    {
        "name": "Mohit Carpenter",
        "skills": "furniture repair, door fixing",
        "price": "medium",
        "location": "Noida",
        "rating": 4.2,
        "phone": "9543210987",
        "experience": "7 years"
    },
    {
        "name": "Ravi Carpenter",
        "skills": "modular kitchen, wood polishing",
        "price": "high",
        "location": "Delhi",
        "rating": 4.6,
        "phone": "9432109876",
        "experience": "9 years"
    }
]

# 🔥 Convert to embeddings
vendor_texts = [
    f"{v['name']} {v['skills']} {v['price']} {v['location']} {v['experience']}"
    for v in vendors
]

vendor_embeddings = model.encode(vendor_texts)
# 🔥 Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 🔥 Explainable AI
def get_reason(query, vendor):
    reasons = []

    query = query.lower()
    skills = vendor["skills"].lower()

    if "plumber" in query or "leak" in query:
        if "leak" in skills or "pipe" in skills:
            reasons.append("Matches plumbing & leakage work")

    if "cheap" in query or "low" in query:
        if vendor["price"] == "low":
            reasons.append("Fits your low budget")

    if "electric" in query:
        if "wiring" in skills:
            reasons.append("Matches electrical work")

    if not reasons:
        reasons.append("Best overall match based on your query")

    return reasons

# 🔥 UI
st.title("🏠 SmartSeva AI - Intelligent Vendor Finder")

query = st.text_input(
    "🔍 Describe your need (e.g. cheap plumber for leakage)"
)

# 🔥 Search
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
        st.write(f"🧠 Experience: {v['experience']}")
        st.write(f"📞 Contact: {v['phone']}")
        st.write(f"🔗 Match Score: {round(score, 2)}")

        reasons = get_reason(query, v)
        st.markdown("### 💡 Why this match?")
        for reason in reasons:
            st.write(f"✔ {reason}")

        st.write("---")
