# SmartSeva AI

**AI-powered Intelligent Vendor Finder for home services.**

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Solution](#solution)
3. [How It Works](#how-it-works)
4. [Features](#features)
5. [Tech Stack](#tech-stack)
6. [Setup](#setup)
7. [Usage](#usage)
8. [Contributors](#contributors)
9. [License](#license)

## Problem Statement

Finding reliable home service providers is difficult. Users need an intelligent system that understands natural language queries and recommends the best vendors.

---

## Solution

SmartSeva AI uses semantic search with sentence embeddings to match user queries with relevant service providers. The solution integrates **semantic search** with **machine learning** to provide accurate and context-aware vendor recommendations.

---

## How It Works

1. **User Query**: The user provides a query, such as "cheap plumber for leakage."
2. **Sentence Transformer (Embedding)**: The query is converted into a numerical embedding using a pre-trained model.
3. **Compare with Vendor Embeddings**: The system compares the query embedding with stored vendor embeddings.
4. **Cosine Similarity**: The system uses cosine similarity to rank vendors based on their relevance to the query.
5. **Ranking & Rating Boost**: Vendors are ranked, and a rating-based ranking boost is applied.
6. **Top Results & Explanation**: The top 3 vendors are presented to the user with explanations of why they were selected.

---

## Features

- **Semantic Search**: Matches user queries with relevant service providers based on embeddings.
- **AI-based Recommendation**: Ranks vendors based on similarity and user reviews.
- **Explainable AI**: Provides reasons for each match (e.g., "Matches plumbing & leakage work").
- **Rating-based Ranking**: Boosts highly rated vendors.
- **Vendor Contact Details**: Displays contact information for each vendor.

---

## Tech Stack

- **Python**: Programming language used to build the application.
- **Streamlit**: Framework for building the web application.
- **Sentence Transformers**: Pre-trained models for generating text embeddings.
- **NumPy**: Used for numerical operations and similarity calculations.
- **Git LFS (Large File Storage)**: To manage large files such as models.

---

## Setup

To get started with **SmartSeva AI**, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sagark1406/smartseva-ai.git
   cd smartseva-ai
