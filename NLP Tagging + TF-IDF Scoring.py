import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# === Start Timer ===
start_time = time.time()

df = pd.read_csv("C:/Users/rishi/Desktop/project/Eligible companies_updated_bd.csv")
df['Description'] = df['Description'].fillna('').astype(str)

# === Load spaCy model ===
nlp = spacy.load("en_core_web_sm")

# === Define theme patterns ===
theme_keywords = {
    "AI-Chips": ["AI", "artificial intelligence", "neural network", "deep learning", "machine learning", "chip,", "semiconductor", "generative AI", "TPU", "data center"],
    "Nuclear Energy": ["nuclear", "uranium", "atomic", "fission", "fusion"],
    "SMRs": ["small modular reactor", "SMR", "modular nuclear"],
    "Quantum Computing": ["quantum", "superposition", "entanglement", "qubit"]
}

# === Phrase Matching ===
def match_themes(text):
    matched = set()
    text_lower = text.lower()
    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.add(theme)
    return list(matched)

df['Matched_Themes'] = df['Description'].apply(match_themes)
print("NLP tagging complete.")

# === TF-IDF Setup ===
print("🔹 Computing TF-IDF scores...")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
feature_names = vectorizer.get_feature_names_out()

# === Compute Theme TF-IDF Scores ===
def compute_theme_score(row_idx):
    scores = {}
    tfidf_vector = tfidf_matrix[row_idx]
    tfidf_dict = dict(zip(feature_names, tfidf_vector.toarray()[0]))
    for theme, keywords in theme_keywords.items():
        score = sum(tfidf_dict.get(kw.lower(), 0) for kw in keywords)
        scores[theme] = round(score, 4)
    return scores

tfidf_scores = [compute_theme_score(i) for i in range(len(df))]
df_scores = pd.DataFrame(tfidf_scores)

# === Combine Results ===
df_combined = pd.concat([df, df_scores], axis=1)
df_combined['Dominant_Theme'] = df_combined[theme_keywords.keys()].idxmax(axis=1)
print("Theme scoring complete.")

# === Save Output ===
output_path = "C:/Users/rishi/Desktop/project/target_sector_NLP.csv"
df_combined.to_csv(output_path, index=False)

# === End Timer ===
end_time = time.time()
print(f"Finished! Time taken: {round(end_time - start_time, 2)} seconds.")
