# Text Vectorization: Bag of Words vs TF-IDF

A hands-on comparison of text feature extraction techniques for NLP, using scikit-learn on a small movie review dataset. Each method is evaluated by training a Naive Bayes sentiment classifier.

## What's Covered

- *Dataset creation*: 15 labeled movie reviews (positive/negative) in a pandas DataFrame
- *Bag of Words (CountVectorizer)*: vocabulary, matrix shape, density, word frequencies
- *TF-IDF (TfidfVectorizer)*: weighted term importance
- *Sparse vs dense memory comparison* and sparsity analysis
- *N-grams*: unigrams vs bigrams, top bigram analysis
- *Model comparison*: BoW, TF-IDF, BoW + bigrams, TF-IDF + bigrams with MultinomialNB
- *Feature importance*: top words driving positive and negative sentiment
- *Parameter tuning*: max_features and ngram_range experiments
- *Visualizations*: frequency bar charts, word cloud, sparsity plots, heatmap

## Tech Stack

Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, WordCloud

## Setup

bash
git clone https://github.com/<your-username>/nlp-text-vectorization-bow-tfidf.git
cd nlp-text-vectorization-bow-tfidf
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud


## Usage

Run the script or notebook top to bottom. Each section prints its results and shows its plots.

python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
X = vectorizer.fit_transform(df["review"])


## Key Takeaways

- BoW counts words; TF-IDF down-weights words common across all documents
- Text matrices are mostly zeros, so sparse storage saves a lot of memory
- Bigrams capture short phrases (poor acting) but need more data to help
- With only 15 reviews, accuracy numbers are unstable and for learning only

## Next Steps

- Use a larger real dataset (e.g. IMDB reviews)
- Add the NLTK preprocessing pipeline before vectorizing
- Try Logistic Regression and cross-validation
- Explore word embeddings (Word2Vec, GloVe)

## Author

*Farhad Ali Zahid*
AI Operations student | Linux, AWS, Cloud, DevOps

[LinkedIn](https://www.linkedin.com/in/farhad-ali-zahid-99a846446)
