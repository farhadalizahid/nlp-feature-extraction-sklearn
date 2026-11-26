# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")

# Create sample movie reviews dataset
sample_reviews = [
    "This movie is absolutely fantastic and amazing",
    "I loved this film, it was great and wonderful",
    "Terrible movie, very bad and disappointing",
    "Amazing cinematography and excellent acting performance",
    "Boring film with poor storyline and bad acting",
    "Outstanding movie with brilliant direction and script",
    "Worst movie ever, completely waste of time",
    "Incredible story with fantastic visual effects",
    "Poor quality film with terrible sound effects",
    "Excellent movie with great character development",
    "Awful acting and boring plot throughout",
    "Masterpiece with outstanding performances by all actors",
    "Disappointing film with weak storyline and direction",
    "Brilliant movie with amazing special effects",
    "Terrible script and poor character development"
]

# Create corresponding labels (1 for positive, 0 for negative)
labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Create DataFrame
df = pd.DataFrame({
    'review': sample_reviews,
    'sentiment': labels
})

print("Sample Dataset:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"Positive reviews: {sum(labels)}")
print(f"Negative reviews: {len(labels) - sum(labels)}")

# Initialize CountVectorizer with basic parameters
count_vectorizer = CountVectorizer(
    lowercase=True,           # Convert to lowercase
    stop_words='english',     # Remove English stop words
    max_features=100          # Limit to top 100 features
)

# Fit and transform the text data
bow_matrix = count_vectorizer.fit_transform(df['review'])

print("Bag of Words Matrix Information:")
print(f"Matrix shape: {bow_matrix.shape}")
print(f"Matrix type: {type(bow_matrix)}")
print(f"Matrix density: {bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.4f}")

# Get feature names (vocabulary)
feature_names = count_vectorizer.get_feature_names_out()
print(f"\nVocabulary size: {len(feature_names)}")
print(f"First 20 features: {feature_names[:20]}")


# Convert sparse matrix to dense for visualization
bow_dense = bow_matrix.toarray()

# Create DataFrame for better visualization
bow_df = pd.DataFrame(bow_dense, columns=feature_names)

print("BoW Matrix (first 5 documents, first 10 features):")
print(bow_df.iloc[:5, :10])

# Show word frequencies across all documents
word_frequencies = np.sum(bow_dense, axis=0)
word_freq_df = pd.DataFrame({
    'word': feature_names,
    'frequency': word_frequencies
}).sort_values('frequency', ascending=False)

print("\nTop 15 most frequent words:")
print(word_freq_df.head(15))

# Plot top 20 most frequent words
plt.figure(figsize=(12, 6))
top_words = word_freq_df.head(20)
plt.bar(range(len(top_words)), top_words['frequency'])
plt.xticks(range(len(top_words)), top_words['word'], rotation=45, ha='right')
plt.title('Top 20 Most Frequent Words in BoW')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Create word cloud
wordcloud_text = ' '.join(df['review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Reviews')
plt.show()

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=100,
    ngram_range=(1, 1)  # Only unigrams for now
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

print("TF-IDF Matrix Information:")
print(f"Matrix shape: {tfidf_matrix.shape}")
print(f"Matrix type: {type(tfidf_matrix)}")
print(f"Matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

# Get feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\nVocabulary size: {len(tfidf_feature_names)}")



# Compare memory usage
import sys

print("Memory Usage Comparison:")
print(f"Sparse BoW matrix: {bow_matrix.data.nbytes + bow_matrix.indices.nbytes + bow_matrix.indptr.nbytes} bytes")
print(f"Dense BoW matrix: {bow_dense.nbytes} bytes")
print(f"Sparse TF-IDF matrix: {tfidf_matrix.data.nbytes + tfidf_matrix.indices.nbytes + tfidf_matrix.indptr.nbytes} bytes")
tfidf_dense = tfidf_matrix.toarray()
print(f"Dense TF-IDF matrix: {tfidf_dense.nbytes} bytes")

# Show sparsity information
print(f"\nSparsity Information:")
print(f"BoW matrix - Non-zero elements: {bow_matrix.nnz} out of {bow_matrix.shape[0] * bow_matrix.shape[1]}")
print(f"BoW matrix - Sparsity: {(1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.2f}%")
print(f"TF-IDF matrix - Non-zero elements: {tfidf_matrix.nnz} out of {tfidf_matrix.shape[0] * tfidf_matrix.shape[1]}")
print(f"TF-IDF matrix - Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")




# Visualize sparse matrix structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# BoW matrix sparsity pattern
ax1.spy(bow_matrix, markersize=2)
ax1.set_title('BoW Matrix Sparsity Pattern')
ax1.set_xlabel('Features')
ax1.set_ylabel('Documents')

# TF-IDF matrix sparsity pattern
ax2.spy(tfidf_matrix, markersize=2)
ax2.set_title('TF-IDF Matrix Sparsity Pattern')
ax2.set_xlabel('Features')
ax2.set_ylabel('Documents')

plt.tight_layout()
plt.show()

# Show actual values in a heatmap (for small subset)
plt.figure(figsize=(12, 8))
subset_bow = bow_df.iloc[:10, :20]  # First 10 docs, first 20 features
sns.heatmap(subset_bow, annot=True, fmt='d', cmap='Blues')
plt.title('BoW Matrix Heatmap (Subset)')
plt.xlabel('Features')
plt.ylabel('Documents')
plt.tight_layout()
plt.show()


# Create vectorizers with bigrams
bow_bigram = CountVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    max_features=200
)

tfidf_bigram = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=200
)

# Fit and transform
bow_bigram_matrix = bow_bigram.fit_transform(df['review'])
tfidf_bigram_matrix = tfidf_bigram.fit_transform(df['review'])

print("Bigram Analysis:")
print(f"BoW with bigrams shape: {bow_bigram_matrix.shape}")
print(f"TF-IDF with bigrams shape: {tfidf_bigram_matrix.shape}")

# Get bigram features
bigram_features = bow_bigram.get_feature_names_out()
bigrams_only = [feature for feature in bigram_features if ' ' in feature]

print(f"\nTotal bigrams found: {len(bigrams_only)}")
print("Sample bigrams:", bigrams_only[:10])


# Get bigram frequencies
bow_bigram_dense = bow_bigram_matrix.toarray()
bigram_frequencies = np.sum(bow_bigram_dense, axis=0)

# Create DataFrame for bigram analysis
bigram_freq_df = pd.DataFrame({
    'ngram': bigram_features,
    'frequency': bigram_frequencies,
    'type': ['bigram' if ' ' in ngram else 'unigram' for ngram in bigram_features]
}).sort_values('frequency', ascending=False)

# Show top bigrams
top_bigrams = bigram_freq_df[bigram_freq_df['type'] == 'bigram'].head(15)
print("Top 15 Bigrams:")
print(top_bigrams)

# Visualize top bigrams
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_bigrams)), top_bigrams['frequency'])
plt.xticks(range(len(top_bigrams)), top_bigrams['ngram'], rotation=45, ha='right')
plt.title('Top 15 Most Frequent Bigrams')
plt.xlabel('Bigrams')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create different vectorizers for comparison
vectorizers = {
    'BoW': CountVectorizer(lowercase=True, stop_words='english', max_features=100),
    'TF-IDF': TfidfVectorizer(lowercase=True, stop_words='english', max_features=100),
    'BoW_Bigrams': CountVectorizer(lowercase=True, stop_words='english',
                                   ngram_range=(1, 2), max_features=200),
    'TF-IDF_Bigrams': TfidfVectorizer(lowercase=True, stop_words='english',
                                      ngram_range=(1, 2), max_features=200)
}

# Store results
results = {}



# Train and evaluate models with different vectorizers
for name, vectorizer in vectorizers.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"{'='*50}")

    # Vectorize training and test data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training matrix shape: {X_train_vec.shape}")
    print(f"Test matrix shape: {X_test_vec.shape}")

    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test_vec)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'vectorizer': vectorizer,
        'classifier': classifier,
        'matrix_shape': X_train_vec.shape
    }

    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))



# Create comparison DataFrame
comparison_results = pd.DataFrame({
    'Method': list(results.keys()),
    'Accuracy': [results[method]['accuracy'] for method in results.keys()],
    'Features': [results[method]['matrix_shape'][1] for method in results.keys()]
})

print("Performance Comparison:")
print(comparison_results)

# Visualize comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy comparison
ax1.bar(comparison_results['Method'], comparison_results['Accuracy'])
ax1.set_title('Classification Accuracy by Feature Extraction Method')
ax1.set_ylabel('Accuracy')
ax1.set_xticklabels(comparison_results['Method'], rotation=45, ha='right')

# Feature count comparison
ax2.bar(comparison_results['Method'], comparison_results['Features'])
ax2.set_title('Number of Features by Method')
ax2.set_ylabel('Number of Features')
ax2.set_xticklabels(comparison_results['Method'], rotation=45, ha='right')

plt.tight_layout()
plt.show()



# Get the best performing model
best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_vectorizer = results[best_method]['vectorizer']
best_classifier = results[best_method]['classifier']

print(f"Best performing method: {best_method}")
print(f"Best accuracy: {results[best_method]['accuracy']:.4f}")

# Get feature names and importance scores
feature_names = best_vectorizer.get_feature_names_out()
feature_importance = best_classifier.feature_log_prob_

# Calculate feature importance difference (positive - negative class)
importance_diff = feature_importance[1] - feature_importance[0]

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_diff
}).sort_values('importance', ascending=False)

print("\nTop 10 features for positive sentiment:")
print(feature_importance_df.head(10))

print("\nTop 10 features for negative sentiment:")
print(feature_importance_df.tail(10))


# Plot feature importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Top positive features
top_positive = feature_importance_df.head(10)
ax1.barh(range(len(top_positive)), top_positive['importance'])
ax1.set_yticks(range(len(top_positive)))
ax1.set_yticklabels(top_positive['feature'])
ax1.set_title('Top 10 Features for Positive Sentiment')
ax1.set_xlabel('Importance Score')

# Top negative features
top_negative = feature_importance_df.tail(10)
ax2.barh(range(len(top_negative)), top_negative['importance'])
ax2.set_yticks(range(len(top_negative)))
ax2.set_yticklabels(top_negative['feature'])
ax2.set_title('Top 10 Features for Negative Sentiment')
ax2.set_xlabel('Importance Score')

plt.tight_layout()
plt.show()



# Test different parameters
parameter_tests = [
    {'max_features': 50, 'ngram_range': (1, 1)},
    {'max_features': 100, 'ngram_range': (1, 1)},
    {'max_features': 200, 'ngram_range': (1, 1)},
    {'max_features': 100, 'ngram_range': (1, 2)},
    {'max_features': 200, 'ngram_range': (1, 2)},
    {'max_features': 300, 'ngram_range': (1, 2)}
]

tuning_results = []

for params in parameter_tests:
    # Create vectorizer with current parameters
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=params['max_features'],
        ngram_range=params['ngram_range']
    )

    # Vectorize data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    tuning_results.append({
        'max_features': params['max_features'],
        'ngram_range': str(params['ngram_range']),
        'accuracy': accuracy,
        'matrix_shape': X_train_vec.shape
    })

# Display results
tuning_df = pd.DataFrame(tuning_results)
print("Parameter Tuning Results:")
print(tuning_df.sort_values('accuracy', ascending=False))




