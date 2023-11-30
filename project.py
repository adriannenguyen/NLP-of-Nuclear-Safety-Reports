import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Load the CSV file into a Pandas DataFrame
file_path = "/workspaces/NLP-of-Nuclear-Safety-Reports/Traits Dataset - Sheet1.csv"
df = pd.read_csv(file_path)

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Define seed words for each safety trait
seed_words = {
    "Personal Accountability": [""],
    "Questioning Attitude": [""],
    "Effective Safety Communication": [""],
    "Leadership Safety Values and Actions": [""],
    "Decision Making": [""],
    "Respectful Work Environment": [""],
    "Continuous Learning": [""],
    "Problem Identification and Resolution": [""],
    "Environment for Raising Concerns": [""],
    "Work Processes": [""]
}

# Function to extract features from text using seed words
def extract_features(text):
    doc = nlp(text)
    features = {}
    for trait, words in seed_words.items():
        features[trait] = any(word in doc.text for word in words)
    return features

# Apply the extract_features function to the 'Issue Statement' column
df['Features'] = df['Issue Statement'].apply(extract_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Issue Statement'], df['Safety Trait(s)'], test_size=0.2, random_state=42
)

# Build a pipeline with a CountVectorizer and a Multinomial Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")