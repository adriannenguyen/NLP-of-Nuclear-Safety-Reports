import pandas as pd
import spacy
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load spaCy's English model globally
nlp = spacy.load("en_core_web_sm")

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function for text cleaning and lowercasing
def clean_text(text, nlp):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and token.text not in STOP_WORDS]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to extract features from text using seed words
def extract_features(text, seed_words):
    features = {}
    for trait, words in seed_words.items():
        lemmatized_tokens = [token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_stop]
        features[trait] = any(lemma in words for lemma in lemmatized_tokens)
    return features

def evaluate_model(X, y, model):
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Display cross-validation results
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Accuracy: {:.2f}".format(cv_scores.mean()))

def train_model(X_train, X_test, y_train, y_test, model):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=1)

    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F-1 Score: {f1:.2f}")

def main():
    # Load the CSV file into a Pandas DataFrame
    file_path = "/workspaces/NLP-of-Nuclear-Safety-Reports/Traits Dataset - Sheet1.csv"
    df = pd.read_csv(file_path)

    # Apply text cleaning to the 'Issue Statement' column
    df["Issue Statement"] = df["Issue Statement"].apply(lambda x: clean_text(x, nlp))

    # Define primary and secondary seed words for each safety trait    
    seed_words = {
        "Personal Accountability": ["responsibility", "accountability", "help", "support", "trained", "qualified", "understand", "complete", "involvement"],
        "Questioning Attitude": ["complacency", "complacent", "challenge", "error", "hazard", "caution", "discrepancy", "anomaly", "assumption", "question", "uncertain", "unknown", "risk", "trend", "unexpected", "unclear", "degrading", "aging"],
        "Effective Safety Communication": ["communication", "licensee", "event", "report", "documentation", "request", "LER", "information", "safety", "prompt", "share", "respond", "listen", "concern", "expectation", "clear"],
        "Leadership Safety Values and Actions": ["leadership", "management", "leader", "owner", "ownership", "program", "guidance", "policy", "resource", "staffing", "oversight", "reinforce", "priority", "plan", "delegate", "align", "define", "manage", "resolve", "address", "translate", "funding", "implementation", "violation"],
        "Decision Making": ["thorough", "conservative", "systematic", "consistent", "process", "choice", "consequence", "authority", "future", "timely", "executive", "senior"],
        "Respectful Work Environment": ["trust", "respect", "opinion", "dignity", "fair", "disagree", "receptive", "valuable", "tolerate", "value", "insight", "perspective", "collaboration", "conflict", "listening"],
        "Continuous Learning": ["learn", "training", "assessment", "improve", "performance", "scrutiny", "monitor", "adopt", "idea", "benchmarking", "knowledge", "competent", "skills", "develop", "acquire"],
        "Problem Identification and Resolution": ["identify", "corrective", "action", "issue", "yellow", "red", "prevent", "foreign", "poor", "inadequate", "degraded", "evaluation", "problem", "cause", "root", "investigation", "investigate", "recommendation", "resolution", "mitigate"],
        "Environment for Raising Concerns": ["environment", "fear", "harassment", "discrimination", "promote", "severity", "failure", "submit", "report", "expired", "raise"],
        "Work Processes": ["engineering", "control", "activity", "contingency", "production", "schedule", "work", "margin", "operate", "maintain", "maintenance", "procedure", "package", "accurate", "current", "backlog", "instruction", "operation", "design", "requirement", "standard"]
    }
    
    secondary_seed_words = {
        "Personal Accountability_secondary": ["ownership", "dedication", "commitment", "reliability", "dependability", "self-discipline", "initiative", "reponsiveness", "proactive"],
        "Questioning Attitude_secondary": ["vigilance", "doubt", "scrutiny", "curiosity", "inquisitiveness", "skepticism", "critical", "examination", "alertness", "vigilant"],
        "Effective Safety Communication_secondary": ["clarity", "transparency", "articulation", "cooperation", "collaboration", "engagement", "feedback", "openness", "expressiveness", "interaction"],
        "Leadership Safety Values and Actions_secondary": ["integrity", "ethical", "trustworthy", "fairness", "responsiveness", "accountability", "transparency", "consistency", "visionary", "motivational"],
        "Decision Making": ["analysis", "deliberation_secondary", "judicious", "logical", "reasoned", "informed", "well-thought-out", "strategic", "sensible", "considered"],
        "Respectful Work Environment_secondary": ["tolerance", "inclusivity", "equality", "courtesy", "appreciation", "empathy", "cooperation", "harmony", "understanding", "courteous"],
        "Continuous Learning_secondary": ["adaptability", "flexibility", "innovation", "upskilling", "knowledge-sharing", "exploration", "growth mindset", "persistence", "curiosity", "development"],
        "Problem Identification and Resolution_secondary": ["analytical", "troubleshooting", "diagnostic", "proactive", "systematic", "strategic", "efficient", "solutions-oriented", "resilient", "resourceful"],
        "Environment for Raising Concerns_secondary": ["trust", "confidentiality", "safety", "openness", "non-retaliation", "encouragement", "whistleblower protection", "supportive", "secure", "confidential"],
        "Work Processes_secondary": ["efficiency", "optimization", "streamlining", "precision", "adherence", "standardization", "efficacy", "quality", "safety", "consistency"]
    }

    # Combine primary and secondary seed words
    combined_seed_words = {**seed_words, **secondary_seed_words}

    # Apply the extract_features function to the 'Issue Statement' column
    df["Features"] = df["Issue Statement"].apply(lambda x: extract_features(x, combined_seed_words))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["Issue Statement"], df["Safety trait(s)"], test_size=0.2, random_state=42
    )

    # Build a pipeline with a CountVectorizer and a Multinomial Naive Bayes classifier
    # nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    # train_and_evaluate_model(X_train, X_test, y_train, y_test, nb_model)
    # Accuracy: 0.50
    # Precision: 0.35
    # Recall: 0.50
    # F-1 Score: 0.41

    # Build a pipeline with a TF-IDF Vectorizer and a RandomForestClassifier
    rf_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    print("\nRandom Forest Model:")
    train_model(X_train, X_test, y_train, y_test, rf_model)
    evaluate_model(X_train, y_train, model=rf_model)
    # Accuracy: 0.58
    # Precision: 0.77
    # Recall: 0.58
    # F-1 Score: 0.46


if __name__ == "__main__":
    main()