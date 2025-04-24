from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence-BERT model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def semantic_similarity(query, doc):
    """
    Calculate semantic similarity between query and document using Sentence-BERT.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    doc_embedding = sentence_model.encode(doc, convert_to_tensor=True)
    return cosine_similarity([query_embedding.cpu().numpy()], [doc_embedding.cpu().numpy()])[0][0]


def extract_features(query, doc, vectorizer=None):
    features = []

    # TF-IDF similarity
    try:
        tfidf = vectorizer.transform([query, doc])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        similarity = 0.0
    features.append(similarity)

    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc)

    # Semantic similarity
    features.append(semantic_similarity(query, doc))

    # Negation handling
    negation_terms = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor',
                      'nobody', 'without', 'isn’t', 'wasn’t', 'aren’t', 'weren’t', 'can’t',
                      'couldn’t', 'won’t', 'wouldn’t', 'don’t', 'doesn’t', 'didn’t', 'hasn’t',
                      'haven’t', 'hadn’t', 'unlikely', 'impossible', 'untrue']
    q_neg = sum(t in negation_terms for t in query_tokens)
    d_neg = sum(t in negation_terms for t in doc_tokens)
    features.append(abs(q_neg - d_neg))

    # N-gram overlap
    features.append(kgram_overlap(query_tokens, doc_tokens, k=2))  # Bigram overlap
    features.append(kgram_overlap(query_tokens, doc_tokens, k=3))  # Trigram overlap

    # Query and document length features
    features.append(len(query_tokens) / max(len(doc_tokens), 1))  # Normalized length ratio
    features.append(abs(len(query_tokens) - len(doc_tokens)))    # Absolute length difference

    return features


def kgram_overlap(query_tokens, doc_tokens, k):
    """
    Calculate the k-gram overlap between query tokens and document tokens.
    """
    query_kgrams = set(zip(*[query_tokens[i:] for i in range(k)]))
    doc_kgrams = set(zip(*[doc_tokens[i:] for i in range(k)]))
    return len(query_kgrams & doc_kgrams) / max(len(query_kgrams), len(doc_kgrams), 1)


class RankingModel:
    def __init__(self):
        self.model = None  # Model will be set after GridSearchCV
        self.vectorizer = None
        self.trained = False

    def build_vectorizer(self, instances):
        texts = []
        for inst in instances:
            texts.extend([inst["q1"], inst["q2"], inst["doc1"], inst["doc2"]])
        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts)
        return vectorizer

    def train(self, instances):
        self.vectorizer = self.build_vectorizer(instances)

        x, y = [], []

        for inst in instances:
            x.append(extract_features(inst["q1"], inst["doc1"], self.vectorizer))
            y.append(1)
            x.append(extract_features(inst["q1"], inst["doc2"], self.vectorizer))
            y.append(0)
            x.append(extract_features(inst["q2"], inst["doc1"], self.vectorizer))
            y.append(0)
            x.append(extract_features(inst["q2"], inst["doc2"], self.vectorizer))
            y.append(1)

        # Define the parameter grid for XGBClassifier
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        # Perform grid search to find the best model
        grid_search = GridSearchCV(
            XGBClassifier(eval_metric='logloss'),  # Removed use_label_encoder
            param_grid,
            cv=5
        )
        grid_search.fit(x, y)

        # Store the best model and update the trained flag
        self.model = grid_search.best_estimator_
        self.trained = True

    def score(self, query, doc):
        if not self.trained or self.model is None:
            raise ValueError("The model has not been trained yet.")
        features = extract_features(query, doc, self.vectorizer)
        return self.model.predict_proba([features])[0][1]

    def rank_documents(self, query, doc1, doc2):
        return self.score(query, doc1), self.score(query, doc2)


def evaluate(instances, model):
    correct = 0
    total = len(instances) * 2  # Each instance has two comparisons

    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]

        # Evaluate q1 with d1 and d2
        s1_d1, s1_d2 = model.rank_documents(q1, d1, d2)
        if s1_d1 > s1_d2:
            correct += 1

        # Evaluate q2 with d1 and d2
        s2_d1, s2_d2 = model.rank_documents(q2, d1, d2)
        if s2_d2 > s2_d1:
            correct += 1

    return (correct / total) * 100


def main():
    print("Loading dataset...")
    dataset = load_dataset("orionweller/nevir")
    train_set = list(dataset["train"])
    val_set = list(dataset["validation"])
    test_set = list(dataset["test"])

    print(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test instances.\n")

    model = RankingModel()

    print("Training model... Please wait.")
    model.train(train_set)
    print("Training complete.\n")

    print("Evaluating model...")
    print(f"Train accuracy: {evaluate(train_set, model):.2f}%")
    print(f"Validation accuracy: {evaluate(val_set, model):.2f}%")
    print(f"Test accuracy: {evaluate(test_set, model):.2f}%")


if __name__ == "__main__":
    main()
