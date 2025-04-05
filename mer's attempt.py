from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import re


def tokenize(text):
    """Lowercase and split text into words using regex."""
    return re.findall(r'\b\w+\b', text.lower())


def extract_features(query, doc):
    """
    Extract numerical features from a query-document pair.
    Features:
      - TF-IDF similarity
      - Count of negation and comparative terms
      - Term matches for "more"/"less" cues
      - Word overlap
    """
    features = []

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([query, doc])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        similarity = 0.0
    features.append(similarity)

    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc)
    query_text = ' '.join(query_tokens)
    doc_text = ' '.join(doc_tokens)

    # Term lists
    negation_terms = ['not', 'no', 'never']
    comparative_terms = ['less', 'more', 'fewer', 'greater', 'increased',
                         'decreased', 'higher', 'lower']
    more_terms = ["unprecedented", "increased", "more", "excessive"]
    less_terms = ["only", "less", "fewer", "reduced"]

    # Negation/comparative counts
    features.append(sum(1 for term in negation_terms if term in query_tokens))
    features.append(sum(1 for term in comparative_terms if term in query_tokens))

    # More/less cues in doc if present in query
    features.append(sum(1 for term in more_terms if "more" in query_tokens and term in doc_text))
    features.append(sum(1 for term in less_terms if "less" in query_tokens and term in doc_text))

    # Word overlap
    overlap = len(set(query_tokens) & set(doc_tokens)) / max(len(set(query_tokens)), 1)
    features.append(overlap)

    return features


class RankingModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced')
        self.trained = False

    def train(self, instances):
        """Train the model using the training instances."""
        x, y = [], []

        for inst in instances:
            # q1: doc1 is relevant
            x.append(extract_features(inst["q1"], inst["doc1"]))
            y.append(1)
            x.append(extract_features(inst["q1"], inst["doc2"]))
            y.append(0)
            # q2: doc2 is relevant
            x.append(extract_features(inst["q2"], inst["doc1"]))
            y.append(0)
            x.append(extract_features(inst["q2"], inst["doc2"]))
            y.append(1)

        self.model.fit(x, y)
        self.trained = True

    def score(self, query, doc):
        """Return the relevance probability for a query-document pair."""
        features = extract_features(query, doc)
        return self.model.predict_proba([features])[0][1]

    def rank_documents(self, query, doc1, doc2):
        """Return scores for both documents given a query."""
        return self.score(query, doc1), self.score(query, doc2)


def evaluate(instances, model):
    """
    Compute ranking accuracy for the dataset.
    For each instance:
      - q1: doc1 should score higher than doc2
      - q2: doc2 should score higher than doc1
    """
    correct = 0
    total = len(instances) * 2

    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]

        s1_d1, s1_d2 = model.rank_documents(q1, d1, d2)
        s2_d1, s2_d2 = model.rank_documents(q2, d1, d2)

        if s1_d1 > s1_d2:
            correct += 1
        if s2_d2 > s2_d1:
            correct += 1

    return (correct / total) * 100


def main():
    print("Loading da dataset...")
    dataset = load_dataset("orionweller/nevir")
    train_set = list(dataset["train"])
    val_set = list(dataset["validation"])
    test_set = list(dataset["test"])

    print(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test instances.\n")

    model = RankingModel()

    print("Training da model...")
    model.train(train_set)
    print("Training complete! Yippee\n")

    print("Evaluating model...please hold")
    print(f"Train accuracy: {evaluate(train_set, model):.2f}%")
    print(f"Validation accuracy: {evaluate(val_set, model):.2f}%")
    print(f"Test accuracy: {evaluate(test_set, model):.2f}%")


if __name__ == "__main__":
    main()
