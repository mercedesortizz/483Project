from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import re
import numpy as np


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def extract_features(query, doc, vectorizer=None):
    features = []

    # TF-IDF similarity
    try:
        tfidf = vectorizer.transform([query, doc])
        sim = (tfidf[0] @ tfidf[1].T).toarray()[0][0]
    except:
        sim = 0.0
    features.append(sim)

    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc)

    negation_terms = {
        'no', 'not', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', 'nobody',
        'without', 'isn’t', 'wasn’t', 'aren’t', 'weren’t', 'can’t', 'couldn’t', 'won’t',
        'wouldn’t', 'don’t', 'doesn’t', 'didn’t', 'hasn’t', 'haven’t', 'hadn’t',
        'unlikely', 'impossible', 'untrue'
    }
    neg_prefixes = ['non', 'in', 'im', 'dis', 'a']
    comparative_terms = ['less', 'more', 'fewer', 'greater', 'increased', 'decreased', 'higher', 'lower']
    more_terms = ["unprecedented", "increased", "more", "excessive"]
    less_terms = ["only", "less", "fewer", "reduced"]

    # Negation features
    q_neg = sum(t in negation_terms for t in query_tokens)
    d_neg = sum(t in negation_terms for t in doc_tokens)
    q_neg_pref = sum(t.startswith(p) for t in query_tokens for p in neg_prefixes)
    d_neg_pref = sum(t.startswith(p) for t in doc_tokens for p in neg_prefixes)
    features.append(q_neg + q_neg_pref)
    features.append(d_neg + d_neg_pref)
    features.append(abs((q_neg + q_neg_pref) - (d_neg + d_neg_pref)))

    # Comparative
    features.append(sum(t in comparative_terms for t in query_tokens))
    features.append(sum(t in more_terms for t in doc_tokens if "more" in query_tokens))
    features.append(sum(t in less_terms for t in doc_tokens if "less" in query_tokens))

    # Token overlap
    overlap = len(set(query_tokens) & set(doc_tokens)) / max(len(set(query_tokens)), 1)
    features.append(overlap)

    # Length features
    features.append(len(query_tokens) / max(len(doc_tokens), 1))
    features.append(abs(len(query_tokens) - len(doc_tokens)))

    return features


def extract_pairwise_features(query, doc1, doc2, vectorizer):
    f1 = extract_features(query, doc1, vectorizer)
    f2 = extract_features(query, doc2, vectorizer)
    return list(np.subtract(f1, f2))  # doc1 - doc2


class RankingModel:
    def __init__(self):
        self.model = XGBClassifier(eval_metric='logloss')
        self.vectorizer = None

    def build_vectorizer(self, instances):
        texts = []
        for inst in instances:
            texts += [inst["q1"], inst["q2"], inst["doc1"], inst["doc2"]]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.9, min_df=2)
        vectorizer.fit(texts)
        return vectorizer

    def train(self, instances):
        self.vectorizer = self.build_vectorizer(instances)
        x, y = [], []

        for inst in instances:
            # q1 prefers doc1 over doc2 → label = 1
            x.append(extract_pairwise_features(inst["q1"], inst["doc1"], inst["doc2"], self.vectorizer))
            y.append(1)

            # doc2 over doc1 → label = 0 (reversed input)
            x.append(extract_pairwise_features(inst["q1"], inst["doc2"], inst["doc1"], self.vectorizer))
            y.append(0)

            # q2 prefers doc2 over doc1 → label = 1
            x.append(extract_pairwise_features(inst["q2"], inst["doc2"], inst["doc1"], self.vectorizer))
            y.append(1)

            # reversed → doc1 over doc2 → label = 0
            x.append(extract_pairwise_features(inst["q2"], inst["doc1"], inst["doc2"], self.vectorizer))
            y.append(0)

        self.model.fit(np.array(x), np.array(y))


    def predict_preference(self, query, doc1, doc2):
        features = extract_pairwise_features(query, doc1, doc2, self.vectorizer)
        return int(self.model.predict([features])[0])  # 1 if doc1 preferred over doc2


def evaluate(instances, model):
    correct = 0
    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]

        # q1 should prefer d1, q2 should prefer d2
        if model.predict_preference(q1, d1, d2) == 1 and model.predict_preference(q2, d2, d1) == 1:
            correct += 1

    return 100 * correct / len(instances)

def main():
    print("Loading da dataset...")
    dataset = load_dataset("orionweller/nevir")
    train_set = list(dataset["train"])
    val_set = list(dataset["validation"])
    test_set = list(dataset["test"])

    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}\n")

    model = RankingModel()

    print("Training model...Be patient damn it!")
    model.train(train_set)
    print("Training complete.\n")

    print("Evaluating model...be grateful for the results!")
    print(f"Train accuracy: {evaluate(train_set, model):.2f}%")
    print(f"Validation accuracy: {evaluate(val_set, model):.2f}%")
    print(f"Test accuracy: {evaluate(test_set, model):.2f}%")


if __name__ == "__main__":
    main()
