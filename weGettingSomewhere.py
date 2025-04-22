from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        similarity = 0.0
    features.append(similarity)

    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc)
    query_text = ' '.join(query_tokens)
    doc_text = ' '.join(doc_tokens)

    # Lexicons
    negation_terms = [
        'no', 'not', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor',
        'nobody', 'without', 'isn’t', 'wasn’t', 'aren’t', 'weren’t', 'can’t',
        'couldn’t', 'won’t', 'wouldn’t', 'don’t', 'doesn’t', 'didn’t', 'hasn’t',
        'haven’t', 'hadn’t', 'unlikely', 'impossible', 'untrue'
    ]
    negation_prefixes = ['non', 'in', 'im', 'dis', 'a']
    comparative_terms = ['less', 'more', 'fewer', 'greater', 'increased', 'decreased', 'higher', 'lower']
    more_terms = ["unprecedented", "increased", "more", "excessive"]
    less_terms = ["only", "less", "fewer", "reduced"]

    # Counts
    q_neg = sum(t in negation_terms for t in query_tokens)
    d_neg = sum(t in negation_terms for t in doc_tokens)

    q_neg_pref = sum(t.startswith(p) for t in query_tokens for p in negation_prefixes)
    d_neg_pref = sum(t.startswith(p) for t in doc_tokens for p in negation_prefixes)

    features.append(q_neg + q_neg_pref)
    features.append(d_neg + d_neg_pref)

    # Negation mismatch
    features.append(abs((q_neg + q_neg_pref) - (d_neg + d_neg_pref)))

    # Comparative clues
    features.append(sum(t in comparative_terms for t in query_tokens))

    # Query contains "more" + doc supports more-ness
    features.append(sum(t in more_terms for t in doc_tokens if "more" in query_tokens))
    features.append(sum(t in less_terms for t in doc_tokens if "less" in query_tokens))

    # Lexical overlap
    overlap = len(set(query_tokens) & set(doc_tokens)) / max(len(set(query_tokens)), 1)
    features.append(overlap)

    # Normalized lengths
    features.append(len(query_tokens) / max(len(doc_tokens), 1))
    features.append(abs(len(query_tokens) - len(doc_tokens)))

    return features


class RankingModel:
    def __init__(self):
        self.model = XGBClassifier(eval_metric='logloss')
        self.vectorizer = None
        self.trained = False

    def build_vectorizer(self, instances):
        texts = []
        for inst in instances:
            texts.extend([inst["q1"], inst["q2"], inst["doc1"], inst["doc2"]])
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, stop_words="english")
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

        self.model.fit(np.array(x), np.array(y))
        self.trained = True

    def score(self, query, doc):
        features = extract_features(query, doc, self.vectorizer)
        return self.model.predict_proba([features])[0][1]

    def rank_documents(self, query, doc1, doc2):
        return self.score(query, doc1), self.score(query, doc2)


def evaluate(instances, model):
    correct_pairs = 0
    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]
        s1_d1, s1_d2 = model.rank_documents(q1, d1, d2)
        s2_d1, s2_d2 = model.rank_documents(q2, d1, d2)
        # QUESTION FOR BLANCO, might have to remove the Or
        if (s1_d1 > s1_d2 and s2_d2 > s2_d1) or (s1_d2 > s1_d1 and s2_d1 > s2_d2):
            correct_pairs += 1
    return (correct_pairs / len(instances)) * 100


def main():
    print("Loading da dataset...")
    dataset = load_dataset("orionweller/nevir")
    train_set = list(dataset["train"])
    val_set = list(dataset["validation"])
    test_set = list(dataset["test"])

    print(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test instances.\n")

    model = RankingModel()

    print("Training model.... Be patient damn it!")
    model.train(train_set)
    print("Training complete.\n")

    print("Evaluating model.... be grateful for the results!")
    print(f"Train accuracy: {evaluate(train_set, model):.2f}%")
    print(f"Validation accuracy: {evaluate(val_set, model):.2f}%")
    print(f"Test accuracy: {evaluate(test_set, model):.2f}%")


if __name__ == "__main__":
    main()
