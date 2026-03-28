import json
import math
import pickle
import string
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5


class InvertedIndex:
    CACHE_DIR = Path("cache")
    INDEX_PATH = CACHE_DIR / "index.pkl"
    DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
    TF_PATH = CACHE_DIR / "term_frequencies.pkl"

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        stopwords = get_stopwords()
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        return sorted(self.index.get(term.lower(), set()))

    def get_tf(self, doc_id, term):
        doc_id_ = int(doc_id)
        stopwords = get_stopwords()
        tokens = tokenize(term, stopwords)
        if len(tokens) > 1:
            raise ValueError("Too many tokens")
        token = tokens[0]
        tf = self.term_frequencies[doc_id_]

        count = tf[token]
        return count

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    def get_bm25_idf(self, term: str) -> float:
        stopwords = get_stopwords()
        tokens = tokenize(term, stopwords)
        if len(tokens) > 1:
            raise ValueError("Too many tokens")
        documentFrequency = self.get_documents(tokens[0])
        bm25_idf = math.log(
            (len(self.docmap) - len(documentFrequency) + 0.5)
            / (len(documentFrequency) + 0.5)
            + 1
        )
        return bm25_idf

    def build(self):
        data = get_movies()
        for m in data["movies"]:
            movietext = f"{m['title']} {m['description']}"
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], movietext)

    def save(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.DOCMAP_PATH, "wb") as g:
            pickle.dump(self.docmap, g)

        with open(self.TF_PATH, "wb") as t:
            pickle.dump(self.term_frequencies, t)

    def load(self):
        if not self.DOCMAP_PATH.exists():
            raise FileNotFoundError(
                f"Could not find the index file at: {self.DOCMAP_PATH}"
            )
        if not self.INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Could not find the index file at: {self.INDEX_PATH}"
            )

        if not self.TF_PATH.exists():
            raise FileNotFoundError(f"Could not find the index file at: {self.TF_PATH}")

        with open(self.DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.INDEX_PATH, "rb") as g:
            self.index = pickle.load(g)

        with open(self.TF_PATH, "rb") as t:
            self.term_frequencies = pickle.load(t)


def bm25_tf_command(doc_id, term, k1=BM25_K1):
    inverted_index = InvertedIndex()
    inverted_index.load()
    bm25_tf = inverted_index.get_bm25_tf(doc_id, term, k1)
    return bm25_tf

def idf_command(query: str):
    inverted_index = InvertedIndex()
    stopwords = get_stopwords()
    tokens = tokenize(query, stopwords)
    if len(tokens) > 1:
        raise ValueError("Too many tokens")

    inverted_index.load()
    docs = inverted_index.get_documents(tokens[0])
    docmap = inverted_index.docmap
    idf = math.log((len(docmap) + 1) / (len(docs) + 1))
    return idf


def bm25_idf_command(query: str):
    inverted_index = InvertedIndex()
    inverted_index.load()
    bm25_idf = inverted_index.get_bm25_idf(query)
    return bm25_idf


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return []

    stopwords = get_stopwords()
    tokens = tokenize(query, stopwords)

    results = []
    seen = set()

    for token in tokens:
        set_of_ids = idx.get_documents(token)
        for id_ in set_of_ids:
            if id_ in seen:
                continue
            seen.add(id_)
            results.append(idx.docmap[id_])

            if len(results) >= limit:
                return results

    return results


def get_movies():
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        return data


def preprocess_text(text):
    punct = string.punctuation
    table = str.maketrans("", "", punct)
    return text.translate(table)


@lru_cache(maxsize=1)
def get_stopwords():
    with open("data/stopwords.txt", "r") as f:
        return tuple(f.read().splitlines())


def tokenize(text: str, stopwords: tuple[str, ...]) -> list[str]:
    text = preprocess_text(text.lower())
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in stopwords]
    return stem_tokens(tokens)


def stem_tokens(tokens: list[str]):
    return [stemmer.stem(word) for word in tokens]


def find_by_title(query):
    found_movies = []
    stopwords = get_stopwords()

    data = get_movies()

    query_tokens = tokenize(query, stopwords)

    for movie in data["movies"]:
        title_tokens = tokenize(movie["title"], stopwords)

        if is_token_in_tokens(query_tokens, title_tokens):
            found_movies.append(movie)

        if len(found_movies) == 5:
            return found_movies

    return found_movies


def is_token_in_tokens(tokens_a, tokens_b):
    for token_a in tokens_a:
        for token_b in tokens_b:
            if token_a in token_b:
                return True
    return False
