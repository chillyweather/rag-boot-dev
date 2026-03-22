import json
import string
import pickle

from pathlib import Path
from functools import lru_cache
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
DEFAULT_SEARCH_LIMIT = 5


class InvertedIndex:
    CACHE_DIR = Path("cache")
    INDEX_PATH = CACHE_DIR / "index.pkl"
    DOCMAP_PATH = CACHE_DIR / "docmap.pkl"

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text):
        stopwords = get_stopwords()
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term.lower(), set()))

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

    def load(self):
        if not self.DOCMAP_PATH.exists():
            raise FileNotFoundError(
                f"Could not find the index file at: {self.DOCMAP_PATH}"
            )
        if not self.INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Could not find the index file at: {self.INDEX_PATH}"
            )

        with open(self.DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.INDEX_PATH, "rb") as g:
            self.index = pickle.load(g)


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
    tokens = text.split(" ")
    tokens = [t for t in tokens if t not in stopwords]
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
