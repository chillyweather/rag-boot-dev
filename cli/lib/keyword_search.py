import json
import string
from functools import lru_cache

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text):
        stopwords = get_stopwords()
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_document(self, term):
        return sorted(self.index.get(term.lower(), set()))

    def build(self):
        movies = get_movies()
        for m in movies:
            movietext = f"{m['title']} {m['description']}"
            self.docmap[m[id], m]
            self.__add_document(m[id], movietext)


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
