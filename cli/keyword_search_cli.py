#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    InvertedIndex,
    bm25_idf_command,
    bm25_tf_command,
    idf_command,
    search_command,
)

BM25_K1 = 1.5

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Search for term frequency")
    tf_parser.add_argument("doc_id", type=str, help="Document id")
    tf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Search for ")
    tfidf_parser.add_argument("doc_id", type=str, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
        )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    idf_parser = subparsers.add_parser("idf", help="Search how rare the term")
    idf_parser.add_argument("term", type=str, help="Search term")

    subparsers.add_parser("build", help="Build reverse index of the movies")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: ", args.query)
            movies = search_command(args.query)
            if movies:
                for i, value in enumerate(movies):
                    print(f"{i + 1}. {value['title']}")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            idf = idf_command(args.term)
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
