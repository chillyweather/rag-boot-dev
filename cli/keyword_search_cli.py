#!/usr/bin/env python3

import argparse
from lib.keyword_search import find_by_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: ", args.query)
            movies = find_by_title(args.query)
            if movies:
                for i, value in enumerate(movies):
                    print(f"{i + 1}. {value['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
