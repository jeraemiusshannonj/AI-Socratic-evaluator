"""Build-time corpus tooling.

Kept out of the `socratic` package on purpose: nothing in here runs at
request time. The app imports `socratic`; only `scripts/build_corpus.py`
imports this.
"""
