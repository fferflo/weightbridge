from .base import matcher

@matcher
def match_unique_leafs(state, match):
    yield match