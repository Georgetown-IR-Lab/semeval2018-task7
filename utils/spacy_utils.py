import spacy.tokens


def get_token_root_distance(
        token: spacy.tokens.Token,
        memoization_map: dict=None
) -> int:
    """Returns the distance of a token from the root of the parse tree."""

    if memoization_map is None:
        memoization_map = {}

    if not token.i in memoization_map:
        if token.head == token:
            memoization_map[token.i] = 0
        else:
            memoization_map[token.i] = 1 + get_token_root_distance(
                token.head, memoization_map
            )

    return memoization_map[token.i]
