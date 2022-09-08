"""Module that contains levenshtein distance function
"""

import numpy as np


def levenshtein_distance_dp(token1: list[str], token2: list[str]) -> list[tuple]:
    """A Levenshtein distance dynamic programming function that returns an optimum path taken to
    transform the previous word into the new word. Takes in two lists of tokens, where token1 is the
    Yext token string, and token2 is the publisher token string, and returns a list of tuples
    containing the edit type and token pairs consisting of the Yext token and corresponding
    publisher token.

    Args:
        token1 (list): List of individual Yext tokens that collectively make up a token string
        token2 (list): List of individual publisher tokens that collectively make up a token string

    Returns:
        list: A list of tuples containing token pairs consisting of edit type, and the Yext token
              and corresponding publisher token. Using the example strings of ["token1", "token2"],
              ["token1", "token3"], we can expect to see the following:

                [('no_change', ('token1', 'token1')), ('swap', ('token2', 'token3'))]

    """
    # initialize the distances and backpointers matrices.
    # every entry in distances[i][j] indicates the minimum Levenshtein distance from token1[:i] to
    # token2[:j]
    distances = np.zeros((len(token1) + 1, len(token2) + 1))
    # every entry in backpointers[i][j] indicates the previous state on the optimum path from
    # token1[:i] to token2[:j].
    # schema is ((previous i, previous j), (step type, (token1 element, token2 element)))
    backpointers = np.zeros((len(token1) + 1, len(token2) + 1)).tolist()

    for t1 in range(len(token1) + 1):
        # set every distance in the first column equal to the current word size
        distances[t1][0] = t1
        # set every backpointer in the first column to point to the row above and delete the current
        # element
        backpointers[t1][0] = (t1 - 1, 0, ("delete", (token1[t1 - 1], "<na>")))

    for t2 in range(len(token2) + 1):
        # set every distance in the first row equal to the current word size
        distances[0][t2] = t2
        # set every backpointer in the first row to point to the column to the left (prev.) and
        # insert the current element
        backpointers[0][t2] = (0, t2 - 1, ("insert", ("<na>", token2[t2 - 1])))

    # overwrite the first row and column to indicate that there is no change
    backpointers[0][0] = (-1, -1, ("no_change", ("<na>", "<na>")))

    # a is the distance to the left node(insert)
    a = 0
    # b is the distance to the top node(delete)
    b = 0
    # c is the distance to the top-left node(swap)
    c = 0

    # iterate through every element of the matrix
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            # if it's the same element, there is no change and set current distance equal to top-
            # left distance and set backpointer to point to top-left backpointer
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
                backpointers[t1][t2] = (
                    t1 - 1,
                    t2 - 1,
                    ("no_change", (token1[t1 - 1], token2[t2 - 1])),
                )
            # otherwise, determine the distances a,b,c
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                # if a is the min. distance, set the distance to current element = a+1, and set the
                # optimal path to pass through that node
                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                    backpointers[t1][t2] = (
                        t1,
                        t2 - 1,
                        ("insert", (token1[t1 - 1], token2[t2 - 1])),
                    )

                # if b is the min. distance, set the distance to current element = a+1, and set the
                # optimal path to pass through that node
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                    backpointers[t1][t2] = (
                        t1 - 1,
                        t2,
                        ("delete", (token1[t1 - 1], token2[t2 - 1])),
                    )

                # if c is the min. distance, set the distance to current element = a+1, and set the
                # optimal path to pass through that node
                else:
                    distances[t1][t2] = c + 1
                    backpointers[t1][t2] = (
                        t1 - 1,
                        t2 - 1,
                        ("swap", (token1[t1 - 1], token2[t2 - 1])),
                    )

    def follow_backpointers(end):
        x, y, (step, (t1, t2)) = end
        previous_state = backpointers[x][y]
        # if we're at the beginning, return the first element of the matrix
        if previous_state[0] == previous_state[1] == -1:
            return [(step, (t1, t2))]
        # otherwise, return follow the backpointer to the previous state, concatenating the element
        # with a recursive call to the follow_backpointers() on the previous state
        else:
            return follow_backpointers(previous_state) + [(step, (t1, t2))]

    return follow_backpointers(backpointers[-1][-1])
