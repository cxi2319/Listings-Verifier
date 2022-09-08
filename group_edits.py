"""Module that contains the group by edit type function
"""

import listings_verifier.implementation_scripts.levenshtein_dp as lev


def groupby_edit_type(token1: list[str], token2: list[str]) -> list[list]:
    """Calls the Levenshtein distance function and takes the output list of tuples containing all
    edits from the Levenshtein function, and groups them by edit type in three individual lists -
    deletions, insertions and swaps. Then returns a master list containing tuples of the token pairs
    in the order of deletions, insertions, and swaps.

    Args:
        token1 (List[str]): List of individual Yext tokens that collectively make up a token string
        token2 (List[str]): List of individual publisher tokens that collectively make up a token
                            string

    Returns:
        list[list]: List of tuples containing edit tokens in the following order: deletion tokens,
                    insertion tokens, and swap tokens.
    """

    # initialize individual edit type lists
    delete_list = []
    insert_list = []
    swap_list = []

    # initialize master list of lists to return, containing the grouped-by edit lists appended to
    # each other
    return_list = []

    # call Levenshtein distance function on the Yext and publisher tokenized strings to create a
    # list of tuples containing edit type and token pairs
    edit_list = lev.levenshtein_distance_dp(token1, token2)

    # iterate through each tuple in the edit_list and check the type of edit
    for item in edit_list:
        # track deletions
        if item[0] == "delete":
            delete_list.append(item[1])
        # track insertions
        elif item[0] == "insert":
            insert_list.append(item[1])
        # track swaps
        elif item[0] == "swap":
            swap_list.append(item[1])

    # append the individual edit lists to the master return list
    return_list.append(delete_list)
    return_list.append(insert_list)
    return_list.append(swap_list)

    return return_list
