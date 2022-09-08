"""
Module to generate the token-level edit probabilities, as well as overall Yext/publisher probabilities, and run comparisons on the two strings

"""
import Levenshtein

import listings_verifier.implementation_scripts.levenshtein_dp as lev


def get_insert_prob(word: str, insertion_prob_dict: dict) -> float:
    """Takes an insertion token and the dictionary of all insertion token probabilities, and returns
    a probability value to be used in the numerator calculation in the compare function.

    Args:
        word (str): Insertion token
        get_insertions_dict (dict): Dictionary containing token insertions and token probability as
        the key-value pair.

    Returns:
        float: Probability value of the arg insertion token appearing relative to all insertion
        tokens.
    """

    # Check to see if the token has been observed in the dataset of token insertions
    if word not in insertion_prob_dict:
        # We want to return a very improbable value
        return 1 / len(word)  # already confirmed word != None

    return insertion_prob_dict[word]


def get_delete_prob(word: str, deletion_prob_dict: dict) -> float:
    """Takes a deletion token and the dictionary of all deletion token probabilities, and returns a
    probability value to be used in the numerator calculation in the compare function.

    Args:
        word (str): Deletion token
        deletion_prob_dict (dict): Dictionary containing token deletions and token probability as
        the key-value pair.

    Returns:
        float: Probability value of the arg deletion token appearing relative to all deletion
        tokens.
    """

    # Check to see if the token has been observed in the dataset of token deletions
    if word not in deletion_prob_dict:
        # We want to return a very improbable value
        return 1 / len(word)  # already confirmed word != None

    return deletion_prob_dict[word]


def get_swap_prob(pair: str, swaps_prob_dict: dict) -> float:
    """Takes a token swap pair and the dictionary of all token swap pair probabilities, and returns
    a probability value to be used in the numerator calculation in the compare function.

    Args:
        pair (str): Token swap pair
        swaps_prob_dict (dict): Dictionary containing token swap pairs and swap pair probability as
        the key-value pair.

    Returns:
        float: Probability value of the arg token swap pair appearing relative to all swap token
        pairs.
    """

    # # Check to see if the token has been observed in the dataset of token swap pairs
    if pair not in swaps_prob_dict:
        # If not, we want to return a very improbable value
        pair_split = pair.split(",")
        # use levenshtein_distance to calculate percentage change for "unseen"
        # transforms
        return 1 - Levenshtein.distance(pair_split[0], pair_split[1]) / (len(pair) - 1)

    return swaps_prob_dict[pair]


def get_total_yext_tokens(word: str, yext_prob_dict: dict, yext_count_dict: dict) -> float:
    """Takes an token, a dictionary containing all Yext tokens and their probabilities, and a
    dictionary containing all Yext tokens and the count of occurrences. Checks to see if that token
    has been observed among Yext tokens before. Then calculates the probability of the arg token
    occurring if it exists in the dict, or if unobserved, returns a very small value based on
    1/(sum of all Yext token occurrences).

    Args:
        word (str): A token (could be insertion, deletion, or swap)
        yext_prob_dict (dict): Dictionary containing all Yext tokens and token probability as
        the key-value pair.
        yext_count_dict (dict): Dictionary containing all Yext tokens and token count as
        the key-value pair.

    Returns:
        float: Probability of the arg token occurring relative to all Yext tokens.
    """

    # Check to see if the token has been observed in the dataset of all Yext tokens
    if word not in yext_prob_dict:
        # very improbable
        return 1 / sum(yext_count_dict.values())

    return yext_prob_dict[word]


def get_total_pub_tokens(word: str, pub_prob_dict: dict, pub_count_dict: dict) -> float:
    """Takes an token, a dictionary containing all publisher tokens and their probabilities, and a
    dictionary containing all publisher tokens and the count of occurrences. Checks to see if that
    tokenhas been observed among publisher tokens before. Then calculates the probability of the arg
    token occurring if it exists in the dict, or if unobserved, returns a very small value based on
    1/(sum of all publisher token occurrences).

    Args:
        word (str): A token (could be insertion, deletion, or swap)
        pub_prob_dict (dict): Dictionary containing all publisher tokens and token probability as
        the key-value pair.
        pub_count_dict (dict): Dictionary containing all publisher tokens and token count as
        the key-value pair.

    Returns:
        float: Probability of the arg token occurring relative to all publisher tokens.
    """

    # Check to see if the token has been observed in the dataset of all pub tokens
    if word not in pub_prob_dict:
        # very improbable
        return 1 / sum(pub_count_dict.values())

    return pub_prob_dict[word]


def compare(
    token1: list[str],
    token2: list[str],
    deletion_prob_dict: dict,
    insertion_prob_dict: dict,
    swap_prob_dict: dict,
    yext_token_prob_dict: dict,
    yext_token_count_dict: dict,
    pub_token_prob_dict: dict,
    pub_token_count_dict: dict,
) -> float:
    """For each token pair in the two strings, checks whether a token is a transformation or
    insertion/deletion and pulls the token probability for the given edit type, then calculates the
    overall probability of tokens1 (the Yext token string) being transformed/edited into tokens2
    (the publisher token string).

    Args:
        token1 (list): List of tokens in the Yext token string
        token2 (list): List of tokens in the publisher token string

    Returns:
        float: The probability of the Yext token string (tokens1) being transformed into
        the publisher token string (tokens2), where a value close to 1 indicates
        high probability and a value close to zero indicates an uncommon edit or
        transform
    """
    # Initialize probability, defaults to 1
    prob = 1.0
    # Call the Levenshtein distance function on the two tokens
    edit_list = lev.levenshtein_distance_dp(token1, token2)

    # Iterate through each tuple in the edit_list
    for item in edit_list:
        # Check the first element in the tuple to see what each token pair's edit type is
        if item[0] == "delete":
            # Multiply the probability by the probability of the first item in the second element
            # of the tuple, which is the deleted token
            prob = prob * get_delete_prob(item[1][0], deletion_prob_dict)
        elif item[0] == "insert":
            # Multiply the probability by the probability of the second item in the second element
            # of the tuple, which is the inserted token
            prob = prob * get_insert_prob(item[1][1], insertion_prob_dict)
        elif item[0] == "swap":
            # Create a string from the first and second items of the second element of the tuple
            # to create the token swap pair
            string1 = item[1][0] + ", " + item[1][1]
            prob = prob * get_swap_prob(string1, swap_prob_dict)

    # Checks to see if probability is 1, aka the strings are exactly identical
    if prob == 1:
        return prob
    # If there are differences, perform edit probability calculations
    else:
        # Iterate through each token in the token string to get the sum of probabilities in the
        # numerator
        for i in token1:
            # Add together the different token change probabilities
            prob = prob + get_total_yext_tokens(i, yext_token_prob_dict, yext_token_count_dict)

        # Initialize denominator variable, default equal to 1
        denom = 1.0
        for i in token2:
            # Add together the different token change probabilities in the denom as well
            denom = denom + get_total_pub_tokens(i, pub_token_prob_dict, pub_token_count_dict)

        # Bayesian probability value of a token string transformation
        return prob / denom
