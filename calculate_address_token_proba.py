"""
Module to tokenize Yext and publisher Address field data, extract token-level probabilities and calculate the transition probabilities of Yext Address data to publisher Address data. Returns a TSV file containing all raw Yext and publisher strings, tokenized strings, edits, and edit probability.

"""

import logging
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import pandas as pd
import html
import typing

from snowflake import connector
from snowflake.connector.errors import DatabaseError
from snowflake.connector.pandas_tools import write_pandas
from wordfreq import tokenize
from datetime import date

import os
import sys

if "DS_MONOREPO" not in os.environ:
    LOGGER.error(
        "Ensure DS_MONOREPO environment variable is set to the root of your datascience monorepo!"
    )
    raise ImportError

DS_MONOREPO = os.getenv("DS_MONOREPO")
sys.path.append(DS_MONOREPO)

import listings_verifier.implementation_scripts.compare_tokens as ct
import listings_verifier.implementation_scripts.group_edits as ge

from common.utils.DownloadUtils import get_data_from_snowflake
from common.utils.UploadUtils import upload_to_snowflake

SNOWFLAKE_ACCT = "tw61901.us-east-1"
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
WAREHOUSE = "HUMAN_WH"
SNOWFLAKE_PASS = os.getenv("SNOWFLAKE_PASS")
ROLE = "DATA_SCIENCE"


# Snowflake table location and warehouse, for uploading the DataFrame
WAREHOUSE = "HUMAN_WH"
DATABASE = "PROD_DATA_SCIENCE"
SCHEMA = "PUBLIC"
TABLE = "HISTORICAL_ADDRESS_TRANSITION_PROBA"


def query_base_information() -> pd.DataFrame:
    """Queries Snowflake for basic information: publisher ID, field type, Yext and publisher field
    data (both raw and tokenized), and verifier prediction, where both Yext and publisher field
    values are not empty.

    Returns:
        pd.DataFrame: DataFrame of extract from Snowflake.
    """

    query = """
            SELECT 
                publisher_id, 
                entity_id, 
                yext_listing_id, 
                field_name, 
                array_to_string(entity_value, ';') AS entity_string, 
                array_to_string(publisher_value, ';') AS publisher_string, 
                verifier_predicted_match 
            FROM  
                PROD_LISTINGS_LOCAL.PUBLIC.LISTING_VERIFICATIONS_AS_OF_MAY_2022 
            WHERE 
                entity_value is NOT NULL
                AND publisher_value is NOT NULL
                AND field_name = 'Address'
            LIMIT 100

    """

    LOGGER.info("Connecting to Snowflake...")
    LOGGER.info("Pulling the base DataFrame...")

    df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    df.columns = [
        "PUBLISHER_ID",
        "ENTITY_ID",
        "YEXT_LISTING_ID",
        "FIELD_NAME",
        "ENTITY_STRING",
        "PUBLISHER_STRING",
        "VERIFIER_PREDICTED_MATCH",
    ]

    LOGGER.info("Base DataFrame pulled!")

    return df


def clean_and_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """Takes in the base DataFrame and for all Yext and publisher field strings, applies HTML
    decoding to allow for better tokenization. Then it tokenizes the HTML-decoded Yext and publisher
    field using the UAX-29 tokenizer, creating two new columns: entity_string_tokenized and
    publisher_string_tokenized.

    Args:
        df (pd.DataFrame): DataFrame containing basic Yext and publisher field data

    Returns:
        pd.DataFrame: Returns the same DataFrame, but with HTML decoding on the original field value strings, and with two new columns containing tokenized Yext and publisher field data
    """

    LOGGER.info("Cleaning the base DataFrame...")

    # apply the HTML decoder to Yext and publisher field value columns
    df["ENTITY_STRING"] = df["ENTITY_STRING"].apply(html.unescape)
    df["PUBLISHER_STRING"] = df["PUBLISHER_STRING"].apply(html.unescape)

    LOGGER.info("Tokenizing the base DataFrame...")
    # use the UAX29 tokenizer on the cleaned field value columns and create new tokenized string
    # cols
    df["ENTITY_STRING_TOKENIZED"] = df["ENTITY_STRING"].apply(tokenize, args=("en",))
    df["PUBLISHER_STRING_TOKENIZED"] = df["PUBLISHER_STRING"].apply(tokenize, args=("en",))

    LOGGER.info("Removing blank rows from the base DataFrame...")
    # remove rows with blank tokenized string values
    df = df[df["ENTITY_STRING_TOKENIZED"].str.len() != 0]
    df = df[df["PUBLISHER_STRING_TOKENIZED"].str.len() != 0]

    LOGGER.info("Base DataFrame cleaned and tokenized!")

    return df


def parse_deletions(list: list[tuple]) -> list[str]:
    """Takes a list of tuples, pulls the deletion token out of the tuple, and returns a master list of strings containing all the relevant deletion tokens

    Args:
        list (list[tuple]): List of tuples containing the deletion tokens and their backpointers. The backpointer token will be removed from each tuple

    Returns:
        list[str]: List of deletion tokens, to be returned in the column
    """
    # Initialize empty list to be returned
    return_list = []
    # Iterate through each tuple in the deletion list
    for item in list:
        # Append the first token of the tuple, which is the deleted token, to the master list
        return_list.append(item[0])
    return return_list


def parse_insertions(list: list[tuple]) -> list[str]:
    """Takes a list of tuples, pulls the insertion token out of the tuple, and returns a master list of strings containing all the relevant insertion tokens

    Args:
        list (list[tuple]): List of tuples containing the insertion tokens and their backpointers. The backpointer token will be removed from each tuple

    Returns:
        list[str]: List of insertion tokens, to be returned in the column
    """
    # Initialize empty list to be returned
    return_list = []
    # Iterate through each tuple in the insertion list
    for item in list:
        # Append the second token of the tuple, which is the deleted token, to the master list
        return_list.append(item[1])
    return return_list


def get_deletions_dict() -> dict:
    """Queries Snowflake and creates a DataFrame containing all token deletions and token
    probability. Then converts the DataFrame into a dictionary containing tokens and token
    probability. This is accessed in the numerator calculation in the comparison function.

    Returns:
        dict: Dictionary containing token deletions and token probability as the key-value pair.
    """

    query = """
                SELECT 
                    TOKEN, 
                    TOKEN_PROBABILITY
                FROM 
                    PROD_DATA_SCIENCE.PUBLIC.ADDRESS_TOKEN_DELETE_PROBA
                ORDER BY 
                    2 DESC
    """

    # Create the DataFrame containing deletion tokens and their probabilities
    deletion_prob_df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    deletion_prob_df.columns = ["token", "token_probability"]

    # Convert the 'token' and 'token_probability' columns into a dictionary
    deletion_prob_dict = deletion_prob_df.set_index("token").to_dict()["token_probability"]

    return deletion_prob_dict


def get_insertions_dict() -> dict:
    """Queries Snowflake and returns a DataFrame containing all token insertions and token
    probability. Then converts the DataFrame into a dictionary containing tokens and token
    probability. This is accessed in the numerator calculation in the comparison function.

    Returns:
        dict: Dictionary containing token insertions and token probability as the key-value pair.
    """

    query = """
                SELECT 
                    TOKEN, 
                    TOKEN_PROBABILITY 
                FROM 
                    PROD_DATA_SCIENCE.PUBLIC.ADDRESS_TOKEN_INSERT_PROBA         
                ORDER BY 
                    2 DESC
    """

    # Create the DataFrame containing insertion tokens and their probabilities
    insertion_prob_df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    insertion_prob_df.columns = ["token", "token_probability"]

    # Convert the 'token' and 'token_probability' columns into a dictionary
    insertion_prob_dict = insertion_prob_df.set_index("token").to_dict()["token_probability"]

    return insertion_prob_dict


def get_swaps_dict() -> dict:
    """Queries Snowflake and returns a DataFrame containing all token swap pairs and token pair
    probability. Then converts the DataFrame into a dictionary containing token swap pairs and
    token pair probability. This is accessed in the numerator calculation in the comparison
    function.

    Returns:
        dict: Dictionary containing token swap pairs and token pair probability as the key-value
              pair.
    """

    query = """
                SELECT
                    TOKEN, 
                    TOKEN_PROBABILITY
                FROM
                    PROD_DATA_SCIENCE.PUBLIC.ADDRESS_TOKEN_TRANSFORM_PROBA
                ORDER BY 
                    2 DESC
    """

    # Create the DataFrame containing swap token pairs and their probabilities
    swaps_prob_df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    swaps_prob_df.columns = ["token", "token_probability"]

    # Convert the 'token' and 'token_probability' columns into a dictionary
    swaps_prob_dict = swaps_prob_df.set_index("token").to_dict()["token_probability"]

    return swaps_prob_dict


def get_all_yext_prob_df() -> pd.DataFrame:
    """Queries Snowflake and returns a DataFrame containing all Yext tokens and token probability.

    Returns:
        pd.DataFrame: DataFrame containing all Yext tokens, token count, and token probability.
    """

    query = """
                SELECT 
                    TOKEN, 
                    COUNT, 
                    TOKEN_PROBABILITY 
                FROM 
                    PROD_DATA_SCIENCE.PUBLIC.ALL_YEXT_ADDRESS_TOKEN_PROBA 
                ORDER BY 
                    3 DESC
    """

    # Create the DataFrame containing Yext tokens and their probabilities
    yext_prob_df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    yext_prob_df.columns = ["token", "count", "token_probability"]

    return yext_prob_df


def get_all_pub_prob_df() -> pd.DataFrame:
    """Queries Snowflake and returns a DataFrame containing all Yext tokens and token probability.

    Returns:
        pd.DataFrame: DataFrame containing all publisher tokens, token count, and token probability.
    """

    query = """
                SELECT
                    TOKEN, COUNT, TOKEN_PROBABILITY
                FROM
                    PROD_DATA_SCIENCE.PUBLIC.ALL_PUB_ADDRESS_TOKEN_PROBA
                ORDER BY 
                    3 DESC
    """

    # Create the DataFrame containing publisher tokens and their probabilities
    pub_prob_df = get_data_from_snowflake(
        query, SNOWFLAKE_ACCT, SNOWFLAKE_USER, WAREHOUSE, SNOWFLAKE_PASS, ROLE
    )
    pub_prob_df.columns = ["token", "count", "token_probability"]

    return pub_prob_df


def get_all_yext_prob_dict(yext_prob_df: pd.DataFrame) -> dict:
    """Takes the DataFrame containing all Yext token count and probability data, then converts the
    DataFrame into a dictionary containing tokens and token probability. This is accessed in the
    numerator calculation in the comparison function.

    Args:
        yext_prob_df (pd.DataFrame): DataFrame containing all Yext token count and probability data.

    Returns:
        dict: Dictionary containing all Yext tokens and token probability as the key-value pair.
    """

    # Create the dictionary containing Yext tokens and Yext token probability
    yext_prob_dict = yext_prob_df.set_index("token").to_dict()["token_probability"]

    return yext_prob_dict


def get_all_pub_prob_dict(pub_prob_df: pd.DataFrame) -> dict:
    """Takes the DataFrame containing all publisher token count and probability data, then converts
    the DataFrame into a dictionary containing tokens and token probability. This is accessed in the
     denominator calculation in the comparison function.

    Args:
        pub_prob_df (pd.DataFrame): DataFrame containing all publisher token count and probability
        data.

    Returns:
        dict: Dictionary containing all publisher tokens and token probability as the key-value
        pair.
    """

    # Create the dictionary containing pub tokens and pub token probability
    pub_prob_dict = pub_prob_df.set_index("token").to_dict()["token_probability"]

    return pub_prob_dict


def get_all_yext_count_dict(yext_prob_df: pd.DataFrame) -> dict:
    """Takes the DataFrame containing all Yext token count and probability data, then converts the
    DataFrame into a dictionary containing tokens and token count. This is accessed in the numerator
    calculation in the comparison function.

    Args:
        yext_prob_df (pd.DataFrame): DataFrame containing all Yext token count and probability data.

    Returns:
        dict: Dictionary containing all Yext tokens and token count as the key-value pair.
    """

    # Create the dictionary containing Yext tokens and Yext token counts
    yext_count_dict = yext_prob_df.set_index("token").to_dict()["count"]

    return yext_count_dict


def get_all_pub_count_dict(pub_prob_df: pd.DataFrame) -> dict:
    """Takes the DataFrame containing all publisher token count and probability data, then converts
    the DataFrame into a dictionary containing tokens and token count. This is accessed in the
    numerator calculation in the comparison function.

    Args:
        pub_prob_df (pd.DataFrame): DataFrame containing all publisher token count and probability
        data.

    Returns:
        dict: Dictionary containing all publisher tokens and token count as the key-value pair.
    """
    # Create the dictionary containing pub tokens and pub token counts
    pub_count_dict = pub_prob_df.set_index("token").to_dict()["count"]

    return pub_count_dict


def process_final_df(
    df: pd.DataFrame,
    deletion_prob_dict: dict,
    insertion_prob_dict: dict,
    swap_prob_dict: dict,
    yext_token_prob_dict: dict,
    yext_token_prob_count: dict,
    pub_token_prob_dict: dict,
    pub_token_count_dict: dict,
    snowflake: bool,
) -> typing.TextIO:
    """Takes the cleaned and tokenized DataFrame, processes it for edit types, calculates token
    string probabilties and returns a final DataFrame as a .tsv file containing the original
    DataFrame, plus additional columns that track
    - all edits (deletions, insertions, swaps)
    - probability of token string transformation

    Optionally, users can specify whether to write the resulting DataFrame to Snowflake instead of returning the DataFrame in a .tsv file

    Args:
        df (pd.DataFrame): Cleaned, tokenized DataFrame containing all Yext and publisher token
        string data
        snowflake (bool): Determines whether to upload to Snowflake. Defaults to False
        deletion_prob_dict (dict): Dictionary containing all deletion tokens and their probabilities
        insertion_prob_dict (dict): Dictionary containing all insertion tokens and their
        probabilities
        swap_prob_dict (dict): Dictionary containing all swap token pairs and their probabilities
        yext_token_prob_dict (dict): Dictionary containing all Yext tokens and their probabilities
        yext_token_prob_count (dict): Dictionary containing all publisher tokens and their
        probabilities
        pub_token_prob_dict (dict): Dictionary containing all Yext tokens and the count of
        occurrences
        pub_token_count_dict (dict): Dictionary containing all publisher tokens and the count of
        occurrences

    Returns:
        typing.TextIO: .tsv file containing the DataFrame output, with Yext and publisher strings,
        tokenized strings, all edits grouped by edit type, and transition probability columns
    """

    LOGGER.info("Processing the final DataFrame to be returned...")

    LOGGER.info("Filtering out the unecessary columns...")
    # filter out unnecessary columns: we only want publisher ID and the tokenized strings
    filtered_df = df[["PUBLISHER_ID", "ENTITY_STRING_TOKENIZED", "PUBLISHER_STRING_TOKENIZED"]]
    LOGGER.info("Columns filtered out!")

    LOGGER.info("Calculating all edit types...")
    # for each row, create a column that includes all token-level edits, regardless of type
    filtered_df["ALL_EDITS"] = filtered_df.apply(
        lambda x: ge.groupby_edit_type(
            x["ENTITY_STRING_TOKENIZED"], x["PUBLISHER_STRING_TOKENIZED"]
        ),
        axis=1,
    )
    LOGGER.info("All edit types calculated!")

    LOGGER.info("Filtering deletions, insertions, and swaps...")
    # from the all_edits column, create additional deletion, insertion, and swap column
    filtered_df.loc[:, "DELETIONS"] = filtered_df.ALL_EDITS.map(lambda x: x[0])
    filtered_df.loc[:, "INSERTIONS"] = filtered_df.ALL_EDITS.map(lambda x: x[1])
    filtered_df.loc[:, "SWAPS"] = filtered_df.ALL_EDITS.map(lambda x: x[2])
    # pull only the relevant edit token (insertion or deletion, depending on edit type) from the deletion and insertion token tuples
    filtered_df["DELETIONS"] = filtered_df.apply(lambda x: parse_deletions(x["DELETIONS"]), axis=1)
    filtered_df["INSERTIONS"] = filtered_df.apply(
        lambda x: parse_insertions(x["INSERTIONS"]), axis=1
    )
    LOGGER.info("All edit types filtered!")

    LOGGER.info("Calculating token transformation probabilities...")
    # create a new column containing the token string edit probability and call the comparison
    # function
    filtered_df["EDIT_PROBABILITY"] = filtered_df.apply(
        lambda x: ct.compare(
            x["ENTITY_STRING_TOKENIZED"],
            x["PUBLISHER_STRING_TOKENIZED"],
            deletion_prob_dict,
            insertion_prob_dict,
            swap_prob_dict,
            yext_token_prob_dict,
            yext_token_prob_count,
            pub_token_prob_dict,
            pub_token_count_dict,
        ),
        axis=1,
    )

    # Add a run date column
    filtered_df["RUN_DATE"] = date.today()

    LOGGER.info("Done!")

    # Upload to Snowflake if specified as an arg
    if snowflake:
        LOGGER.info("Loading DataFrame into a Snowflake table...")
        upload_to_snowflake(
            filtered_df,
            SNOWFLAKE_ACCT,
            SNOWFLAKE_USER,
            SNOWFLAKE_PASS,
            WAREHOUSE,
            DATABASE,
            SCHEMA,
            TABLE,
            ROLE,
        )
        LOGGER.info("Data successfully uploaded!")
    else:
        LOGGER.info("Saving final DataFrame to a .tsv file...")
        filtered_df_csv = filtered_df.to_csv(
            r"/Users/cxi/datascience/listings_verifier/dataframe_outputs/address_token_transform_proba.tsv",
            sep="\t",
            index=False,
            header=False,
        )
        return filtered_df_csv


def main(args):
    # Query Snowflake table for full base dataset
    base_df = query_base_information()
    # Apply cleaning and tokenization to the Yext and publisher values in the base df
    cleaned_tokenized = clean_and_tokenize(base_df)

    # Initialize dicts that contain token edit probabilities
    # to be used in compare function
    deletion_prob_dict = get_deletions_dict()
    insertion_prob_dict = get_insertions_dict()
    swap_prob_dict = get_swaps_dict()

    # Initialize df that contain Yext token probabilities and counts
    yext_token_df = get_all_yext_prob_df()

    # Create separate dictionaries for Yext token probabilities and counts
    # to be used in compare function
    yext_token_prob_dict = get_all_yext_prob_dict(yext_token_df)
    yext_token_prob_count = get_all_yext_count_dict(yext_token_df)

    # Initialize df that contains publisher token probabilities and counts
    pub_token_df = get_all_pub_prob_df()

    # Create separate dictionaries for Yext token probabilities and counts
    # to be used in compare function
    pub_token_prob_dict = get_all_pub_prob_dict(pub_token_df)
    pub_token_count_dict = get_all_pub_count_dict(pub_token_df)

    # Process the final dataframe and do all probability calculations for each token in each row, creating a new column.
    final_df = process_final_df(
        cleaned_tokenized,
        deletion_prob_dict,
        insertion_prob_dict,
        swap_prob_dict,
        yext_token_prob_dict,
        yext_token_prob_count,
        pub_token_prob_dict,
        pub_token_count_dict,
        args.snowflake,
    )

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate probabilities for token string transformations."
    )
    parser.add_argument(
        "-s",
        "--snowflake",
        action="store_true",
        help="Whether to write data to Snowflake table. Defaults to False.",
        default=False,
    )
    args = parser.parse_args()

    LOGGER.info(args)
    main(args)
