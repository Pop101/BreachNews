import re
import os
import time
import itertools
from glob import glob

import numpy as np
import pandas as pd

SUBS_TO_EXCLUDE = [
    "redditrequest",
    "modhelp",
    "needamod",
    "modclub",
    "modeveryone",
]


def get_tenures():
    tenures = pd.read_csv("../data/mod_tenures.csv")
    tenures = tenures.set_index(["mod", "subreddit"])
    tenures.started = pd.to_datetime(tenures.started, unit="s", utc=True)
    tenures.ended = pd.to_datetime(tenures.ended, unit="s", utc=True)

    print(f"Loaded {len(tenures):,d} non-bot mod tenures.")

    tenures = tenures.loc[
        (tenures.started >= pd.Timestamp("1/1/2011", tz="utc"))
        & (tenures.started <= pd.Timestamp("30/6/2021", tz="utc")),
        :,
    ]
    print(f"Filtered down to {len(tenures):,d} tenures in our time-range.")

    tenures = tenures.reset_index().set_index(["mod", "subreddit", "started"])
    return tenures


def get_activity():
    activity = pd.read_csv("../data/mod_activity_before_joining.csv")
    activity.started = pd.to_datetime(activity.started, unit="s", utc=True)
    activity = activity.set_index(["mod", "subreddit", "started"])
    activity = activity.drop("Unnamed: 0", axis="columns")

    print(f"Loaded {len(activity):,d} mod activity features.")
    return activity


def get_activity_during_tenure():
    activity = pd.read_csv("../data/mod_activity_during_tenure.csv")
    activity.started = pd.to_datetime(activity.started, unit="s", utc=True)
    activity = activity.set_index(["mod", "subreddit", "started"])
    activity = activity.drop("Unnamed: 0", axis="columns")  # drop the index column

    print(f"Loaded {len(activity):,d} mod activity-by-tenure features.")
    return activity


def get_account_lifetimes_by_sub():
    print("Loading account lifetimes by sub...")
    lifespans_by_sub = pd.read_csv("../data/mod_account_lifespans.csv")

    lifespans_by_sub = lifespans_by_sub.rename(columns={"author": "mod"})
    lifespans_by_sub.first_activity = pd.to_datetime(
        lifespans_by_sub.first_activity, unit="s", utc=True
    )
    lifespans_by_sub.last_activity = pd.to_datetime(
        lifespans_by_sub.last_activity, unit="s", utc=True
    )

    lifespans_by_sub = lifespans_by_sub.set_index(["mod", "subreddit"])
    lifespans_by_sub = lifespans_by_sub.drop("Unnamed: 0", axis="columns")

    return lifespans_by_sub


def get_account_lifetimes():
    df = get_account_lifetimes_by_sub()

    # we want column first_activity to be min and last_activity to be max
    return df.groupby("mod").agg({"first_activity": "min", "last_activity": "max"})


def get_top10k_subs():
    print("Loading top 10k subs...")
    top10k_subs = pd.read_csv("../data/embedding-metadata.tsv", sep="\t")
    top10k_subs = top10k_subs.rename(columns={"community": "subreddit"})
    top10k_subs = top10k_subs.set_index("subreddit")
    return top10k_subs


def get_df():
    print("Loading data...")
    tenures = get_tenures()
    activity = get_activity()
    activity_during_tenure = get_activity_during_tenure()
    df = tenures.join(activity, how="left").fillna(0)
    df = df.join(activity_during_tenure.drop("ended", axis=1), how="left").fillna(0)

    for posts_column in df.columns:
        if "posts" not in posts_column:
            continue

        comments_column = re.sub(r"posts", "comments", posts_column)
        total_column = re.sub(r"posts", "items", posts_column)

        if (posts_column in df.columns) and (comments_column in df.columns):
            df[total_column] = df.loc[:, posts_column] + df.loc[:, comments_column]
        else:
            print("Missing a column, skipping!")

    for in_sub_column in df.columns:
        if "_in_sub" not in in_sub_column:
            continue

        total_column = re.sub(r"_in_sub", "", in_sub_column)
        excl_sub_column = re.sub(r"_in_sub", "_excl_sub", in_sub_column)

        if (total_column in df.columns) and (in_sub_column in df.columns):
            df[excl_sub_column] = df.loc[:, total_column] - df.loc[:, in_sub_column]
        else:
            print("Missing a column, skipping!")
    return df


def get_outcomes(pat="all"):
    start_time = time.monotonic()

    if pat == "all":
        pat = "R[SC]_*.jsonl"
    elif pat == "posts":
        pat = "RS_*.jsonl"
    elif pat == "comments":
        pat = "RC_*.jsonl"
    else:
        raise NotImplementedError(f"Invalid pattern: {pat}")

    print(f"Loading moderator sentiment items with pattern {pat}...")
    mod_sentiment_threads = []
    for p in glob(
        os.path.join(
            "/projects/bdata/reddit_moderator_recruiting/final_pass_results", pat
        )
    ):
        mod_sentiment_threads.append(pd.read_json(p, lines=True))

    print(f"Loaded from {len(mod_sentiment_threads)} files.")

    mod_sentiment_threads = pd.concat(mod_sentiment_threads)
    # remove duplicates
    mod_sentiment_threads = mod_sentiment_threads.drop_duplicates(
        subset="id"
    ).set_index("id")

    # remove subreddits to exclude
    mod_sentiment_threads = mod_sentiment_threads.loc[
        ~mod_sentiment_threads.subreddit.isin(SUBS_TO_EXCLUDE), :
    ]

    print(
        f"Finished loading {len(mod_sentiment_threads):,d} classified moderator sentiment items in {time.monotonic()-start_time:.1f} seconds."
    )

    return mod_sentiment_threads


def get_removed_content_counts():
    start_time = time.monotonic()

    counts = pd.read_csv(
        "/projects/bdata/reddit_derivative_data/removed_item_counts.csv"
    )
    counts[["year", "month"]] = counts.month.str.split(pat="-", n=2, expand=True)
    # remove subreddits to exclude
    counts = counts.loc[~counts.subreddit.isin(SUBS_TO_EXCLUDE), :]
    counts.year = counts.year.astype("int")
    counts.month = counts.month.astype("int")
    counts = counts.set_index(["subreddit", "year", "month"])

    print(
        f"Finished loading {len(counts):,d} removed/deleted/all item counts in {time.monotonic()-start_time:.1f} seconds."
    )
    return counts


def get_counts():
    start_time = time.monotonic()

    print("Loading data and computing indices...")
    outcomes = get_outcomes()
    counts = get_removed_content_counts()

    outcomes.created_utc = pd.to_datetime(outcomes.created_utc, unit="s")
    outcomes["year"] = outcomes.created_utc.dt.year
    outcomes["month"] = outcomes.created_utc.dt.month

    # get boolean cols
    for sentiment in outcomes.sentiment_predicted.unique():
        outcomes[f"sentiment_{sentiment}"] = outcomes.sentiment_predicted == sentiment

    # count
    outcomes = outcomes.groupby(["subreddit", "year", "month"]).agg(
        {
            "sentiment_predicted": "count",
            "sentiment_positive": "sum",
            "sentiment_neutral": "sum",
            "sentiment_negative": "sum",
        }
    )

    outcomes = outcomes.rename(
        {
            "sentiment_predicted": "all_mod_items",
            "sentiment_positive": "positive_mod_items",
            "sentiment_neutral": "neutral_mod_items",
            "sentiment_negative": "negative_mod_items",
        },
        axis="columns",
    )

    print("Joining... might take a little bit.")
    counts = counts.join(outcomes, how="outer")

    print(
        f"Finished loading {len(counts):,d} counts in {(time.monotonic()-start_time)/60:.2f} minutes."
    )
    return counts