import os
import time
import pandas as pd
import seaborn as sns
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt

start_time = time.monotonic()
print("Reading input file...")
# FIXME: low_memory=False seems to be deprecated option; figure out dtypes
data_csv = pd.read_csv("./run1.csv", encoding="ISO-8859-1", low_memory=False)
print(
    f"\nFinished loading {len(data_csv):,d} rows in {time.monotonic()-start_time:.1f} seconds.\n"
)

if not os.path.exists("figures"):
    os.makedirs("figures")

data_csv = data_csv.reindex(sorted(data_csv.columns), axis=1)
pd.set_option("display.max_columns", None)

# unique articles by URL
unique_articles = data_csv.drop_duplicates(subset="URL")

publication_counts = unique_articles["Publication"].value_counts()

# FIXME: column name "Publication" keeps getting appearing as separate row (its in header...)
publication_counts = publication_counts[publication_counts.index != "Publication"]

print(publication_counts)
total_articles = publication_counts.sum()
print(f"\nTotal number of unique articles: {total_articles:,d}\n")

# sort publications by count
publication_counts_sorted = publication_counts.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=publication_counts_sorted.index,
    y=publication_counts_sorted.values,
    palette="viridis",
)
plt.title("Number of Unique Articles per Publication")
plt.xlabel("Publication")
plt.ylabel("Number of Unique Articles")
plt.xticks(rotation=45, ha="right")

# count labels
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.tight_layout()

# save viz as png
plt.savefig("figures/unique_articles_per_publication_all.png", dpi=300)
# plt.show()
