import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
data_csv = data_csv.drop_duplicates(subset="URL")

# Fill NaN values in 'CompanyMentioned' with "False"
data_csv["CompanyMentioned"] = data_csv["CompanyMentioned"].fillna("False")

# Ensure 'CompanyMentioned' is of type string
data_csv["CompanyMentioned"] = data_csv["CompanyMentioned"].astype(str)

# Group by 'Publication' and 'CompanyMentioned' to get counts for each combination
article_counts = (
    data_csv.groupby(["Publication", "CompanyMentioned"]).size().unstack(fill_value=0)
)

# Remove the row where 'Publication' is "Publication" (seems to be an error in the data)
article_counts = article_counts[article_counts.index != "Publication"]

# Add a 'Total' column for the overall unique article count per publication
article_counts["Total"] = article_counts.sum(axis=1)

article_counts = article_counts.sort_values(by="Total", ascending=False)

# # Print out the counts for each publication
# print("Publication-wise unique article counts:")
# for publication, row in article_counts.iterrows():
#     print(f"Publication: {publication}")
#     print(f"  Total articles: {row['Total']}")
#     print(f"  CompanyMentioned=False: {row.get('False', 0)}")
#     print(f"  CompanyMentioned=True: {row.get('True', 0)}")
#     print()

# Plotting the stacked bar chart
plt.figure(figsize=(12, 8))
article_counts[["False", "True"]].plot(
    kind="bar",
    stacked=True,
    color=["lightcoral", "lightgreen"],
    ax=plt.gca(),
    # legend=False,
)
plt.title("Unique Article Counts by Publication with Company Mentioned")
plt.xlabel("Publication")
plt.ylabel("Number of Unique Articles")
plt.xticks(rotation=45)

# Add a label with counts for True and False
for i, (publication, row) in enumerate(article_counts.iterrows()):
    plt.text(
        i,
        row["True"] + row["False"] / 2,
        f"False: {int(row.get('False', 0)):,}\nTrue: {int(row.get('True', 0)):,}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )

plt.tight_layout()

# Save and show the plot
plt.savefig("figures/stacked_company_mentions_by_publication.png")
# plt.show()
