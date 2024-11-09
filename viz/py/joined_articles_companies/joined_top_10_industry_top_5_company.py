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
data_csv = pd.read_csv(
    "../../data/article_data/joined_articles_companies_no_duplicates.csv",
    encoding="ISO-8859-1",
    low_memory=False,
)
print(
    f"\nFinished loading {len(data_csv):,d} rows in {time.monotonic()-start_time:.1f} seconds.\n"
)

if not os.path.exists("figures"):
    os.makedirs("figures")

data_csv = data_csv.reindex(sorted(data_csv.columns), axis=1)
pd.set_option("display.max_columns", None)

# Filter out the company name "True" prior to creating the plots\
# TRUE == NO COMPANY MENTIONED, BUT STILL ABOUT BREACH??
# FIXME: SHOULD JUST BE FALSE
data_csv = data_csv[data_csv["name"] != "true"]

# Group by industry and company, and count the number of rows for each combination
company_counts = data_csv.groupby(["industry", "name"]).size().unstack(fill_value=0)

# Sort the industries by total count in descending order
industry_counts = company_counts.sum(axis=1).sort_values(ascending=False)

# Plot the top 5 companies in each of the top 10 industries
for industry in industry_counts.nlargest(10).index:
    top_5_companies = company_counts.loc[industry].nlargest(5)
    plt.bar(top_5_companies.index, top_5_companies.values)
    plt.title(
      "\n".join(
        [f"Top 5 Companies in", f"\"{industry}\""]
      ),
      fontsize=12,
      fontweight="bold"
    )
    plt.xlabel("Company", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Articles", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
      f"figures/joined/top_5_{industry.replace('/', '_').replace(' ', '_').replace('&', 'and')}.png"
    )
    plt.clf()

# Combine all plots into a single image
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(4 * 5, 6 * 2))
fig.suptitle(
  "Top 5 Companies in Each of the Top 10 Industries",
  fontsize=24,
  fontweight="bold"
)
for i, industry in enumerate(industry_counts.nlargest(10).index):
    top_5_companies = company_counts.loc[industry].nlargest(5)
    axs[i // 5, i % 5].bar(top_5_companies.index, top_5_companies.values)
    axs[i // 5, i % 5].set_title(
      "\n".join(
        [f"Top 5 Companies in", f"\"{industry}\""]
      ),
      fontsize=12,
      fontweight="bold"
    )
    # axs[i // 5, i % 5].set_xlabel("Company", fontsize=12, fontweight="bold")
    # axs[i // 5, i % 5].set_ylabel("Number of Articles", fontsize=12, fontweight="bold")
    axs[i // 5, i % 5].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("figures/top_5_companies_all.png")
plt.clf()
