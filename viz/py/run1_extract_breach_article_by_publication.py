import os
import time
import pandas as pd

# Path to the original CSV file
input_file_path = "./run1.csv"

# Output directory for breach-related articles
output_dir = "breach_articles_by_publication"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
data = pd.read_csv(input_file_path, encoding="ISO-8859-1")

start_time = time.monotonic()
print("Reading input file...")
data_csv = pd.read_csv(input_file_path, encoding="ISO-8859-1", low_memory=False)
print(
    f"\nFinished loading {len(data_csv):,d} rows in {time.monotonic()-start_time:.1f} seconds.\n"
)

# Drop duplicates based on 'URL' column
data = data.drop_duplicates(subset="URL")

# Fill NaN values in 'BreachMentioned' column and filter for rows where BreachMentioned is "True"
data["BreachMentioned"] = data["BreachMentioned"].fillna("False").astype(str)
breach_data = data[data["BreachMentioned"] != "False"]

# Group by publication and save each group to a separate file
for publication, group in breach_data.groupby("Publication"):
    # Replace spaces or special characters in publication names for file names
    safe_publication_name = "".join(c if c.isalnum() else "_" for c in publication)
    output_file_path = os.path.join(output_dir, f"{safe_publication_name}_breaches.csv")
    
    # Export each group to a separate CSV file
    group.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Saved {len(group)} rows for '{publication}' to {output_file_path}")
