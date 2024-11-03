import os
import pandas as pd
import seaborn as sns
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt

# Load CSV file
data_csv = pd.read_csv("./joined_articles_companies.csv", encoding="ISO-8859-1")

# Create a figures/ folder if it doesn't exist
if not os.path.exists("figures"):
    os.makedirs("figures")

data_csv = data_csv.reindex(sorted(data_csv.columns), axis=1)
pd.set_option('display.max_columns', None)

# print('Found', len(data_csv.columns), 'columns')
# for c in data_csv.columns:
#   print (c)

matrix = msno.matrix(data_csv, labels = True)
matrix_copy = matrix.get_figure()
matrix_copy.savefig('./figures/matrix.png', bbox_inches = 'tight')

heatmap = msno.heatmap(data_csv, labels = True, fontsize=9, figsize=(14, 14), n=len(data_csv.columns))
heatmap_copy = heatmap.get_figure()
heatmap_copy.savefig('./figures/heatmap.png', bbox_inches = 'tight')