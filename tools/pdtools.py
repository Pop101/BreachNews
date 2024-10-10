

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import pandas as pd
from pandas import DataFrame
from typing import Callable
import numpy as np
import inspect

from collections.abc import Iterable
from typing import Any

import warnings

tqdm.pandas()

def parallel_applymap(df, func, worker_count):
    # From https://stackoverflow.com/questions/58270225/pandas-dataframe-applymap-parallel-execution
    def _apply(shard):
        return shard.progress_applymap(func)

    shards = np.array_split(df, worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as e:
        futures = e.map(_apply, shards)
    return pd.concat(list(futures))

def sort_dataframe_by_key(dataframe: DataFrame, column: str, key: Callable) -> DataFrame:
    # https://stackoverflow.com/questions/52475458/how-to-sort-pandas-dataframe-with-a-key
    """ Sort a dataframe from a column using the key """
    sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
    return DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)

def fleiss_pivot(dfs, column) -> DataFrame:
    """
    Transforms a list of dataframes into a single dataframe with the following properties:
    - Identical indexing to input dataframes
    - One column for every unique value in `column`
    - One row for each row in the combined input dataframes
    - Each cell is the count of the value in the corresponding row of the input dataframes
    """
    
    keys = set()
    unique_labels = set()
    dfs_lst = []
    for df in dfs:
        unique_labels.update(df[column].unique())
        
        if len(keys) == 0:
            keys = set(df.index.names)
        elif keys != set(df.index.names):
            raise ValueError('All dataframes must have the same index')
        
        dfs_lst.append(df)
    keys = list(keys)
    
    # Create a combined DF with key=keys, columns=unique_labels
    output_df = pd.DataFrame(columns=keys + list(unique_labels))
    output_df.set_index(keys, inplace=True)
    
    for df in dfs_lst:
        for index, row in df.iterrows():
            if not index in output_df.index:
                output_df.loc[index] = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output_df.at[index, row[column]] += 1
    
    output_df.fillna(0, inplace=True)
    
    # Ensure that each row sums to the same value
    # Correcting errors in a way that worsens IRR
    row_sums = output_df.sum(axis=1)
    max_row = row_sums.idxmax()
    if not row_sums.eq(row_sums.iloc[0]).all():
        # Add to all other rows, ensuring equally dividing among that row's columns
        for index, row in output_df.iterrows():
            if index == max_row:
                continue
            
            diff = row_sums[max_row] - row_sums[index]
            for col in output_df.columns:
                if col in row.index:
                    output_df.at[index, col] += diff // len(row)
                    diff -= diff // len(row)
            if diff > 0:
                output_df.at[index, output_df.columns[-1]] += diff
                
    return output_df

def bootstrap_ci(df, f, frac=0.2, n=1000, alpha=0.05) -> Any:
    """
    Compute the confidence interval for a function output on a dataframe
    The function must return a sortable value, or a data structure (list|dict|pd.DataFrame) of sortable values,
    and take in a pd.DataFrame as an argument
    """
    from scipy.stats._common import ConfidenceInterval
    from scipy.stats import scoreatpercentile
    
    def quantile(x, q):
        if len(x) == 0:
            return None
        if isinstance(x[0], ConfidenceInterval):
            return ConfidenceInterval(np.quantile([ci.low for ci in x], q), np.quantile([ci.high for ci in x], q))
        if isinstance(x[0], (pd.Series, pd.DataFrame)):
            return x.quantile(q)
        if isinstance(x[0], (int, float, np.number)):
            return np.quantile(x, q)
        
        return min(x, lambda y: abs(x - scoreatpercentile(y, q)))
    
    outputs = list()
    for _ in range(n):
        sample = df.sample(frac=frac, replace=True)
        outputs.append(f(sample))
    
    # Remove all none values
    outputs = list(filter(lambda x: x is not None, outputs))
    
    if isinstance(outputs[0], (list, set, tuple)):
        # Assert all outputs are the same length
        if not all(len(x) == len(outputs[0]) for x in outputs):
            raise ValueError('All outputs must be the same length')
        
        # Combine outputs
        values = [[x for x in output] for output in outputs]
        return [ ConfidenceInterval(quantile(value, q = alpha/2), quantile(value, q = 1-(alpha/2))) for value in values ]
    
    if isinstance(outputs[0], dict):
        # Assert all outputs have the same keys
        if not all(set(x.keys()) == set(outputs[0].keys()) for x in outputs):
            raise ValueError('All outputs must have the same keys')
        
        # Combine outputs
        values = { key: [x[key] for x in outputs] for key in outputs[0].keys() }
        return { key: ConfidenceInterval(quantile(value, q = alpha/2), quantile(value, q = 1-(alpha/2))) for key, value in values.items() }
    
    if isinstance(outputs[0], (pd.Series, pd.DataFrame)):
        # For a dataframe, we want an cell-wise confidence interval
        values = pd.concat(outputs)
        values = values.groupby(values.index).agg(lambda x: list(v for v in x if pd.notnull(v)))
        return values.map(lambda x: ConfidenceInterval(quantile(x, q = alpha/2), quantile(x, q = 1-(alpha/2))))
    
    if isinstance(outputs[0], (int, float, np.number)):
        return ConfidenceInterval(quantile(outputs, q = alpha/2), quantile(outputs, q = 1-(alpha/2)))
    
    raise ValueError('Unsupported function output type: {}'.format(type(outputs[0])))

def cluster(series:Iterable[Any], filter=lambda x, y: True)->list[list[Any]]:
    """
    Clusters a series of values based on a filter function.
    This function returns true iff {x, y} should be in a cluster.
    Returns a list of connected components.
    """
    
    import networkx as nx
    
    if not callable(filter):
        raise ValueError('Filter must be a callable function')
    if len(inspect.signature(filter).parameters) != 2:
        raise ValueError('Filter must take two arguments')
    
    series = list(series)
    
    # Create a graph of all values in the series
    graph = nx.Graph()
    for x in series:
        graph.add_node(x)
    
    for i, x in enumerate(series):
        for y in series[i+1:]:
            if filter(x, y):
                graph.add_edge(x, y)
    
    # Find all connected components
    return list(nx.connected_components(graph))

def kmeans(X:Iterable[Any], n_clusters:int, max_iter:int=100, tol:float=1e-4, distance_func:Callable=None, vector_func:Callable=None) -> pd.Series:
    """
    K-means clustering algorithm.
    X: The data to cluster
    n_clusters: The number of clusters to find
    max_iter: The maximum number of iterations
    tol: The tolerance for convergence
    distance_func: The distance function to use
    vector_func: The function to convert a data point to a vector
    
    NOTE: instead of distance_func, just use sklearn DBScan
    """
    
    if vector_func is None and distance_func is None and not all(isinstance(x, np.ndarray) for x in X):
        raise ValueError('If no vector_func or distance_func is provided, all data points must be numpy arrays')
    if vector_func is None:
        vector_func = lambda x: x
    if distance_func is None:
        distance_func = lambda x, y: np.linalg.norm(x - y)
    if not callable(distance_func):
        raise ValueError('Distance function must be a callable function')
    if not callable(vector_func):
        raise ValueError('Vector function must be a callable function')
    
    X = pd.DataFrame(((x, vector_func(x)) for x in X), columns=['data', 'vector'])
    
    # Assert identical vector shapes
    if hasattr(X['vector'].iloc[0], 'shape') and not all(X['vector'].map(lambda x: x.shape == X['vector'].iloc[0].shape)):
        raise ValueError('All vectors must have the same shape')
    
    # Initialize the centroids
    if all(isinstance(x, np.ndarray) for x in X['vector']):
        centroids = X.sample(n_clusters)
        centroids = np.vstack(centroids['vector'].values)
    else:
        centroids = np.array([X.sample()['vector'].values for _ in range(n_clusters)])
        centroids = centroids.flatten()

    for _ in range(max_iter):
        # Assign each point to the nearest centroid
        X['cluster'] = X.apply(lambda x: np.argmin([distance_func(x['vector'], centroid) for centroid in centroids]), axis=1)
        
        # Update the centroids
        new_centroids = np.array([X[X['cluster'] == i]['vector'].mean(axis=0) for i in range(n_clusters)])
            
        # Check for convergence
        if (np.linalg.norm(new_centroids - centroids) < tol).all():
            break
        
        centroids = new_centroids
    
    return X.drop(columns='vector').set_index('data')['cluster']

def chunkify(df: pd.DataFrame, chunk_size: int, use_tqdm: bool = None):
    """
    Splits a dataframe into chunks of size `chunk_size`.
    Yields each chunk.
    """
    from tqdm import tqdm
    
    length = len(df)
    start = 0
    num_iterations = length // chunk_size + (1 if length % chunk_size != 0 else 0)

    assert chunk_size > 0, 'Chunk size must be greater than 0'
    
    if (num_iterations > 100 and use_tqdm is None) or use_tqdm:
        iterator = tqdm(list(range(0, length, chunk_size)))
    else:
        iterator = range(0, length, chunk_size)

    for start in iterator:
        end = min(start + chunk_size, length)
        yield df[start:end]
    
    # Example usage:
    # for chunk in chunk_dataframe(df, chunk_size):
    #     process(chunk)