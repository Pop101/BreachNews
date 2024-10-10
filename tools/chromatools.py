from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb

import pickle

def save_collection(collection: "Collection", filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(collection.get(), f)
        
def load_collection(filename: str, chroma=None, collection=None) -> "Collection":
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)
        
    if chroma is None:
        chroma = chromadb.Client()
    
    _col = chroma.get_or_create_collection(collection)
    
    _col.upsert(
        ids=saved_data['ids'],
        embeddings=saved_data['embeddings'],
        metadatas=saved_data['metadatas'],
        documents=saved_data['documents']
    )
    
    return _col