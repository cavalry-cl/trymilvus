from pymilvus import MilvusClient, DataType, connections, db, Role, Collection, FieldSchema, CollectionSchema
import numpy as np

def initialize_client(uri, db_name):
    connections.connect(uri=uri)
    if db_name not in db.list_database():
        db.create_database(db_name)
    db.using_database(db_name)
    client = MilvusClient(uri=uri, db_name=db_name)
    return client

def initialize_collection(client, collection_name, schema):
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)
    return collection

def create_entries(vectors, ids=None):
    assert(ids == None or vectors.shape[0] == len(ids))
    n = vectors.shape[0]
    dim = vectors.shape[1]
    entries = []
    if ids == None:
        ids = np.arange(n) + 1
    for (id, vec) in zip(ids, vectors):
        dict = {"id": id, "vector": vec}
        entries.append(dict)
    return entries


def create_vector_index(client, collection_name, index_name="vector_index", field_name="embedding", metric_type="L2", index_type="FLAT", params={}):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=field_name,
        index_type=index_type,
        metric_type=metric_type,
        params=params,
        index_name=index_name,
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )


def create_scalar_index(client, collection_name, index_name="scalar_index", field_name="id", index_type="STL_SORT"):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=field_name,
        index_type=index_type,
        index_name=index_name,
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )
