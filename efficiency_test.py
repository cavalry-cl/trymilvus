from gen import generate_vector
import database_manager as dm
import time
import multiprocessing
import random
import numpy as np
from pymilvus import FieldSchema, DataType, CollectionSchema
import json
import os


def search(client, collection_name, dim):
    # client.load_collection(collection_name=collection_name)
    q_vec_0 = [0 for _ in range(dim)]
    search_res = client.search(
        collection_name=collection_name,
        data=[q_vec_0],
        limit=3,
    )
    # result = [(x["id"],x["distance"]) for x in search_res[0]]
    # print(result)
    
def batch_search(client, collection_name, batch_size, dim):
    # client.load_collection(collection_name=collection_name)
    q_vecs = generate_vector(cnt=batch_size, dim=dim).tolist()
    search_res = client.search(
        collection_name=collection_name,
        data=q_vecs,
        limit=3,
    )
    # client.release_collection(collection_name=collection_name)
    # result = [(x["id"],x["distance"]) for x in search_res[0]]
    # print(result)
    


if __name__=="__main__":

    
    # settings
    uri = "http://localhost:19530"
    db_name = "efficiency_test"


    dim = 512
    batch_cnt = 200
    batch_size = 2500
    q_batch_size = 10

    collection_name = "test_HNSW"
    # initialize collection
    client = dm.initialize_client(uri, db_name)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vector")
    fields = [id_field, vector_field]
    schema = CollectionSchema(fields=fields)
    col = dm.initialize_collection(client, collection_name, schema)
    dm.create_scalar_index(client=client, collection_name=collection_name, field_name="id")
    dm.create_vector_index(client=client, collection_name=collection_name, field_name="vector", index_type="HNSW", params={"efConstruction":16, "M":8})



    t1 = time.time()
    client.load_collection(collection_name=collection_name)
    t2 = time.time()
    print(f'It takes {t2-t1}s to load.')

    # insert 1000 entries
    insert_costs = []
    query_costs = []
    upsert_costs = []
    for i in range(batch_cnt):
        t1 = time.time()
        vectors = generate_vector(cnt=batch_size, dim=dim)
        data = dm.create_entries(vectors=vectors, ids=range(i*batch_size, (i+1)*batch_size))
        col.insert(data=data)
        t2 = time.time()
        insert_costs.append((t2-t1)/batch_size)
        if i % 20 == 0:
            print(f'Inserting the {i}-th batch cost {t2-t1}s.')

        t1 = time.time()
        vectors = generate_vector(cnt=batch_size, dim=dim)
        data = dm.create_entries(vectors=vectors, ids=range(i*batch_size, (i+1)*batch_size))
        col.upsert(data=data)
        t2 = time.time()
        upsert_costs.append((t2-t1)/batch_size)
        if i % 20 == 0:
            print(f'Upserting the {i}-th batch cost {t2-t1}s.')

        t1 = time.time()
        batch_search(client=client, collection_name=collection_name, batch_size=q_batch_size, dim=dim)
        t2 = time.time()
        query_costs.append((t2-t1)/q_batch_size)
        if i % 20 == 0:
            print(f'Query costs {t2-t1}s.')



    client.release_collection(collection_name=collection_name)
    client.drop_collection(collection_name=collection_name)
    # print(f'{} entities are inserted.')

    # client.drop_collection(collection_name=collection_name)

    np.save('insert_costs.npy', np.array(insert_costs))
    np.save('query_costs.npy', np.array(query_costs))
    np.save('upsert_costs.npy', np.array(upsert_costs))
