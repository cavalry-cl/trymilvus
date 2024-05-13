from gen import generate_vector
import database_manager as dm
import time
import multiprocessing
import random
import numpy as np
from pymilvus import FieldSchema, DataType, CollectionSchema
import json
import os
def convert_to_path(id):
    return "vec/" + str(id) + ".npy"


uri = "http://localhost:19530"
db_name = "test_multi"
collection_name = "test_multi"

def gen_with_delay(cnt, dim, mx, save_path):
    time.sleep(0.5)
    generate_vector(cnt, dim, mx, save_path)

def search(client, dim):
    client.load_collection(collection_name=collection_name)
    q_vec_0 = [0 for _ in range(dim)]
    search_res = client.search(
        collection_name=collection_name,
        data=[q_vec_0],
        limit=3,
    )
    result = [(x["id"],x["distance"]) for x in search_res[0]]
    print(result)
    
    client.release_collection(collection_name=collection_name)

def gen_test(cnt, dim, mx, latest_vector_id):
    client = dm.initialize_client(uri, db_name)
    for i in range(10):
        latest_vector_id.value = i
        # latest_vector_id.value = random.randint(0,1000000)
        gen_with_delay(cnt, dim, mx, convert_to_path(i))
        search(client, dim)
def upsert_test(latest_vector_id):
    client = dm.initialize_client(uri, db_name)
    ids = [[0,1,2,3,4],
           [4,1,2,0,7],
           [9,7,8,5,6],
           [0,5,9,8,7],
           [1,5,6,4,7],
           [0,1,9,8,7]]
    for i in range(6):
        time.sleep(5)
        save_path = convert_to_path(latest_vector_id.value)
        print("upsert "+save_path)
        vectors = np.load(save_path)
        data = dm.create_entries(vectors=vectors, ids=ids[i])
        client.upsert(collection_name=collection_name, data=data)


# hyper-parameters
dim = 16
mx = 3
cnt = 5

if __name__=="__main__":
    # initialize collection
    client = dm.initialize_client(uri, db_name)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim, description="vector")
    fields = [id_field, vector_field]
    schema = CollectionSchema(fields=fields)
    dm.initialize_collection(client, collection_name, schema)
    dm.create_scalar_index(client=client, collection_name=collection_name, field_name="id")
    dm.create_vector_index(client=client, collection_name=collection_name, field_name="vector")

    # insert 10 entries
    vectors = generate_vector(cnt=10, dim=dim, mx=mx)
    data = dm.create_entries(vectors=vectors)
    client.upsert(collection_name=collection_name, data=data)

    latest_vector_id = multiprocessing.Value("i", 0)
    t1 = time.time()
    print(client.list_collections())
    print(client.list_indexes(collection_name=collection_name))
    gen_process = multiprocessing.Process(target=gen_test, args=(cnt, dim, mx, latest_vector_id), name="gen_process")
    db_process = multiprocessing.Process(target=upsert_test, args=(latest_vector_id,), name="db_process")
    gen_process.start()
    db_process.start()
    gen_process.join()
    db_process.join()
    t2 = time.time()
    print(t2-t1)

