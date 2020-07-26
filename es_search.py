import glob, re, pandas as pd
import numpy as np
from mtcnn.mtcnn import MTCNN
from scipy import misc
from keras_facenet import FaceNet
import base64
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from flask import Flask, jsonify, request
import json, ujson
import imageio, io
from PIL import Image

#Face localization
detector = MTCNN()
#Embedding build
embedder = FaceNet()
#Elastic search instance
ES_HOST = {"host":"localhost", "port":9200}
es = Elasticsearch(hosts=[ES_HOST])

#Vector to base64String
dfloat32 = np.dtype('>f4')
def encode_array(arr):
    b64str = base64.b64encode(arr.astype(dfloat32)).decode("utf-8")
    return b64str


#Create New Index
def create_index(INDEX_NAME):
    try:
        if es.indices.exists(INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)

        es.indices.create(index=INDEX_NAME)

        col_map = {
            "properties":{
                "embedding_vector":{
                    "type": "binary",
                  "doc_values": True
                }
            }
        }

        es.indices.put_mapping(index=INDEX_NAME, body=col_map)
        return 1
    except Exception as e:
        print(e)
        return 0


#Return prominent face in image
def detectFace(img):
    bb = detector.detect_faces(img)
    bbs = []
    for x in bb:
        if x['confidence'] > 0.9:
            bbs.append(x)
    
    print('&&&', len(bbs))
    if len(bbs) == 0:
        return [], False
    
    bounding_box = bbs[0]['box']
    h_adj = int(0.0002*bounding_box[3])
    w_adj = int(0.0002*bounding_box[2])
    cropp = img[max(bounding_box[1]-h_adj, 0):max(bounding_box[1]+bounding_box[3]+h_adj, 0), max(bounding_box[0]-w_adj, 0):max(bounding_box[0]+bounding_box[2]+w_adj,0)]
    return cropp, True

#Image embedding extraction
def embedding_extract(img):
    try:
        c,f = detectFace(imageio.imread(io.BytesIO(base64.b64decode(img))))
        if f:
            embds = embedder.embeddings([c])[0]
            return embds
        else:
            return np.array([])
    except:
        return np.array([])

#Search query
def search_doc(emd, idi, INDEX_NAME):
    query = {
              "query": {
                "function_score": {
                  "boost_mode": "replace",
                  "script_score": {
                    "script": {
                      "source": "binary_vector_score",
                      "lang": "knn",
                      "params": {
                        "cosine": True,
                        "field": "embedding_vector",
                        "vector": emd.tolist()
                      }
                    }
                  }
                }
              },
              "size": 1
            }
    resp =es.search(index=INDEX_NAME, body=query)
    return resp

#Index an image
def index_doc(em, idi, INDEX_NAME):
    try:
        em = encode_array(em)
        es.index(index=INDEX_NAME, body={"embedding_vector": em, "id":idi})
        return 1
    except:
        return 0

def image_search(dic):
    img = dic['image']
    i = dic['id']
    loc = dic['location']

    em = embedding_extract(img)
    print('here')
    if len(em) == 0:
        print("No face")
        return {'status': 'other', 'message': 'no face', 'id': None, 'score': None, 'status_code': 1}
    else:
        resp = search_doc(em, i, loc)
        if len(resp['hits']['hits']) == 0:
            st = index_doc(em, i, loc)
            if st == 1:
                print('Indexed')
                return {'status': 'non-existing', 'message': 'image indexed', 'id': None, 'score': None, 'status_code': 0}
            else:
                print('Indexing error')
                return {'status': 'other', 'message': 'indexing error','id': None, 'score': None, 'status_code': 0}
        else:
            scr = max([x['_score'] for x in resp['hits']['hits']])
            if max([x['_score'] for x in resp['hits']['hits']]) < 0.8:
                st = index_doc(em, i, loc)
                if st == 1:
                    print('Indexed')
                    return {'status': 'non-existing', 'message': 'image indexed', 'id': None, 'score': resp['hits']['hits'][0]['_score'], 'status_code': 0}
                else:
                    print('Indexing error')
                    return {'status': 'non-existing', 'message': 'indexing error', 'id': None, 'score': resp['hits']['hits'][0]['_score'], 'status_code': 0}
            else:
                return {'status': 'existing', 'message': 'match found', 'id': resp['hits']['hits'][0]['_source']['id'], 'score': resp['hits']['hits'][0]['_score'], 'status_code': 1}

def driver(dic):
    loc = dic['location']
    if es.indices.exists(loc):
        print('t1')
        r = image_search(dic)
    else:
        k = create_index(loc)
        if k == 1:
            r = image_search(dic)
        else:
            r = {'status': 'other', 'message': 'index creation error', 'id': None, 'score': None, 'status_code': 0}
    return r


#=====================================================APP=========================================================================
app = Flask(__name__)

#Database reset
@app.route('/reset', methods=['GET'])
def reset_db():
    try:
        all_indices = es.indices.get('*').keys()
        all_indices = [x for x in all_indices if 'kibana' not in x.lower() and 'ptp' not in x.lower()]
        for i in all_indices:
            es.indices.delete(index=i)
            es.indices.create(index=i)
            col_map = {
                "properties":{
                    "embedding_vector":{
                        "type": "binary",
                        "doc_values": True
                    }
                }
            }
            es.indices.put_mapping(index=i, body=col_map)
        return jsonify(result={"status": 200})
    except Exception as e:
        print(e)
        return jsonify(result={"status": 400})

@app.route('/search', methods=['POST'])
def apicall():
    try:
        d = request.get_data().decode("utf-8")
        test_json = ujson.loads(d)
        test_json['location'] = test_json['location'].lower()
        print(test_json.keys())

    except Exception as e:
        print('**', e)
        responses = jsonify(**{'status': 'other', 'message': 'Error in input', 'id': None, 'score': None, 'status_code': 0})
        responses.status_code = 400
        return responses

    try:
        a = driver(test_json)
        print('********', a)
        responses =  jsonify(**a)
        print('###############', responses)
        #responses = jsonify(predictions=numFaces,**test_json)
        #print numFaces
        responses.status_code = 200
        return responses
    except Exception as e:
        print(e)
        responses = jsonify(**{'status': 'other', 'message': 'Error in processing', 'id': None, 'score': None, 'status_code': 0})
        responses.status_code = 500
        return responses
#app.run(host='0.0.0.0', port=8890,debug=False)                    
