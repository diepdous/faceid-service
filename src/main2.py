import csv
import tensorflow as tf
from align import detect_face
import facenet
import sys
import os
from os.path import expanduser
import copy
import cv2
import numpy as np
from scipy import spatial

import urllib.request
import base64
import uuid
from app import app

from PIL import Image

from flask import Flask, request, redirect, jsonify, Blueprint, render_template, Response

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

def is_not_blank(s):
    return bool(s and s.strip())

def load_embs(csv_file_name):
    csv_file=open(csv_file_name, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=',')

    nrows = 0
    for row in csv_reader:
        nrows += 1
    vEmb = np.zeros((nrows - 1, 128), dtype='float32')
    vID = np.zeros(nrows - 1, dtype='int')

    csv_file.close()

    csv_file=open(csv_file_name, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=',')

    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            for j in range(128):
                vEmb[line_count - 1][j] = float(row[j])
            vID[line_count - 1] = int(row[128])
            line_count += 1
    csv_file.close()
    return vEmb, vID

def load_user(file_name):
    file = open(file_name, 'r')
    Lines = file.readlines()

    count = 0
    aTmp = {}
    # Strips the newline character
    for line in Lines:
        count += 1
        # print("Line{}: {}".format(count, line.strip()))
        txt=line.strip()
        x = txt.split("#")
        # print(x[0])
        # print(x[1])
        aTmp[x[0]]=str(x[1])
    return aTmp

def load_mtcnn():
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return  pnet, rnet, onet

def align_one_face(image_file_name,pnet, rnet,onet,image_size=160, margin=11):
    img_list = []
    box_list = []
    #img = cv2.imread(os.path.expanduser(image))[:, :, ::-1]
    img = cv2.imread(image_file_name)
    #huy
    #cv2.imshow("def align_face",img)
    #cv2.waitKey()
    ########################################

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return None, None, None
    for box in bounding_boxes:
        det = np.squeeze(box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(cropped[:, :, ::-1],(image_size, image_size))[:, :, ::-1]
        prewhitened = facenet.prewhiten(aligned)
        #huy
        Ic = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        #cv2.imshow("def align_face",aligned)
        #cv2.imshow("def align_face",Ic)
        #cv2.waitKey()
        ########################################
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return img,images,bounding_boxes

def embedding(images):
    # check is model exists
    model_path = 'facemodel.pb'
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}
            embs = sess.run(embeddings, feed_dict=feed_dict)
            #print(emb)

    return embs

def UpdateNbest(d,id,vDist_best,vID_best):
    nbest=vDist_best.shape[0]
    for pos in range(nbest):
        if vDist_best[pos] > d:
            for p2 in range(1,nbest-pos):
                pp=nbest-p2
                vDist_best[pp]  = vDist_best[pp - 1]
                vID_best[pp] = vID_best[pp - 1]
            vDist_best[pos] = d
            vID_best[pos] = id
            break;
    return vDist_best, vID_best

def get_ID_by_KNN(one_emb,vEmb,vID,nbest,thresh):
    countItem=vEmb.shape[0]
    aDist=np.zeros((countItem,1),dtype='float32')

    vID_nbest=np.zeros((nbest,1),dtype='int32')
    vDist_nbest=np.zeros((nbest,1),dtype='float32')

    for k in range(nbest):
        vDist_nbest[k] = float("inf")
        vID_nbest[k] = -1

    for i in range(countItem):
        #d_cos=1 - spatial.distance.cosine(one_emb, vEmb[i])
        d_cos=spatial.distance.cosine(one_emb, vEmb[i])
        aDist[i]=1.0-d_cos
        vDist_nbest, vID_nbest=UpdateNbest(d_cos, vID[i], vDist_nbest, vID_nbest)

    vID_nbest_unique,counts = np.unique(vID_nbest,return_counts=True)
    a=np.where(counts==max(counts))
    faceID=vID_nbest_unique[a[0][0]]
    ok=0
    for i in range(countItem):
        if (vID[i]==faceID) & (aDist[i] > thresh):
            ok=1
            break
    if ok==1:
        return faceID
    else:
        return -1

def tag_one_face_image_knn(image_file_name,pnet, rnet,onet,vEmb_group,vID_group,image_size=160, margin=11,nbest=5,thresh=0.7,drawing=False):

    I_org,images,bounding_boxes=align_one_face(image_file_name,pnet, rnet,onet,image_size,margin)

    if bounding_boxes is None:
        return None,None,None,None

    count_face=len(bounding_boxes)
    if count_face < 1:
        return None,None,None,None

    embs=embedding(images)
    aFaceID=np.zeros((count_face,1),dtype='int32')
    for i in range(count_face):
        aFaceID[i] = get_ID_by_KNN(embs[i],vEmb_group,vID_group,nbest=5,thresh=0.7)

    return I_org,None,bounding_boxes,aFaceID

# Xac dinh nhom
vEmb_group,vID_group=load_embs('face.csv')
# Lay danh sach Users
aFaceName=load_user('FaceNames.txt')
# Load model cho face crop
pnet, rnet, onet=load_mtcnn()

@app.route('/')
def index():
    return "Face365 - Sign in"

@app.route('/upload', methods=['POST'])
def upload_file():
    fid = int(request.form.get('fid'))
    if fid <= 0:
        resp = jsonify({'status': 'false', 'message' : 'fid is required', 'fid' : fid})
        resp.status_code = 400
        return resp
    file_data = request.form.get('file')
    if not is_not_blank(file_data):
        resp = jsonify({'status': 'false', 'message' : 'file not found'})
        resp.status_code = 400
        return resp
    # file = base64.b64decode(file_data)
    # img = imread(io.BytesIO(base64.b64decode(file_data)))
    filename = str(uuid.uuid4()) + ".png";
    saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(saved_file, "wb") as fh:
        fh.write(base64.b64decode(file_data))

    I_org,aFaceCrop,bounding_boxes,aFaceID = tag_one_face_image_knn(saved_file,pnet, rnet,onet,vEmb_group,vID_group,image_size=160, margin=11,nbest=5,thresh=0.7,drawing=True)
    if aFaceID is None:
        resp = jsonify({'status': 'false', 'message' : 'Can not detect face'})
        resp.status_code = 400
        return resp

    sFaceID = []
    sFaceName = []
    iFaceID = 0
    for i in range(len(aFaceID)):
        if(aFaceID[i][0] > 0):
            sFaceID.append(str(aFaceID[i][0]))
            sFaceName.append(aFaceName[str(aFaceID[i][0])])
            iFaceID = int(aFaceID[i][0])

    if(iFaceID == fid):
        resp = jsonify({'status': 'true', 'message' : 'success', 'iFaceID' : iFaceID, 'lstFaceId' : ','.join(sFaceID), 'lstFaceName' : ','.join(sFaceName)})
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'status': 'false', 'message' : 'success', 'iFaceID' : iFaceID, 'lstFaceId' : ','.join(sFaceID), 'lstFaceName' : ','.join(sFaceName)})
        resp.status_code = 200
        return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True,host='0.0.0.0',port=port)