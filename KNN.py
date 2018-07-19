import os
import h5py
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from sklearn import random_projection
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--maxpool_train', default='maxpool_train.npy', help='file with vectors from training')
parser.add_argument('--maxpool_test', default='maxpool_test.npy', help='file with vectors from testing')
parser.add_argument('--labels_file', default='data/mn10/ply_data_train.h5', help='file with test labels')
parser.add_argument('--n_components', type=int, default=40, help='dimensions to project on')
parser.add_argument('--n', type=int, default=2, help='n*k neighbours in projected space')
FLAGS = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_FILENAME_TRAIN = FLAGS.maxpool_train
EXTRACT_FILENAME_TEST = FLAGS.maxpool_test
DUMP_DIR = FLAGS.dump_dir
LABELS_FILE = FLAGS.labels_file
N_COMPS = FLAGS.n_components
N = FLAGS.n

# load original labels
# labels[i] corresponds to point_clouds[i]
f = h5py.File(os.path.join(BASE_DIR, LABELS_FILE))
labels = f['label'][:] #vector with entries 0-9 mapping to lines in shape_names.txt
point_clouds = f['data'][:]
f.close()

label_values = np.r_[0:10]

# load predicted values of test set
LOG = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'r')
lines = LOG.readlines()

pred_val = []
for line in range(len(lines)):
	lines[line],temp = lines[line].split(",")
	pred_val.append(lines[line])
pred_val = np.array(pred_val)

# load maxpool vectors
data = np.load(EXTRACT_FILENAME_TRAIN)
queries = np.load(EXTRACT_FILENAME_TEST)

train_vectors = []
test_vectors = []

for idx in range(len(data)):
	for batch in range(len(data[idx])):
		train_vectors.append(np.squeeze(data[idx][batch]))
for idx in range(len(queries)):
	for batch in range(len(queries[idx])):
		test_vectors.append(np.squeeze(queries[idx][batch]))

train_vectors = np.array(train_vectors)
test_vectors = np.array(test_vectors)

# project vectors to dimension 
transformer = random_projection.GaussianRandomProjection(n_components=N_COMPS)
train_new = transformer.fit_transform(train_vectors)
test_new = transformer.fit_transform(test_vectors)

kd = KDTree(train_new)

f1_10 = []
for q in range(len(test_new)):
	k = 10
	nk = 10*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_10.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_10[0])))
f1_10 = np.array(f1_10)
for col in range(len(f1_10[0])):
	means[col] = np.mean(f1_10[:][col])
total_mean_10 = np.mean(means)

f1_20 = []
for q in range(len(test_new)):
	k = 20
	nk = 20*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_20.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_20[0])))
f1_20 = np.array(f1_20)
for col in range(len(f1_20[0])):
	means[col] = np.mean(f1_20[:][col])
total_mean_20 = np.mean(means)

f1_50 = []
for q in range(len(test_new)):
	k = 50
	nk = 50*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_50.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_50[0])))
f1_50 = np.array(f1_50)
for col in range(len(f1_50[0])):
	means[col] = np.mean(f1_50[:][col])
total_mean_50 = np.mean(means)

f1_100 = []
for q in range(len(test_new)):
	k = 100
	nk = 100*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_100.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_100[0])))
f1_100 = np.array(f1_100)
for col in range(len(f1_100[0])):
	means[col] = np.mean(f1_100[:][col])
total_mean_100 = np.mean(means)

f1_200 = []
for q in range(len(test_new)):
	k = 200
	nk = 200*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_200.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_200[0])))
f1_200 = np.array(f1_100)
for col in range(len(f1_200[0])):
	means[col] = np.mean(f1_200[:][col])
total_mean_200 = np.mean(means)

f1_300 = []
for q in range(len(test_new)):
	k = 300
	nk = 300*N
	dists,neighbours = kd.query(test_new[q],nk)
	or_dists = []
	for n in range(len(neighbours)):
		dist = np.linalg.norm(test_vectors[q]-train_vectors[neighbours[n]]) # calculate euclidean distance (original space)
		or_dists.append([dist,neighbours[n]]) 						# between query and neighbours
	or_dists = sorted(or_dists) # sort couples of [distance,train_index]
	k_dists_i = np.array(or_dists[0:k]) # select first k couples
	k_neighbours = []
	for x in range(len(k_dists_i)): # extract indices
		k_neighbours.append(k_dists_i[x][1])
	k_neighbours = np.array(k_neighbours)
	or_labels = np.ones((k))*int(pred_val[q]) # array of k with predicted label of query
	k_labels = []
	for n in range(len(k_neighbours)):
		k_labels.append(labels[int(k_neighbours[n])])
	k_labels = np.array(k_labels)	
	f1_300.append(metrics.f1_score(or_labels, k_labels, labels=label_values, average=None))

means = np.zeros((len(f1_300[0])))
f1_300 = np.array(f1_300)
for col in range(len(f1_300[0])):
	means[col] = np.mean(f1_300[:][col])
total_mean_300 = np.mean(means)

total_means = [total_mean_10,total_mean_20,total_mean_50,total_mean_100,total_mean_200,total_mean_300]
max = np.max(total_means)

plt.plot([10,20,50,100,200,300],total_means)
plt.title(r'F1 score vs k', fontsize=20)
plt.axis([0, 310, 0, max])
plt.xlabel('k neighbours')
plt.ylabel('F1 score means')
plt.show()
