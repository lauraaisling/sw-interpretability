import os
import numpy as np
import json
import torch
import torchvision
import my_datasets
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import pairwise_distances

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_activations(activations_avg, ims = None, direction = None):
    if (ims is not None) and (direction is not None):
        activations = activations_avg[ims,:]
        if activations.ndim == 1:
            activations = activations[direction]
        else:
            activations = activations[:, direction]
    elif (direction is not None):
        activations = activations_avg[:,direction]
    elif (ims is not None): 
        activations = activations_avg[ims,:]
    return activations


def get_preds(paths, y, model, batch_start=0, batch_size=32):
    softmaxf = torch.nn.Softmax(dim=1)
    p1 = np.zeros((len(y),1000))
    while batch_start+batch_size < len(y)+batch_size: 
            # preprocessing the inputs 
            inputs = torch.stack([my_datasets.transform_normalize(my_datasets.transform(Image.open(paths[i]).convert("RGB"))) for i in range(batch_start, min(batch_start+batch_size, len(y)))])
            inputs = inputs.clone().detach().requires_grad_(True)
            batch_y=y[batch_start:min(batch_start+batch_size, len(y))]

            # transfering to GPU
            inputs=inputs.to(device)
            y1=model(inputs)

            p1[batch_start:min(batch_start+batch_size, len(y)),:]=softmaxf(y1).detach().cpu()
            batch_start+=batch_size
    return p1


def check_acc(preds, y):
    num_correct = 0
    for idx in range(len(y)):
        if np.argmax(preds[idx]) == y[idx]:
            num_correct+=1
    acc = num_correct / len(y)
    return acc


# Scatterplot to visualise clusters
colors = np.array(['orange', 'blue', 'lime', 'khaki', 'pink', 
                   'green', 'purple', 'yellow'])

# points - a 2D array of (x,y) coordinates of data points
# labels - an array of numeric labels in the interval [0..k-1], one for each point
# centers - a 2D array of (x, y) coordinates of cluster centers
# title - title of the plot
def clustering_scatterplot(points, labels, centers, title):
    n_clusters = np.unique(labels).size
    for i in range(n_clusters):
        h = plt.scatter(points[labels==i,0],
                        points[labels==i,1], 
                        c=colors[i%colors.size],
                        label = 'cluster '+str(i))
    # plot the centers of the clusters
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='r', marker='*', s=500)
    
    _ = plt.title(title)
    _ = plt.legend()
    _ = plt.xlabel('x')
    _ = plt.ylabel('y')


# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,5))
    dendrogram(linkage_matrix, **kwargs)
    plt.close()
    return linkage_matrix


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html 
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
def plot_cosine_similarities(top_ims, maps, min_sim=0.2, max_sim=1, label = ''): 
    activations = get_activations(activations_avg = maps, ims = top_ims)
    activations_dot = np.empty([len(top_ims),len(top_ims)])
    activations_sim = np.empty([len(top_ims),len(top_ims)])
    for i in range(len(top_ims)):
        for j in range(len(top_ims)):
            activations_dot[i,j] = np.dot(activations[i], activations[j])
            activations_sim[i,j] = activations_dot[i,j]/(np.linalg.norm(activations[i])*np.linalg.norm(activations[j])) 

    ax = plt.subplot()
    im = ax.imshow(activations_sim, cmap='viridis', interpolation='nearest', vmin=min_sim, vmax=max_sim) 
    plt.title(f"Cosine similarities {label}")
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(mappable=(im), cax=cbar_ax)
    plt.show()
    plt.close()
    return activations_sim


def plot_euclidan_distances(top_activations, min_dist=None, max_dist=None, label = ''):
    distance_matrix = pairwise_distances(top_activations, metric = 'euclidean')
    #print(distance_matrix)
    
    ax = plt.subplot()
    im = ax.imshow(distance_matrix, cmap='viridis', interpolation='nearest', vmin=min_dist, vmax=max_dist) 
    plt.title(f"Euclidean distances {label}")
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(mappable=(im), cax=cbar_ax)
    plt.show() 
    plt.close()
    print(distance_matrix.max())
    print(distance_matrix.mean())
    return distance_matrix


# For UMAP plot
def getImage(path, zoom=0.2):
    return OffsetImage(plt.imread(path), zoom=zoom)

def rand_jitter(arr, amount):
    stdev = amount * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, ax, s=20, c='b', marker='o', cmap=None, norm=None, 
           vmin=None, vmax=None, alpha=None, linewidths=None, 
           verts=None, hold=None, **kwargs):
    return ax.scatter(rand_jitter(x, amount), rand_jitter(y, amount), 
                      s=s, c=c, marker=marker, cmap=cmap, 
                      norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, 
                      linewidths=linewidths, **kwargs)