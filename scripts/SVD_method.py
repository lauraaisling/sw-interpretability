# import libraries
import sys
sys.path.append('./scripts/')
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.axes as axes
import seaborn as sns
import math
import copy
import numpy as np
sns.set_style("darkgrid")
from PIL import Image
import random # random seed to reproduce MDS and t-SNE plots

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster # k-Means clustering
from sklearn.cluster import KMeans
from sklearn import manifold # MDS and t-SNE
from sklearn.metrics import silhouette_score # silhouette width for clustering
from sklearn import preprocessing # scaling attributes
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import umap
import json 

import torch
import torchvision

from lucent.optvis import render, param, transform, objectives

import importlib as imp
import my_datasets
import utilities 
imp.reload(my_datasets) 
imp.reload(utilities) 
import argparse


parser = argparse.ArgumentParser(description='Arguments for run')
parser.add_argument("--min", help="Min neuron", type=int, required=True)
parser.add_argument("--max", help="Max neuron", type=int, required=True)

args = parser.parse_args()


plt.rcParams["figure.figsize"] = (3,3)
random.seed(2023)

# taking ImageNet dataset
# dataset='ilsvrc12'
dataset='ilsvrc12fine'
paths, count, y, idx_to_labels = my_datasets.get_dataset(dataset)

# print(count, len(paths)) # 1281167 1281167 (for ilsvrc12fine)

# For ilsvrc12fine dataset, paths are mapped differently
if dataset=='ilsvrc12fine':
    idxs=np.arange(0, 1281167, 10) # 1281167
    classes=np.unique(y[idxs])
    ppaths=[paths[i] for i in idxs]
    paths=ppaths

# where to save files
layer='Mixed_7b.cat_2'
SAVEFOLD0=f"../outputs/{dataset}"
# SAVEFOLD0=f"/home/laura/sw-interpretability/scripts/outputs/{dataset}"
SAVEFOLD=f"{SAVEFOLD0}/{layer}/"

if not os.path.exists(f'{SAVEFOLD}outputs/'):
    os.mkdir(f'{SAVEFOLD}outputs/')

if not os.path.exists(f'{SAVEFOLD}outputs/evecs/'):
    os.mkdir(f'{SAVEFOLD}outputs/evecs/')

if not os.path.exists(f'{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/'):
    os.mkdir(f'{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}')

predictions=np.load(f"{SAVEFOLD}/predictions.npy", mmap_mode = 'r')
conv_maps=np.load(f"{SAVEFOLD}/conv_maps.npy", mmap_mode = 'r')

# Global average pool activations since conv maps
conv_maps_avg = conv_maps.mean(3).mean(2)

# uncomment if you haven't already done SVD
# pu, ps, pvh = np.linalg.svd(conv_maps_avg)
# np.save(f"{SAVEFOLD}/pu.npy", pu)
# np.save(f"{SAVEFOLD}/ps.npy", ps)
# np.save(f"{SAVEFOLD}/eigenvectors.npy", pvh)

pvh = np.load(f'{SAVEFOLD}/eigenvectors.npy')
# pu = np.load(f'{SAVEFOLD}/pu.npy')
ps = np.load(f'{SAVEFOLD}/ps.npy')

activations = conv_maps_avg

# image collection params
top = 100

# clustering params
linkage='ward'
metric='euclidean'
distance_threshold = 15

kmeans_outlier_threshold = 15
min_ims_cluster = 5

agg_clusters = []
final_clusters = []
concept_sims = {}
concepts = {}


for direction in range(args.min, args.max):
    # Step 1: Calculate top images and extract activations for selected direction 
    # in this case direction is evec direction
    evec_dot = np.empty([len(conv_maps_avg)])
    evec_sim = np.empty([len(conv_maps_avg)])
    for i in range(len(conv_maps_avg)):
        evec_dot[i] = np.dot(conv_maps_avg[i], pvh[direction])
        evec_sim[i] = evec_dot[i]/(np.linalg.norm(conv_maps_avg[i])*np.linalg.norm(conv_maps_avg[direction]))

    top_ims = evec_dot.argsort()[-top:][::-1]
        
    step_1_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_1_max_projs.png"
    if not os.path.exists(step_1_f):
        fig, ax = plt.subplots(math.ceil(top//5), 5, figsize = (10,20))
        ax = ax.flatten()
        for idx, im_id in enumerate(top_ims):# enumerate(concepts_dot[:,concept].argsort()[-top:][::-1]):
            im = Image.open(paths[im_id])
            ax[idx].imshow(im)
            ax[idx].set_title(f"{im_id}", size = 8)
            ax[idx].axis('off')
        fig.savefig(step_1_f, bbox_inches="tight") 
        plt.close()


    # Step 2: Find number of clusters using agglomerative clustering

    top_activations = utilities.get_activations(activations_avg = activations, ims=top_ims)

    # First, just look at dendrogram
    # clusterer_0 = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric=metric,linkage=linkage )
    # clusterer_0.fit_predict(top_activations)
    # Cluster with set distance threshold
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric=metric,linkage=linkage)
    clusterer.fit_predict(top_activations)

    # get dendrogram
    step_2_linkage_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_2_linkage.png"
    if not os.path.exists(step_2_linkage_f):
        linkage_res = utilities.plot_dendrogram(clusterer, truncate_mode="level") # , p=100 # 
        fig = plt.figure(figsize=(10,5))
        dendrogram(linkage_res, truncate_mode="level")
        fig.savefig(step_2_linkage_f, bbox_inches="tight") 
        plt.close()

    # Visualise how hierarchial clustering clusters number of clusters selected
    clu_labs = clusterer.labels_
    # print(clu_labs)
    clu_lab_order = sorted(range(len(clu_labs)), key=lambda k: clu_labs[k])

    step_2_cluster_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_2_agg_clusters.png"
    if not os.path.exists(step_2_cluster_f):
        fig, ax = plt.subplots(math.ceil(len(top_ims)//5), 5, figsize = (10,20))
        ax = ax.flatten()
        for idx, im_id in enumerate(top_ims[clu_lab_order]):
            im = Image.open(paths[im_id])
            ax[idx].imshow(im)
            ax[idx].set_title(f"{im_id}: cluster {clu_labs[clu_lab_order][idx]}", size = 8)
            ax[idx].axis('off')
        fig.savefig(step_2_cluster_f, bbox_inches="tight")
        plt.close()

    agg_clusters.append(clusterer.n_clusters_)
    # Step 3: Run kmeans with number of clusters from step 2, remove outliers
    # Cluster with number of clusters selected with kmeans since need centroids to remove outliers

    kmeans = KMeans(n_clusters=clusterer.n_clusters_, random_state=0, n_init=5, max_iter=1000).fit(top_activations)
    # plot kmeans clusters before removing outliers
    clu_labs = kmeans.labels_
    # print(clu_labs)
    clu_lab_order = sorted(range(len(clu_labs)), key=lambda k: clu_labs[k])

    step_3_kmeans_raw_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_3_kmeans_raw.png"
    if not os.path.exists(step_3_kmeans_raw_f):
        fig, ax = plt.subplots(math.ceil(len(top_ims)//5), 5, figsize = (10,20))
        ax = ax.flatten()
        for idx, im_id in enumerate(top_ims[clu_lab_order]):
            im = Image.open(paths[im_id])
            ax[idx].imshow(im)
            ax[idx].set_title(f"{im_id}: cluster {clu_labs[clu_lab_order][idx]}", size = 8)
            ax[idx].axis('off')
        fig.savefig(step_3_kmeans_raw_f, bbox_inches="tight")
        plt.close()
    

    # Get squared distance to kmeans centroid of appropriate cluster - transform()
    centroid_dist = kmeans.transform(top_activations)**2
    nearest_centroid_dist = np.zeros(len(clu_labs))
    nearest_centroid_dist = [centroid_dist[i,clu_labs[i]] for i in range(len(clu_labs))]

    # Calculate which images are outliers by removing observations far away from centroid
    # Rename outliers to now belong to cluster label '100'. They will now be ordered at the end.

    clu_labs_rm_outliers = np.empty(len(clu_labs))
    clu_labs_rm_outliers[:] = 100
    for i in range(len(clu_labs)):
        if nearest_centroid_dist[i] < kmeans_outlier_threshold:
            clu_labs_rm_outliers[i] = clu_labs[i]

    # Remove clusters with less than treshold number of items
    for cluster in range(clusterer.n_clusters_):
        count = sum(clu_labs_rm_outliers == cluster)
        # print(count)
        if count < min_ims_cluster: 
            # print(cluster)
            #clu_labs_rm_outliers[i] = clu_labs[i]
            clu_labs_rm_outliers[[clu_labs_rm_outliers[i] == cluster for i in range(len(clu_labs_rm_outliers))]] = 100
    
    # visualise images remaining in clusters after removing small clusters
    clu_labs_rm_outliers_order = sorted(range(len(clu_labs_rm_outliers)), key=lambda k: clu_labs_rm_outliers[k])

    step_3_kmeans_rm_outliers_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_3_kmeans_rm_outliers.png"
    if not os.path.exists(step_3_kmeans_rm_outliers_f):
        fig, ax = plt.subplots(math.ceil(len(top_ims)//5), 5, figsize = (10,20))
        ax = ax.flatten()
        for idx, im_id in enumerate(top_ims[clu_labs_rm_outliers_order]):
            im = Image.open(paths[im_id])
            ax[idx].imshow(im)
            ax[idx].set_title(f"{im_id}: cluster {clu_labs_rm_outliers[clu_labs_rm_outliers_order][idx]}", size = 8)
            ax[idx].axis('off')
        fig.savefig(step_3_kmeans_rm_outliers_f, bbox_inches="tight")
        plt.close()

    # Plot cosine similarities by cluster
    cosine_sim = utilities.plot_cosine_similarities(top_ims, min_sim=0, max_sim=1, maps = activations, label = 'all')
    # cosine_sim = utilities.plot_cosine_similarities(top_ims[clu_labs_rm_outliers_order], min_sim=0, max_sim=1, maps = activations, label = 'ordered')
    step_3_cosine_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_3_kmeans_rm_outliers.png"
    if not os.path.exists(step_3_cosine_f):
        if list(clu_labs_rm_outliers).count(100) == top:  # evaluates to True if all removed from clusters
            pass # if clusters empty
        else:
            cosine_sim = utilities.plot_cosine_similarities(top_ims[clu_labs_rm_outliers_order][:top-list(clu_labs_rm_outliers).count(100)], 
                                                            min_sim=0, max_sim=1, maps = activations, label = 'ordered')
            ax = plt.subplot()
            im = ax.imshow(cosine_sim, cmap='viridis', interpolation='nearest', vmin=0, vmax=1) 
            plt.title(f"Cosine similarities")
            plt.subplots_adjust(right=0.8)
            cbar_ax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(mappable=(im), cax=cbar_ax)
            plt.show()
            fig.savefig(step_3_cosine_f, bbox_inches="tight")
            plt.close()

    # Visualise embeddings of images remaining in clusters
    if list(clu_labs_rm_outliers).count(100) == top:
        pass
    else:
        step_3_UMAP_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_3_UMAP.png"
        if not os.path.exists(step_3_UMAP_f):
            XY_UMAP = umap.UMAP(n_components=2).fit_transform(top_activations[clu_labs_rm_outliers != 100])
            amount = 0#.05
            fig, ax = plt.subplots(figsize = (10,10))
            #ax.set_title("UMAP")
            ax.scatter(XY_UMAP[:,0], XY_UMAP[:,1]) 

            for x0, y0, path in zip(utilities.rand_jitter(XY_UMAP[:,0], amount), utilities.rand_jitter(XY_UMAP[:,1], amount), [paths[i] for i in top_ims[clu_labs_rm_outliers != 100]]):
                ab = AnnotationBbox(utilities.getImage(path), (x0, y0), frameon=False)
                ax.add_artist(ab)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            fig.savefig(step_3_UMAP_f, bbox_inches="tight")
            plt.close()

    
    # Step 4: Calculate concept vectors
    
    concept_vecs = []
    for cluster in np.unique(clu_labs_rm_outliers):
        if cluster == 100: 
            pass
        else:
            concept_vecs.append(top_activations[clu_labs_rm_outliers==cluster].mean(0)/np.linalg.norm(top_activations[clu_labs_rm_outliers==cluster].mean(0)))
    
    final_clusters.append(len(concept_vecs))
    for i in range(len(concept_vecs)): 
        # print("cosine similarity: ", np.dot(concept_vecs[i], 
        concepts_dot = np.empty([len(concept_vecs),len(concept_vecs)])
        concepts_sim = np.empty([len(concept_vecs),len(concept_vecs)])
        for i in range(len(concept_vecs)):
            for j in range(len(concept_vecs)):
                concepts_dot[i,j] = np.dot(concept_vecs[i], concept_vecs[j])
                # same thing as concepts normalised length 1 
                concepts_sim[i,j] = concepts_dot[i,j]/(np.linalg.norm(concept_vecs[i])*np.linalg.norm(concept_vecs[j])) 
    # print(concepts_dot)

    # Maximally projecting images along concept directions
    # Find images with largest projection along concept direction. This is effecttively finding the maximally activating images for these directions.

    concepts_ims_dot = np.empty([len(conv_maps_avg),len(concept_vecs)])
    concepts_ims_sim = np.empty([len(conv_maps_avg),len(concept_vecs)])
    for i in range(len(conv_maps_avg)):
        for concept_id in range(len(concept_vecs)):
            concepts_ims_dot[i,concept_id] = np.dot(conv_maps_avg[i], concept_vecs[concept_id])
            concepts_ims_sim[i,concept_id] = concepts_ims_dot[i,concept_id]/(np.linalg.norm(conv_maps_avg[i])*np.linalg.norm(conv_maps_avg[concept_id]))

    
    top_projs = []
    for concept in range(len(concepts_ims_dot[0,])):
        top_projs.append(concepts_ims_dot[:,concept].argsort()[-top:][::-1])

    # concept file
    concept_vec_li = []
    for concept in range(len(concept_vecs)):
        concept_vec_li.append(concept_vecs[concept].tolist())
        step_4_projections_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/dir_{direction}_step_4_projection_{concept}.png"
        if not os.path.exists(step_4_projections_f):
            fig, ax = plt.subplots(math.ceil(top//5), 5, figsize = (10,20))
            ax = ax.flatten()
            for idx, im_id in enumerate(top_projs[concept]):# enumerate(concepts_dot[:,concept].argsort()[-top:][::-1]):
                im = Image.open(paths[im_id])
                ax[idx].imshow(im)
                ax[idx].set_title(f"{im_id}", size = 8)
                ax[idx].axis('off')
            fig.savefig(step_4_projections_f, bbox_inches="tight")
            plt.close()


    # cosine similarity with concept vectors and SVD direction 
    SVD_sims = []
    for concept in range(len(concept_vecs)):
        SVD_sim = np.dot(concept_vecs[concept], pvh[direction])
        SVD_sim = SVD_sim/(np.linalg.norm(concept_vecs[concept])*np.linalg.norm(pvh[direction]))
        SVD_sims.append(SVD_sim)
    concept_sims[direction] = SVD_sims# .tolist()

    # print(concept_vec_li)
    # save concepts
    concepts[direction] = concept_vec_li # .tolist()
    

concepts_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/concepts_{args.min}-{args.max-1}.json"  
with open(concepts_f, 'w') as json_file:
    json.dump(concepts, json_file) 

similarities_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/similarities_{args.min}-{args.max-1}.json"  
with open(similarities_f, 'w') as json_file:
    json.dump(concept_sims, json_file)

agg_clusters_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/agg_clusters_{args.min}-{args.max-1}.txt"     
with open(agg_clusters_f, 'w') as fp:
    fp.write(''.join(str(item)+ "\n" for item in agg_clusters))

final_clusters_f = f"{SAVEFOLD}outputs/evecs/{str(args.min)}_{str(args.max-1)}/final_clusters_{args.min}-{args.max-1}.txt"     
with open(final_clusters_f, 'w') as fp:
    fp.write(''.join(str(item)+ "\n" for item in final_clusters))


# python SVD_method.py --min=0 --max=100 

