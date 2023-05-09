import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from  scipy.spatial.distance import cdist

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from enum import Enum

import random
import copy
import pickle

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')

########## BEGIN: Disatnce Methods ##########
def norm_matrix(x):
    return x/ np.expand_dims(np.sum(x, axis=-1), axis=-1)

def chi_square(a, b):
    nom = (a - b) ** 2
    denom = a + b
    return np.sum(np.divide(nom, denom, out=np.zeros_like(a), where=denom!=0))

def chi_square_norm(a, b):
    a = norm_matrix(a)
    b = norm_matrix(b)  
    return chi_square(a, b)

def chi_norm(a, b): 
    return np.sqrt(chi_square_norm(a, b))

def phi_square(a, b):
    return np.sqrt(chi_square(a, b) / (np.sum(a) + np.sum(b)))

def phi(a, b):
    return np.sqrt(phi_square(a, b))

########## END: Disatnce Methods ##########

########## BEGIN: Clustering Methods ##########

def cluster_data(X, n_clusters=2, metric=phi, linkage='average', distance_threshold=None):
    d = cdist(X, X, metric)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                         linkage=linkage, 
                                         metric='precomputed', 
                                         distance_threshold=distance_threshold)
    clustering.fit(d)
    return clustering.labels_

########## END: Clustering Methods ##########

########## BEGIN: Visualization Methods ##########
def visualize_clusters(scene_features, labels, fig_name='plot.pdf', title='Clutering Scenes based on their Voxel Class Counts'):
    pca = PCA(2)
    df = pca.fit_transform(scene_features)
    fig, ax = plt.subplots()
    int_labels, u_labels = labels_to_int(labels)
    int_labels = np.array(int_labels)
    for i, label in enumerate(u_labels):
        ax.scatter(df[int_labels == i , 0] , df[int_labels == i , 1] , label = label)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("First dimension of PCA(2) Scene Features.")
    ax.set_ylabel("Second dimension of PCA(2) Scene Features.")
    ax.legend(ncol=max(min(len(u_labels)//10,3),1), fancybox=True, framealpha=0.5)
    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        fig.show()
    plt.close(fig)
########## END: Visualization Methods ##########

def count_scene_classes(annotations, dataset_path, data_split='train_split', 
                                                    gt_dir_name='gts', 
                                                    labels_fname='labels.npz',
                                                    deleted_classes=[],
                                                    min_classes=18):
    deleted_classes = np.array(deleted_classes)
    class_counts = dict()
    for scene_name in annotations[data_split]:
        class_counts[scene_name] = np.zeros(min_classes - deleted_classes.shape[0])
        for frame_token in annotations['scene_infos'][scene_name].keys():
            label_data = np.load(os.path.join(dataset_path, gt_dir_name, scene_name, frame_token, labels_fname))
            # semantics = label_data['semantics'], mask_lidar = label_data['mask_lidar'], mask_camera = label_data['mask_camera']
            semantics = label_data['semantics']
            counts = np.bincount(semantics.flatten(), minlength=min_classes)
            if deleted_classes.shape[0] > 0:
                counts = np.delete(counts, deleted_classes, 0)
            class_counts[scene_name] += counts
    return class_counts

def to_class_scenes(labels, scenes, scenes_included=None):
    class_scenes = dict()
    for label, scene in zip(labels, scenes):
        if scenes_included is None or scene in scenes_included:
            if class_scenes.get(label) is None:
                class_scenes[label] = list()
            class_scenes[label].append(scene)
    return class_scenes

def stratified_sampling_scenes(class_scenes, sample_ratio=0.1, random_state=None):
# Perform stratified sampling to get 10% of the data from the top classes
    sampled_scenes = []
    for class_int_id, (class_id, scenes) in enumerate(class_scenes.items()):
        if len(scenes) == 0:
            continue  # Skip if no scenes have this class
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_ratio, random_state=random_state)
        X = np.zeros(len(scenes))
        y = [None] * len(scenes)
        for i in range(len(scenes)):
            X[i] = i
            y[i] = class_int_id
        selected_scenes = None
        try:
            _, idx = next(sss.split(X, y))
            selected_scenes = [scenes[i] for i in idx]
        except ValueError as ve:
            selected_scenes = scenes
        sampled_scenes.extend(selected_scenes)
    return list(set(sampled_scenes))

def random_sampling_scenes(scenes, sample_ratio=0.1):
    if sample_ratio <= 0 or sample_ratio >1:
        sample_ratio = 1.0
    scenes = copy.deepcopy(scenes)
    length = len(scenes)
    length_to_select = round(length*sample_ratio)
    random.shuffle(scenes)
    return scenes[:length_to_select]

def dict_to_array(data_dict):
    key_to_ind = dict()
    list_v = list()
    list_k = list()
    for i, (k,v) in enumerate(data_dict.items()):
        list_k.append(k)
        list_v.append(v)
        key_to_ind[k]=i
    return key_to_ind, list_k, list_v

def get_selected_scene_info(scenes, scene_to_ind, scene_fts_l, labels):
    new_fts = list()
    new_labels = list()
    for scene in scenes:
        ind = scene_to_ind[scene]
        new_fts.append(scene_fts_l[ind])
        new_labels.append(labels[ind])
    return new_fts, new_labels

def load_annotations(folder_path):
    # Load the annotations file
    with open(os.path.join(folder_path, 'annotations.json'), 'r') as f:
        return json.load(f)

def load_table(table_dir, table_name) -> dict:
    """ Loads a table. """
    with open(os.path.join(table_dir, '{}.json'.format(table_name))) as f:
        table = json.load(f)
    return table

def write_table(table, table_dir, table_name) -> dict:
    """ Loads a table. """
    os.makedirs(table_dir, exist_ok=True)
    with open(os.path.join(table_dir, '{}.json'.format(table_name)), mode='w') as f:
        json.dump(table, f, sort_keys=True, indent=4)
    return table

def write_annotations(annotations, folder_path, ann_fname='annotations.json'):
    # Save the updated annotations file
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, ann_fname), 'w') as f:
        json.dump(annotations, f, sort_keys=True, indent=4)

def remove_scenes(annotations, scenes, del_included=True):
    if del_included:
        scenes_to_delete = scenes
    else:
        scenes_to_delete = set(annotations['scene_infos'].keys()).difference(set(scenes))
    for scene in scenes_to_delete:
            del annotations['scene_infos'][scene]
    return annotations

def filter_list(list_dict, key:str, objs_to_keep=None, objs_to_delete=None):
    objs_to_keep = set(objs_to_keep) if objs_to_keep is not None else None
    objs_to_delete = set(objs_to_delete) if objs_to_delete is not None else None
    if objs_to_keep is not None and objs_to_delete is not None:
        objs_to_keep = objs_to_keep.difference(objs_to_delete)
        objs_to_delete = None

    new_list = list()
    for item in list_dict:
        if objs_to_keep is not None and item[key] in objs_to_keep: 
            new_list.append(item)
        if objs_to_delete is not None and item[key] not in objs_to_delete: 
            new_list.append(item)
    return new_list

def create_new_tables(tables_dir_src:str, tables_dir_dest:str, scenes_to_keep=None, scenes_to_delete=None):
    scenes_table = load_table(tables_dir_src, 'scene')
    if scenes_to_keep is not None or scenes_to_delete is not None:
        scenes_to_keep = set(scenes_to_keep) if scenes_to_keep is not None else None
        scenes_to_delete = set(scenes_to_delete) if scenes_to_delete is not None else None
        if scenes_to_keep is not None and scenes_to_delete is not None:
            scenes_to_keep = scenes_to_keep.difference(scenes_to_delete)
            scenes_to_delete = None

        # Create a new scene table
        new_scenes = list()
        for scene in scenes_table:
            if scenes_to_keep is not None and scene["name"] in scenes_to_keep: 
                new_scenes.append(scene)
            elif scenes_to_delete is not None and scene["name"] not in scenes_to_delete: 
                new_scenes.append(scene)
    else:
        new_scenes = scenes_table
    ##############################
    # Create a new sample table
    scene_tokens = set()
    for scene in new_scenes:
        # toke2scene[scene['token']]=scene['name']
        scene_tokens.add(scene['token'])
    sample_table = load_table(tables_dir_src, 'sample')
    new_sample = list()
    for sample in sample_table:
        if sample["scene_token"] in scene_tokens: 
            new_sample.append(sample)

    ##############################
    # Create a new sample_data table
    sample_data_table = load_table(tables_dir_src, 'sample_data')
    sample_tokens = set()
    for sample in new_sample:
        sample_tokens.add(sample['token'])
    new_sample_data = list()
    for sample_data in sample_data_table:
        if sample_data["sample_token"] in sample_tokens: 
            new_sample_data.append(sample_data)
    ##############################
    # Create a new sample_annotation table
    sample_annotation_table = load_table(tables_dir_src, 'sample_annotation') 
    new_sample_annotation = list()
    for sample_annot in sample_annotation_table:
        if sample_annot["sample_token"] in sample_tokens: 
            new_sample_annotation.append(sample_annot)

    ##############################
    # Write new scene table
    write_table(new_scenes, tables_dir_dest, 'scene')
    # Write new sample table
    write_table(new_sample, tables_dir_dest, 'sample')
    # Write new sample data table
    write_table(new_sample_data, tables_dir_dest, 'sample_data')
    # Write new sample annotation table
    write_table(new_sample_annotation, tables_dir_dest, 'sample_annotation')

def select_scenes_from_desc(tables_dir_src:str, selector):
    scenes_table = load_table(tables_dir_src, 'scene')
    process_descr = lambda s: set([s.strip() for s in s.lower().split(',')])
    included_scenes = dict()
    for scene in scenes_table:
        keys = process_descr(scene['description'])
        selected = selector(keys)
        if selected is not None:
            if included_scenes.get(selected) is None:
                included_scenes[selected]=list()
            included_scenes[selected].append(scene['name'])
    return included_scenes

def assign_scenes_to_desc_classes(tables_dir_src:str, assign_fn):
    scenes_table = load_table(tables_dir_src, 'scene')
    process_descr = lambda s: set([s.strip() for s in s.lower().split(',')])
    # keys_to_scenes = dict()
    scenes_to_key = dict()
    for scene in scenes_table:
        desc_keys = process_descr(scene['description'])
        key = assign_fn(desc_keys)
        # if keys_to_scenes.get(key) is None:
        #    keys_to_scenes[key]=list()
        # keys_to_scenes[key].append(scene['name'])
        scenes_to_key[scene['name']] = key
    return scenes_to_key

# Set the path to the dataset
class SplitStrategies(Enum):
    RandSampling = 'RandomSampling'
    RandStratClusterSampling = 'RandomStratifiedSamplingBasedOnClusters'

def create_split_dataset(dataset_path, save_dir,tables_dir_src, dest_ann_fname='annotations.json', split_strategy=SplitStrategies.RandSampling):
    annotations = load_annotations(dataset_path)
    train_scenes = None
    val_scenes = None
    if split_strategy == SplitStrategies.RandSampling:
        train_scenes = random_sampling_scenes(annotations['train_split'] , sample_ratio=0.1)[:1]
        val_scenes = random_sampling_scenes(annotations['val_split'], sample_ratio=0.1)[:1]
    elif split_strategy == SplitStrategies.RandStratClusterSampling:
        pass

    create_new_tables(tables_dir_src, save_dir, scenes_to_keep=set(train_scenes).union(set(val_scenes)), scenes_to_delete=None)
    annotations = remove_scenes(annotations, set(train_scenes).union(set(val_scenes)), del_included=False)
    annotations['train_split'] = train_scenes
    annotations['val_split'] = val_scenes
    write_annotations(annotations, save_dir, dest_ann_fname)

def retrieve_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)    

def write_pickle(obj, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def rainy_night_det(desc_keys):
    desc_keys = set(desc_keys)
    if 'night' in desc_keys:
        if 'rain' in desc_keys:
            return 'r-n'
        else:
            return 'n'
    elif 'rain' in desc_keys:
        return 'r-d'
    else:
        return 'd'

def extend_labels(labels, scenes, extra_labels_dict):
    new_labels = list()
    for label, scene in zip(labels, scenes):
        if scene in extra_labels_dict:
            new_labels.append('{}-{}'.format(extra_labels_dict[scene], label))
        else:
            new_labels.append(label)
    return new_labels

def labels_to_int(labels):
    unique_labels = list(set(labels))
    label_to_int = dict()
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
    int_labels = list()
    for label in labels:
        int_labels.append(label_to_int[label])
    return int_labels, unique_labels

def generate_clusters_experiments(annotations, dataset_path, tables_dir_src,
                                  n_clusters_l = [2, 4, 8, 16, 32],
                                  metrics_l = [phi, chi_square_norm],
                                  linkage_l = ['average', 'complete', 'single'],
                                  save_dir='./temp/',
                                  dest_ann_fname='annotations.json',
                                  save_pickle=True,
                                  load_pickle=False):
    os.makedirs(save_dir, exist_ok=True)
    if load_pickle:
        filepath = os.path.join(save_dir, 'class_counts.pickle')
        if os.path.isfile(filepath):
            (class_count_train, class_count_val) = retrieve_pickle(filepath)
    else:
        class_count_train = count_scene_classes(annotations, dataset_path, data_split='train_split', 
                                                        gt_dir_name='gts', 
                                                        labels_fname='labels.npz',
                                                        deleted_classes=[17],
                                                        min_classes=18)
        class_count_val = count_scene_classes(annotations, dataset_path, data_split='val_split', 
                                                        gt_dir_name='gts', 
                                                        labels_fname='labels.npz',
                                                        deleted_classes=[17],
                                                        min_classes=18) 
    class_count = {**class_count_train, **class_count_val}
    scene_to_ind, scene_names, scene_fts_l = dict_to_array(class_count)
    # Save a pickle
    if save_pickle and not load_pickle:
        filepath = os.path.join(save_dir, 'class_counts.pickle')
        write_pickle((class_count_train, class_count_val), filepath)
             
    for linkage in linkage_l:
        for metric in metrics_l:
            for n_clusters in n_clusters_l:
                id_str = '{}_{}_{}'.format(linkage, metric.__name__, n_clusters)
                logging.info('Executing experiment {}'.format(id_str))
                copy_annot = copy.deepcopy(annotations)
                labels = cluster_data(scene_fts_l, n_clusters=n_clusters, metric=metric, linkage=linkage, distance_threshold=None)

                train_classes = to_class_scenes(labels, scene_names, scenes_included=class_count_train)
                val_classes = to_class_scenes(labels, scene_names, scenes_included=class_count_val)

                train_scenes = stratified_sampling_scenes(train_classes, sample_ratio=0.1, random_state=None)
                val_scenes = stratified_sampling_scenes(val_classes, sample_ratio=0.1, random_state=None)
                # Save figures of clusters
                exp_dir = os.path.join(save_dir, id_str)
                os.makedirs(exp_dir, exist_ok=True)
                # All the data's clusters (from both validation and train data)
                logging.info('{}: Saving Clusters Figure for the Full Data (Both Train and Validation).'.format(id_str))
                visualize_clusters(scene_fts_l, labels, fig_name=os.path.join(exp_dir, '{}_clusters_full.pdf'.format(id_str)))
                visualize_clusters(scene_fts_l, labels, fig_name=os.path.join(exp_dir, '{}_clusters_full.png'.format(id_str)))
                # All the train data's clusters
                logging.info('{}: Saving Clusters Figure for the Full Train Data.'.format(id_str))
                train_fts_all, train_labels_all = get_selected_scene_info(class_count_train.keys(), scene_to_ind, scene_fts_l, labels)
                visualize_clusters(train_fts_all, train_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_train.pdf'.format(id_str)))
                visualize_clusters(train_fts_all, train_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_train.png'.format(id_str)))
                # All the val data's clusters
                logging.info('{}: Saving Clusters Figure for the Full Validation Data.'.format(id_str))
                val_fts_all, val_labels_all = get_selected_scene_info(class_count_val.keys(), scene_to_ind, scene_fts_l, labels)
                visualize_clusters(val_fts_all, val_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_val.pdf'.format(id_str)))
                visualize_clusters(val_fts_all, val_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_val.png'.format(id_str)))
                # The sampled train data's clusters
                logging.info('{}: Saving Clusters Figure for the Sample Train Data.'.format(id_str))
                train_fts, train_labels = get_selected_scene_info(train_scenes, scene_to_ind, scene_fts_l, labels)
                visualize_clusters(train_fts, train_labels, fig_name=os.path.join(exp_dir, '{}_clusters_train_ss.pdf'.format(id_str)))
                visualize_clusters(train_fts, train_labels, fig_name=os.path.join(exp_dir, '{}_clusters_train_ss.png'.format(id_str)))
                # The sampled val data's clusters
                logging.info('{}: Saving Clusters Figure for the Sample Validation Data.'.format(id_str))
                val_fts, val_labels = get_selected_scene_info(val_scenes, scene_to_ind, scene_fts_l, labels)
                visualize_clusters(val_fts, val_labels, fig_name=os.path.join(exp_dir, '{}_clusters_val_ss.pdf'.format(id_str)))
                visualize_clusters(val_fts, val_labels, fig_name=os.path.join(exp_dir, '{}_clusters_val_ss.png'.format(id_str)))
                write_pickle((scene_fts_l, labels, list(class_count.keys()), list(class_count_train.keys()), list(class_count_val.keys()), train_scenes, val_scenes), filepath)
                # Save annotations file
                logging.info('{}: Wrote new annotations file.'.format(id_str))
                copy_annot = remove_scenes(copy_annot, set(train_scenes).union(set(val_scenes)), del_included=False)
                copy_annot['train_split'] = train_scenes
                copy_annot['val_split'] = val_scenes
                create_new_tables(tables_dir_src, exp_dir, scenes_to_keep=set(train_scenes).union(set(val_scenes)), scenes_to_delete=None)
                write_annotations(copy_annot, exp_dir, dest_ann_fname)

def generate_clusters_experiments_2(annotations, dataset_path, tables_dir_src,
                                  n_clusters_l = [2, 4, 8, 16, 32],
                                  metrics_l = [phi, chi_square_norm],
                                  linkage_l = ['average', 'complete', 'single'],
                                  save_dir='./temp/',
                                  dest_ann_fname='annotations.json',
                                  save_pickle=True,
                                  scene_class_desc_assigner=rainy_night_det,
                                  load_pickle=False):
    os.makedirs(save_dir, exist_ok=True)
    if load_pickle:
        filepath = os.path.join(save_dir, 'class_counts.pickle')
        if os.path.isfile(filepath):
            (class_count_train, class_count_val) = retrieve_pickle(filepath)
    else:
        class_count_train = count_scene_classes(annotations, dataset_path, data_split='train_split', 
                                                        gt_dir_name='gts', 
                                                        labels_fname='labels.npz',
                                                        deleted_classes=[17],
                                                        min_classes=18)
        class_count_val = count_scene_classes(annotations, dataset_path, data_split='val_split', 
                                                        gt_dir_name='gts', 
                                                        labels_fname='labels.npz',
                                                        deleted_classes=[17],
                                                        min_classes=18) 
    class_count = {**class_count_train, **class_count_val}
    scene_to_ind, scene_names, scene_fts_l = dict_to_array(class_count)
    # Save a pickle
    if save_pickle and not load_pickle:
        filepath = os.path.join(save_dir, 'class_counts.pickle')
        write_pickle((class_count_train, class_count_val), filepath)
             
    for linkage in linkage_l:
        for metric in metrics_l:
            for n_clusters in n_clusters_l:
                id_str = '{}_{}_{}'.format(linkage, metric.__name__, n_clusters)
                logging.info('{}:Executing experiment'.format(id_str))
                copy_annot = copy.deepcopy(annotations)
                labels = cluster_data(scene_fts_l, n_clusters=n_clusters, metric=metric, linkage=linkage, distance_threshold=None)
                logging.info('{}:Assign description labels to scenes'.format(id_str))
                scenes_to_label = assign_scenes_to_desc_classes(tables_dir_src, scene_class_desc_assigner)
                labels = extend_labels(labels, scene_names, scenes_to_label)
                logging.info('{}:Convert list of labels to dict of labels to scenes'.format(id_str))
                train_classes = to_class_scenes(labels, scene_names, scenes_included=class_count_train)
                val_classes = to_class_scenes(labels, scene_names, scenes_included=class_count_val)
                logging.info('{}:Initiating Stratified Sampling'.format(id_str))
                train_scenes = stratified_sampling_scenes(train_classes, sample_ratio=0.1, random_state=None)
                val_scenes = stratified_sampling_scenes(val_classes, sample_ratio=0.1, random_state=None)

                # Save figures of clusters
                exp_dir = os.path.join(save_dir, id_str)
                os.makedirs(exp_dir, exist_ok=True)
                # All the data's clusters (from both validation and train data)
                logging.info('{}: Saving Clusters Figure for the Full Data (Both Train and Validation).'.format(id_str))
                visualize_clusters(scene_fts_l, labels, fig_name=os.path.join(exp_dir, '{}_clusters_full.pdf'.format(id_str)))
                visualize_clusters(scene_fts_l, labels, fig_name=os.path.join(exp_dir, '{}_clusters_full.png'.format(id_str)))
                # All the train data's clusters
                logging.info('{}: Saving Clusters Figure for the Full Train Data.'.format(id_str))
                train_fts_all, train_labels_all = get_selected_scene_info(class_count_train.keys(), scene_to_ind, scene_fts_l, labels)
                visualize_clusters(train_fts_all, train_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_train.pdf'.format(id_str)))
                visualize_clusters(train_fts_all, train_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_train.png'.format(id_str)))
                # All the val data's clusters
                logging.info('{}: Saving Clusters Figure for the Full Validation Data.'.format(id_str))
                val_fts_all, val_labels_all = get_selected_scene_info(class_count_val.keys(), scene_to_ind, scene_fts_l, labels)
                visualize_clusters(val_fts_all, val_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_val.pdf'.format(id_str)))
                visualize_clusters(val_fts_all, val_labels_all, fig_name=os.path.join(exp_dir, '{}_clusters_val.png'.format(id_str)))
                # The sampled train data's clusters
                logging.info('{}: Saving Clusters Figure for the Sample Train Data.'.format(id_str))
                train_fts, train_labels = get_selected_scene_info(train_scenes, scene_to_ind, scene_fts_l, labels)
                visualize_clusters(train_fts, train_labels, fig_name=os.path.join(exp_dir, '{}_clusters_train_ss.pdf'.format(id_str)))
                visualize_clusters(train_fts, train_labels, fig_name=os.path.join(exp_dir, '{}_clusters_train_ss.png'.format(id_str)))
                # The sampled val data's clusters
                logging.info('{}: Saving Clusters Figure for the Sample Validation Data.'.format(id_str))
                val_fts, val_labels = get_selected_scene_info(val_scenes, scene_to_ind, scene_fts_l, labels)
                visualize_clusters(val_fts, val_labels, fig_name=os.path.join(exp_dir, '{}_clusters_val_ss.pdf'.format(id_str)))
                visualize_clusters(val_fts, val_labels, fig_name=os.path.join(exp_dir, '{}_clusters_val_ss.png'.format(id_str)))
                write_pickle((scene_fts_l, labels, list(class_count.keys()), list(class_count_train.keys()), list(class_count_val.keys()), train_scenes, val_scenes), filepath)
                # Save annotations file
                logging.info('{}: Wrote new annotations file.'.format(id_str))
                copy_annot = remove_scenes(copy_annot, set(train_scenes).union(set(val_scenes)), del_included=False)
                copy_annot['train_split'] = train_scenes
                copy_annot['val_split'] = val_scenes
                create_new_tables(tables_dir_src, exp_dir, scenes_to_keep=set(train_scenes).union(set(val_scenes)), scenes_to_delete=None)
                write_annotations(copy_annot, exp_dir, dest_ann_fname)

def main():
    dataset_path='../occ3d-nus'
    tables_dir_src = '../occ3d-nus/v1.0-trainval'
    annotations = load_annotations(dataset_path)
    save_dir='./exper_data_splits_special'
    # create_split_dataset(dataset_path=dataset_path, save_dir=save_dir, tables_dir_src=tables_dir_src, dest_ann_fname='annotations.json', split_strategy=SplitStrategies.RandSampling)
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, 'annotations.pickle'), 'wb') as handle:
    #     pickle.dump(annotations , handle, protocol=pickle.HIGHEST_PROTOCOL)
    generate_clusters_experiments_2(annotations, dataset_path, tables_dir_src,
                                #   n_clusters_l = [2, 4, 8, 16, 17, 32],
                                n_clusters_l = [17],
                                #   metrics_l = [phi, chi_square_norm, chi_norm],
                                metrics_l = [phi],
                                #   linkage_l = ['average', 'complete', 'single'],
                                  linkage_l = ['average'],
                                  save_dir=save_dir,
                                  dest_ann_fname='annotations.json',
                                  load_pickle = False)
    
if __name__ == '__main__': 
    main()