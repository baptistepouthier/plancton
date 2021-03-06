import cv2
import numpy as np
import glob
import os


print ('OpenCV VERSION (should be 3.1.0 or later, with nonfree modules installed!):', cv2.__version__)

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Unable to open '%s'. Are you sure it's a valid image path?")
    return img


def neg_img_cal101(positive_folder, cal101_root='101_ObjectCategories', image_suffix='*.jpg'):
    """Simply return list of paths for all images in cal101 dataset, except those in positive_folder."""
    return [path for path in glob.glob(cal101_root + '/*/' + image_suffix) if positive_folder not in path]


def binary_labeled_img_from_cal101(positive_folder, cal101_root='101_ObjectCategories', image_suffix='*.jpg'):
    """
    Generate a balanced dataset of positive and negative images from a directory of images
    where each type of image is separated in its own folder.
    Returns:
    --------
    labeled_img_paths: list of lists
        Of the form [[image_path, label], ...]
        Where label is True or False for positive and negative images respectively
    """
    all_imgs = set(glob.glob(cal101_root + '/*/' + image_suffix))
    pos_imgs = set(glob.glob(os.path.join(cal101_root, positive_folder) + '/' + image_suffix))
    neg_imgs = all_imgs - pos_imgs

    neg_sample_size = len(pos_imgs)
    selected_negs = np.random.choice(list(neg_imgs), size=neg_sample_size, replace=False)

    print (str(len(pos_imgs))+' positive,'+ str(len(selected_negs)) +'negative images selected (out of '+ str(len(neg_imgs)) +' negatives total)')

    labeled_img_paths = [[path, True] for path in pos_imgs] + [[path, False] for path in selected_negs]

    return np.array(labeled_img_paths)
#
#
def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    """
    Get indexes for training, test, and validation rows, given a total number of rows.
    Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]
    Returns:
    --------
    training_idxs, test_idxs, val_idxs
        Both lists of integers
    """
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    val_idxs = np.random.choice(row_range, size=no_val_rows, replace=False)
    # remove validation indexes
    training_idxs = [idx for idx in row_range if idx not in val_idxs]

    print ('Train-test-val split: '+str(len(training_idxs))+' training rows, '+str(len(test_idxs))+' test rows, '+str(len(val_idxs))+' validation rows')

    return training_idxs, test_idxs, val_idxs


def gen_sift_features(labeled_img_paths):
    """
    Generate SIFT features for images
    Parameters:
    -----------
    labeled_img_paths : list of lists
        Of the form [[image_path, label], ...]
    Returns:
    --------
    img_descs : list of SIFT descriptors with same indicies as labeled_img_paths
    y : list of corresponding labels
    """
    # img_keypoints = {}
    img_descs = []

    print ('generating SIFT descriptors for '+str(len(labeled_img_paths))+' images')

    for img_path, _ in labeled_img_paths:
        img = read_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # kp, desc = sift.detectAndCompute(gray, None)
        # img_keypoints[img_path] = kp
        keypoints = []
        step = 10

        for x in range(round((img.shape[1]%step)/2),img.shape[1],step):
            for y in range(round((img.shape[0]%step)/2),img.shape[0],step):
                keypoints.append(cv2.KeyPoint(x,y,1))

        kp, desc = sift.compute(gray,keypoints)

        img_descs.append(desc)

    print ('SIFT descriptors generated.')

    y = np.array(labeled_img_paths)[:,1]

    d = {ni: indi for indi, ni in enumerate(set(y))}  # assign a number to each unique element in the list "labels", stored in d
    label_numbers = [d[ni] for ni in y]  # list comprehension and store the actual numbers in the numbers_label

    return img_descs, y, label_numbers


def cluster_features(img_descs, training_idxs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    # img_descs = gen_sift_features(labeled_img_paths)
    #
    # # Generate indexes of training rows
    # total_rows = len(img_descs)
    # training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, percent_test, percent_val)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]

    # print(any([elm is None for elm in training_descs]))

    # a = [desc_list for desc_list in training_descs]
    # all_train_descriptors = [desc for desc in a]

    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    # all_train_descriptors=[]
    # for desc_list in training_descs:
    #     # print(desc_list)
    #     for desc in desc_list:
    #         all_train_descriptors.append(desc)

    # print(all_train_descriptors)

    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print (str(all_train_descriptors.shape[0])+' descriptors before clustering')

    # Cluster descriptors to get codebook
    print ('Using clustering model ' + str(repr(cluster_model))+ '...')
    print ('Clustering on training set to get codebook of '+str(n_clusters)+ ' words')

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print ('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('done generating BoW histograms.')

    return X, cluster_model

def perform_data_split(X, y, training_idxs, test_idxs, val_idxs):
    """
    Split X and y into train/test/val sets
    Parameters:
    -----------
    X : eg, use img_bow_hist
    y : corresponding labels for X
    training_idxs : list/array of integers used as indicies for training rows
    test_idxs : same
    val_idxs : same
    Returns:
    --------
    X_train, X_test, X_val, y_train, y_test, y_val
    """
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    X_val = X[val_idxs]

    y_train = y[training_idxs]
    y_test = y[test_idxs]
    y_val = y[val_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val


def img_to_vect(img_path, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """

    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)


def __sort_list__(list1,list2):
    zipped_pairs = zip(list2,list1)
    sorted_list = [x for _, x in sorted(zipped_pairs)]
    return sorted_list


def get_class(img_path):
    return img_path.split(os.sep)[-2]

# labeled_image_paths=[]
# for path in all_img_path: labeled_image_paths.append([path,get_class(path)])
#
# img_descs, y, label_numbers = gen_sift_features(labeled_image_paths)




