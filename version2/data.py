from parameters import *
import glob
from skimage import io
from skimage import transform
from sklearn.model_selection import train_test_split
import numpy as np
import os

def get_class(img_path):
    return img_path.split(os.sep)[-2]


def preprocess_img(img,IMG_SIZE):
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    img = img.reshape(img.shape + (1,))

    return img


def __sort_list__(list1,list2):
    zipped_pairs = zip(list2,list1)
    sorted_list = [x for _, x in sorted(zipped_pairs)]
    return sorted_list


def prepare_images():
    print("preparation of the images...")
    all_img_paths = glob.glob(os.path.join(root_dir,'*'+os.sep+'*.jpg'))
    images_list=[]
    labels_list=[]
    for img_path in all_img_paths:

        images_list.append(preprocess_img(io.imread(img_path),img_size)) #images resized
        labels_list.append(get_class(img_path)) #labels

    d = {ni: indi for indi, ni in enumerate(set(labels_list))}  # assign a number to each unique element in the list "labels", stored in d
    numbers_label = [d[ni] for ni in labels_list]  # list comprehension and store the actual numbers in the numbers_label

    index = np.random.permutation(len(images_list))

    all_images = __sort_list__(images_list,index)
    all_label_names = __sort_list__(labels_list,index)
    all_label_numbers = __sort_list__(numbers_label,index)


    return all_images, all_label_names, all_label_numbers



def divide_images_and_labels(images, labels):

    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2) #images :all_images     labels: all_label_numbers
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



#run the process
#all_images, all_label_names, all_label_numbers= prepare_images()



# save(this is not mandatory)
# np.save(COMMON_PATH + "/all_images", all_images)
# np.save(COMMON_PATH + "/all_label_names", all_label_names)
# np.save(COMMON_PATH + "/all_label_numbers", all_label_numbers)

#X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(all_images,dtype='float32'), np.eye(121, dtype='uint8')[all_label_numbers])

# save(this is not mandatory)
# np.save(COMMON_PATH +"/X_test", X_test)
# np.save(COMMON_PATH +"/Y_test", Y_test)
