import numpy as np
import time
import skimage.feature as skfeat
import cv2
from holidays_dataset_handler import HolidaysDatasetHandler
from typing import Any, Callable
from itertools import product

def compute_1d_color_hist(img, bins_per_hist = 32, norm_p=2):
    """
    Compute a 1d color histogram of the image.
  
    - img: Color image (Numpy array)
    - bins_per_hist: Number of bins per histogram

    RETURN: 
    - A numpy array of shape (bins_per_hist * 3,)
    """
    
    histograms = np.zeros((3, bins_per_hist))

    for i in range(3):

        h, _ = np.histogram(img[:,:,i], bins = bins_per_hist, range = (0,255))
        histograms[i, :] = h/np.linalg.norm(h, norm_p)
        
    return histograms.flatten()


def compute_2d_color_hist(img, bins_per_hist = 16):
    """
    Compute a 2d color histogram of the image.
    
    The final descriptor will be the concatenation of 3 normalized 2D histograms: B/G, B/R and G/R.
  
    - img: Color image (Numpy array)
    - bins_per_hist: Number of bins per histogram

    RETURN:
    - A numpy array of shape (bins_per_hist * bins_per_hist * 3,)
    """
    blue, green, red = cv2.split(img)
    histograms = np.zeros((3, bins_per_hist * bins_per_hist))

    h = cv2.calcHist([blue, green],[0,1],None, [bins_per_hist, bins_per_hist], ranges = [0,255,0,255]) # B/G
    histograms[0,:] = h.flatten()/np.linalg.norm(h.flatten())

    h = cv2.calcHist([blue, red],[0,1],None, [bins_per_hist, bins_per_hist], ranges = [0,255,0,255]) # B/R
    histograms[1,:] = h.flatten()/np.linalg.norm(h.flatten())

    h = cv2.calcHist([green, red],[0,1],None, [bins_per_hist, bins_per_hist], ranges = [0,255,0,255]) # G/R
    histograms[2,:] = h.flatten()/np.linalg.norm(h.flatten())


    # YOUR CODE HERE
    return histograms.flatten()
    # -----


def compute_lbp_descriptor(img, p = 8, r = 1):
    """
    Compute a rotation invariant and uniform LBP histogram as image descriptor.
  
    - img: Input image (Numpy array)
    - p: Neighbors to check in radius r
    - r: Radius in pixels

    RETURN: 
    - A numpy array of shape (p + 2,)
    """    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = skfeat.local_binary_pattern(gray_image,p,r,'uniform')
    h,_ = np.histogram(lbp.ravel(), bins = p + 2, range = (0,p+2), density=True)
    return h



class CBIR:
    """
    Class to encapsulate the basic functionalities of a CBIR system.
    """
    
    def __init__(self, desc_func, **params):
        """
        Class constructor.
        
        - desc_func: The function to be used for describing the images
        - params: A variable number of parameters required to call desc_func.
            This is a dictionary that can be unpacked within the function.
            See more info here: https://realpython.com/python-kwargs-and-args/
        """
        self.desc_func = desc_func
        self.params = params
        self.image_descriptors = {}
        
    def build_image_db(self, images: dict):
        """
        Create the CBIR system database.
        
        - images: A dictionary of images (Numpy arrays)
        
        This function should describe each image using desc_func and save the 
        resulting descriptors in a dictionary of Numpy arrays (global descriptors) called image_descriptors.
        The names should be the same than the ones in images.
        """        

        for name, img in images.items():
            self.image_descriptors[name] = self.desc_func(img, **self.params)
            
    def search_image(self, query_descriptor):
        """
        Search an image in the system.
        
        - query_descriptor: Global descriptor of the query image (NumPy array)
        
        RETURNS:
        - An sorted list of tuples, each one with the format (database image name, L2 distance)
        
        This method is responsible for searching for the most similar images in the database 
        based on a query descriptor. It compares the query descriptor with all the descriptors in 
        the image database using the L2 (Euclidean) distance and returns a sorted list of results.
        """

        # List to store the results (image name and L2 distance)
        results = [("",0.0)]*len(self.image_descriptors)
        
        for idx, (name, descriptor) in enumerate(self.image_descriptors.items()):
            dist = np.sqrt(np.sum((query_descriptor - descriptor)**2))
            results[idx] = (name, dist)

        ord_res = sorted(results, key=lambda pair: pair[1])
        return ord_res


def extract_interest_points(img, feat_type = 'SIFT', nfeats = 500, thresh = 50):
    """
    Compute keypoints and their corresponding descriptors from an image.
  
    Parameters:
    - img: Input image (Numpy array).
    - feat_type: Detection / description method ('SIFT', 'FAST_BRIEF', 'ORB').
    - nfeats: Maximum number of features. It can be directly used to configure SIFT and ORB.
    - thresh: Detection threshold. Useful for FAST and ORB.
  
    Returns:
    - kp: A list of detected keypoints (cv2.KeyPoint).
    - des: A numpy array of shape (number_of_kps, descriptor_size) of type:
        - 'np.float32' for SIFT.
        - 'np.uint8' for BRIEF and ORB.
    """
    kp = []
    des = []

    if feat_type == 'SIFT':
        extractor = cv2.SIFT_create(nfeatures=nfeats)
        kp, des = extractor.detectAndCompute(img, None)

    elif feat_type == 'ORB':
        extractor = cv2.ORB_create(nfeatures=nfeats,fastThreshold=thresh)
        kp = extractor.detect(img,None)
        kp, des = extractor.compute(img, kp)
    elif feat_type == 'FAST_BRIEF':
        fast = cv2.FastFeatureDetector_create(threshold=thresh)
        extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp = fast.detect(img,None)
        kp, des = extractor.compute(img,kp)
    return kp, des
    
    
def find_matches(query_desc, database_desc, k=2):
    """
    Match two sets of descriptors. For each query descriptor, this method searches
    for the k closest descriptors in the database set.
  
    Parameters:
    - query_desc (np.ndarray): A NumPy array of shape (num_query_kps, descriptor_size),
      containing descriptors from the query image.
    - database_desc (np.ndarray): A NumPy array of shape (num_database_kps, descriptor_size),
      containing descriptors from the database image.
    - k (int): Number of nearest neighbors to retrieve for each descriptor (default: 2).
  
    Returns:
    - matches (list of list of cv2.DMatch): A list where each element contains k matches,
      sorted by distance.
    """
    matcher = cv2.FlannBasedMatcher() if query_desc[0].dtype=='float32' else cv2.BFMatcher_create()

    return matcher.knnMatch(query_desc, database_desc, k)

    
def filter_matches(matches, ratio=0.75):
    """
    Filters matches using the Nearest Neighbor Distance Ratio (NNDR) criterion.

    Parameters:
    - matches (list of list of cv2.DMatch): A list where each element contains k matches,
      sorted by distance (output from 'find_matches').
    - ratio (float): The threshold for the ratio test. A match is kept if the 
      distance of the best match is less than `ratio *` the distance of the second-best match.
    
    Returns:
    - filtered_matches (list of cv2.DMatch): A list of matches that passed the ratio test.
    """
    return [m for m,n in matches if m.distance < n.distance*ratio]


def evaluate(dataset, method='SIFT', nfeats=3000, thresh=25, ratio=0.75):
    """
    Evaluate the image retrieval performance using local features (SIFT, ORB, etc.).
    This function computes the mean Average Precision (mAP) for the dataset.
    
    Args:
        dataset (HolidaysDatasetHandler): The dataset handler object.
        method (str): The feature extraction method to use ('SIFT', 'SURF', 'ORB').
        nfeats (int): Maximum number of features to extract.
        thresh (int): Threshold for feature detection.
        ratio (float): Nearest Neighbor Distance Ratio for filtering matches.
    
    Returns:
        float: The mean Average Precision (mAP) score for the retrieval system.
    """

    params = {'nfeats':nfeats, 'thresh':thresh, 'ratio':ratio}
    retriever = LocalCBIR(method, **params)
    
    return retriever.compute_mAP(dataset)
    
#########################################################################
#########################################################################
######## CUSTOM CODE (NOT ORIGINALLY INCLUDED IN THE ASSIGNMENT) ########
#########################################################################
#########################################################################
class DatasetHandler(HolidaysDatasetHandler):
    def load_query_images(self):
        """
        Retrieve a dictionary with the query images.

        Returns:
 
            dict: A dict with the query image names as keys and np.arrays storing the corresponding images as values

        Custom method.
        """
        return {name: self.get_image(name) for name in self.get_query_images()}

    def load_database_images(self):
        """
        Retrieve a dictionary with the database images.

        Returns:

            dict: A dict with the database image names as keys and np.arrays storing the corresponding images as values

        Custom method.
        """
        return {name: self.get_image(name) for name in self.get_database_images()}

from abc import ABC, abstractmethod
class AbstractCBIR(ABC):
    def __init__(self, ratio:float=0.75, **kwargs):
        self.image_descriptors = {}
        self.ratio=ratio
        self.params = kwargs

    @abstractmethod
    def compute_descriptor(self, img: np.array)-> np.array:
        """
        Child classes should implement this method. Takes an image and computes its descriptor (be it global or local).
        """
        ...
    
    @abstractmethod
    def compare_descriptors(self, query_descriptor: np.array, key_descriptor: np.array)->float:
        """
        Child classes should implement this method. Takes two descriptors and computes a dissimilarity measure between them.
        This means that when sorting increasingly (default in Python), most similar descriptors should appear first.
        """
        ...

    def build_image_db(self, images: dict):
        """
        Create the CBIR system database.
        
        - images: A dictionary of images (Numpy arrays)
        
        This function should describe each image using desc_func and save the 
        resulting descriptors in a dictionary of Numpy arrays (descriptors) called image_descriptors.
        The names should be the same as the ones in images.
        """        

        for name, img in images.items():
            self.image_descriptors[name] = self.compute_descriptor(img)
            
    def search_image(self, query_descriptor):
        """
        Search an image in the system.
        
        - query_descriptor: Descriptor of the query image (NumPy array)
        
        RETURNS:
        - An sorted list of tuples, each one with the format (database image name, dissimilarity)
        
        This method is responsible for searching for the most similar images in the database 
        based on a query descriptor. It compares the query descriptor with all the descriptors in 
        the image database and returns a sorted list of results.
        """

        # List to store the results (image name and L2 distance)
        results = [("",0.0)]*len(self.image_descriptors)
        
        for idx, (name, key_descriptor) in enumerate(self.image_descriptors.items()):
            results[idx] = (name, self.compare_descriptors(query_descriptor, key_descriptor))

        return sorted(results, key=lambda pair: pair[1])

    def search_image_from_value(self, query_image):
        """
        Search an image in the system.

        - query_image: The image to be searched (NumPy array)

        RETURNS:

        - A sorted list of tuples, each one with the format (database image name, dissimilarity)
        """
        query_descriptor = self.compute_descriptor(query_image)
        return self.search_image(query_descriptor)


    def rank_database_from_image(self, query_image):
        """
        Search an image in the system and return a sorted list of matches.

        - query_image: The image to be searched (NumPy array)

        RETURNS:

        - A list of the names of the images in the database, sorted from most to least similar to the query image.

        Custom method.
        """
        return [name for name, _ in self.search_image_from_value(query_image)]

    def compute_mAP(self, dataset_handler: DatasetHandler):
        """
        Load a database into the instance and compute the mAP for that database.

        - dataset_handler: The handler for the database to be used. It should have `load_database_images` and
          `load_query_images` methods, each returning a dictionary with the image names as keys and the actual
          images as values, regarding the training and test sets, respectively. 
 
        RETURNS:

        - The mean average precision of the current instance for the dataset provided.

        Custom method.
        """
        images = dataset_handler.load_database_images()
        if not self.image_descriptors:
            self.build_image_db(images)

        query_images = dataset_handler.load_query_images()

        ranked_dict = {
            name: self.rank_database_from_image(img)
            for name, img in query_images.items()
        }
 
        return dataset_handler.compute_mAP(ranked_dict)
    
class GlobalCBIR(AbstractCBIR):
    def __init__(self, desc_func, **kwargs):
        """
        Class constructor.
        
        - desc_func: The function to be used for describing the images
        - params: A variable number of parameters required to call desc_func.
            This is a dictionary that can be unpacked within the function.
            See more info here: https://realpython.com/python-kwargs-and-args/
        """
        super().__init__(**kwargs)
        self.desc_func = desc_func
        
    def compute_descriptor(self, img):
        return self.desc_func(img, **self.params)

    def compare_descriptors(self, query_descriptor, key_descriptor):
        """
            Computes the p-norm among the global descriptors. Specify p as a parameter (default is 2)
        """
        p: float = self.params.get('norm_p') or 2
        
        diff = np.abs(query_descriptor - key_descriptor)
        
        return np.pow(np.sum(
            np.pow(diff, p)
        ), 1/p)
        
class LocalCBIR(AbstractCBIR):
    def __init__(self, feat_type, **kwargs):
        super().__init__(**kwargs)
        self.feat_type = feat_type
        
    def compute_descriptor(self, img):
        _, des = extract_interest_points(img, feat_type=self.feat_type, **self.params)
        return des
    
    def compare_descriptors(self, query_descriptor, key_descriptor):
        if query_descriptor is None:
            query_descriptor = []
        if key_descriptor is None:
            key_descriptor = []

        if len(query_descriptor) + len(key_descriptor) == 0:
            return 0

        num_matches = len(filter_matches(find_matches(query_descriptor, key_descriptor), ratio=self.ratio))
        
        return -num_matches # Sign inversion to sort in the right order

def all_dict_combinations(d: dict[str, list])->list[dict]:
    """
    "Unrolls" the given dictionary of lists into a list of dictionary with all possible combinations.
    For example:
        Input: {"foo":[0,1], "bar":[2,3]}
        Output: [{"foo":0, "bar":2}, {"foo":0, "bar":3}, {"foo":1, "bar":2}, {"foo":1, "bar":3}]
    
    Custom method.
    """
    prod = product(*d.values())
    keys = d.keys()
    
    return [dict(zip(keys, combination)) for combination in prod]

def evaluate_in_all_combinations(function: Callable[Any, float], parameter_lists:dict[str, list])-> list[tuple[dict[str, Any], float]]:
    """
    Evaluates the function in all possible combinations of parameters and returns a list with the values and parameter combinations.
    """
    
    values = [(params, function(**params)) for params in all_dict_combinations(parameter_lists)]
    values.sort(key=lambda tup: tup[1], reverse=True)

    return values

def benchmark_ms(f: Callable, *args, **kwargs):
    """
    Executes a given function and measures the time it takes.
    """
    previous_time = time.time()  # In seconds
    value = f(*args, **kwargs)
    elapsed = time.time() - previous_time

    return value, elapsed * 1000

def evaluate_and_benchmark_in_all_combinations (function: Callable[Any, float], parameter_lists: dict[str, list]) -> list[tuple[dict[str, Any], float, float]]:
    """
    Args:
        function: A callable function that computes yields the mAP of a CBIR system given its parameters.
        parameter_lists: All possible combinations of parameters to be benchmarked.

    Returns:
        A list of dictionaries containing time, mAP value and parameter combinations used.
    """

    parameter_combinations = all_dict_combinations(parameter_lists)

    results = [{} for _ in range(len(parameter_combinations))]

    for i in range(len(parameter_combinations)):
        params = parameter_combinations[i]
        mAP, time_ms = benchmark_ms(function, **params)
        results[i] = {"mAP":mAP , "execution_time_ms": time_ms, **params}

    results.sort(key = lambda d: d["mAP"], reverse=True)

    return results