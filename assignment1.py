import numpy as np
import skimage.feature as skfeat
import cv2
from holidays_dataset_handler import HolidaysDatasetHandler

def compute_1d_color_hist(img, bins_per_hist = 32):
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
        histograms[i, :] = h/np.linalg.norm(h)
        
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
    if(query_desc[0].dtype=='float32'):
        matcher = cv2.FlannBasedMatcher()
        matches = matcher.knnMatch(query_desc, database_desc, k)
        return matches
    else:
        matcher = cv2.BFMatcher_create()
        matches = matcher.knnMatch(query_desc, database_desc, k)
        return matches

    
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
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    return good


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
    images = load