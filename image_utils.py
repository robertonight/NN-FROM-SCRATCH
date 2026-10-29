from collections import Counter
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from image_utils import *

# Display
def show_image(image, title=None, cmap=None):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


# Load
def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image

# Color conversion
def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Resize
def resize_ignore_aspect(image, width, height):
    return cv2.resize(image, (width, height))
def resize_keep_aspect(image, width, height):
    h, w = image.shape[:2]
    aspect = w / float(h)

    if aspect > 1:
        new_w = int(height * aspect)
        return cv2.resize(image, (new_w, height))
    elif aspect < 1:
        new_h = int(width / aspect)
        return cv2.resize(image, (width, new_h))
    else:
        return cv2.resize(image, (width, height))


# Transform 
def flip_image(image, mode): 
    #mode 0=vertical, 1=horizontal, -1=both
    return cv2.flip(image, mode)


def rotate_image(image, angle, scale=1.0):
    w, h = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


# Intensity adjust
def adjust_brightness(image, value):
    if value > 0:
        M = np.ones(image.shape, dtype="uint8") * value
        return cv2.add(image, M)
    elif value < 0:
        M = np.ones(image.shape, dtype="uint8") * abs(value)
        return cv2.subtract(image, M)


# Shape detection
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        return "pentagon"
    elif len(approx) == 6:
        return "hexagon"
    else:
        return "circle"

# TODO: À revoir, ne marche pas pour certaines images
def preprocess_for_contours(image, width=300):
    #ratio = image.shape[0] / float(image.shape[0])
    ratio = 1.0
    preprocess_img  = image.copy()
    if len(image.shape) != 2:  # already 1 channel
        preprocess_img = rgb_to_gray(image.copy())
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #preprocess_img = clahe.apply(preprocess_img)
    # Flou pour reduire le bruit
    #blurred = cv2.GaussianBlur(preprocess_img, (5, 5), 0) #10709
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #thresh = cv2.Canny(blurred, 50, 150)

    bilateral = cv2.bilateralFilter(preprocess_img, 11, 17, 17)
    thresh = cv2.Canny(bilateral, 60, 200)

    return preprocess_img, thresh, ratio


def find_and_label_shapes(image, thresh, ratio):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    labeled_image = image.copy()
    for c in cnts:
        if labeled_image is not None:
            result = draw_contours(labeled_image, c, detect_shape(c), ratio)
            if result is not None:
                labeled_image = result
    return labeled_image

def draw_contours(image, contour, shape, ratio):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return  # contour invalide, on ignore
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    c = (contour.astype("float") * ratio).astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(
        image,
        shape,
        (cX, cY),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )
    return image

def find_and_count_shapes(image, thresh, ratio, min_area=100):
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    labeled_image = image.copy()
    shape_count = Counter()

    if hierarchy is None:
        return shape_count, labeled_image
    hierarchy = hierarchy[0]


    for i, c in enumerate(cnts):
        if c is None or len(c) == 0:
            continue
        if hierarchy[i][2] != -1:  # a un enfant → on ignore
            continue
        shape = detect_shape(c)
        if shape is not None and shape not in ["square"]:
            if labeled_image is not None:
                result = draw_contours(labeled_image, c, shape, ratio)
                if result is not None:
                    shape_count[shape] += 1
                    labeled_image = result
    return shape_count, labeled_image

def extract_roi(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

def find_rois(image, thresh, ratio, min_area):
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    labeled_rois = image.copy()
    hierarchy = hierarchy[0]
    rois= []
    
    for i in range(len(cnts)):
        c = cnts[i]
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        shape = detect_shape(c)
        if shape not in ["square", "rectangle"]:
            continue
        has_child = hierarchy[i][2] != -1
        has_parent = hierarchy[i][3] != -1

        if not has_child or has_parent:
            continue  # on ignore les rectangles vides


        c = (c.astype("float") * ratio).astype("int")
        labeled_rois = draw_contours(labeled_rois, c, shape, ratio)
        rois.append(extract_roi(image, c))
    return rois, labeled_rois

def find_rois_2(image, thresh, ratio, min_area):
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    labeled_rois = image.copy()
    hierarchy = hierarchy[0]
    rois = []
    parents_added = set()  # track parents already added

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < min_area:
            continue

        # Skip rectangles as children
        shape = detect_shape(c)
        if shape in ["square", None]:
            continue

        # Only consider leaf shapes (no children)
        has_child = hierarchy[i][2] != -1
        if not has_child:
            continue

        # Must have a parent
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            continue

        # Skip if parent already added
        if parent_idx in parents_added:
            continue

        parent_shape = detect_shape(cnts[parent_idx])
        if parent_shape not in ["square", "rectangle"]:
            continue

        # Add parent contour and ROI
        parent_contour_scaled = (cnts[parent_idx].astype("float") * ratio).astype("int")
        labeled_rois = draw_contours(labeled_rois, parent_contour_scaled, "rectangle", ratio)
        roi = extract_roi(image, cnts[parent_idx])
        rois.append(roi)
        parents_added.add(parent_idx)

    return rois, labeled_rois

def from_label_to_shape(label): # TODO : définir l'utilité de cette fonction
    if "Cercle" in label:
        return "circle"
    elif "Diamant" in label:
        return "rectangle"
    elif "Hexagone" in label:
        return "hexagon"
    elif "Triangle" in label:
        return "triangle"
    else:
        return None

def filter_rois(rois, label):
    filtered_roi = None
    bestScore = 0
    labeled_filtered = None

    expected_shape_count = int(label[-1])
    for roi in rois:

        height_roi, width_roi = roi.shape[:2]

        # On s'assure de garder seulement les carres comme ROI
        aspect_ratio = float(width_roi) / float(height_roi)

        if aspect_ratio < 0.75 or aspect_ratio > 1.35:
            continue


        preprocess_img, thresh, ratio = preprocess_for_contours(roi)
        counter, labeled_image = find_and_count_shapes(preprocess_img, thresh, ratio)

        # Calcul du ratio entre le nombre de formes de la bonne categorie et le nombre de formes total
        shape_ratio = shape_ratio_label(counter, from_label_to_shape(label))

        # Calcul de l'aire du roi
        area = height_roi * width_roi
        # Calcul de la difference entre le nombre attendu de formes dans le ROI et le nombre actuel
        shape_count = sum(counter.values())
        shape_count_difference = abs(expected_shape_count-shape_count)

        # Calcul du score de la ROI
        score = (area * shape_ratio) / ((shape_count_difference + 1)**2) # puissance 2 pour punir severement si le nombre de shapes est mauvais

        #print(f"Counter: {counter}, Label Ratio: {label_ratio:.2f}, Shape Ratio: {shape_ratio:.2f}")
        # #show_image(preprocess_img, f"ROI with shape ratio {shape_ratio:.2f}")
        if score > bestScore:
            bestScore = score
            filtered_roi = roi
            labeled_filtered = labeled_image
    return filtered_roi, labeled_filtered

def shape_ratio_label(counter, label):
    if len(counter) == 0:
        return 0
    denom = counter[label]
    if label == 'circle':
        denom+= counter['hexagon'] + counter['pentagon']
    return denom/sum(counter.values())

def demo():
    print("Running demo...")
    base_path = Path.cwd() / "demo/images"
    '''
    file_name = "124_Circle2.jpg"
    image = load_image(base_path / file_name)
    image = bgr_to_rgb(image)
    show_image(image, "Original")

    gray = rgb_to_gray(image)
    show_image(gray, "Grayscale", cmap="gray")

    resized = resize_ignore_aspect(image, 160, 120)
    show_image(resized, "Resize (Ignore Aspect)")

    scaled = resize_keep_aspect(image, 160, 120)
    show_image(scaled, "Resize (Keep Aspect)")

    show_image(flip_image(image, 0), "Flip Vertical")
    show_image(flip_image(image, 1), "Flip Horizontal")

    show_image(adjust_brightness(image, 100), "Brightness +")
    show_image(adjust_brightness(image, -50), "Brightness -")

    show_image(rotate_image(image, 45), "Rotated")
    '''
    marker_name = "cozmo_custom_markers.jpg"
    shape_img = load_image(base_path / marker_name)
    shape_img = bgr_to_rgb(shape_img)

    preprocess_img, thresh, ratio = preprocess_for_contours(shape_img)
    show_image(thresh, "Threshold", cmap="gray")

    #result = find_and_label_shapes(shape_img.copy(), thresh, ratio)
    #show_image(result, "Detected Shapes")
    
    rois, labeled_rois = find_rois(shape_img.copy(), thresh, ratio, min_area=100)
    show_image(labeled_rois, "ROIs Detected")
    filtered_rois, labeled_filtered = filter_rois(rois)
    show_image(filtered_rois, "Filtered ROI")

if __name__ == "__main__":
    demo()