import cv2
import numpy as np
import matplotlib.pyplot as plt

def cxcywh_to_xyxy(box):
    """
    Convert bounding box from 'cxcywh' format to 'xyxy' format.

    Args:
    - box: A tuple or list (cx, cy, w, h) where:
        cx: center x-coordinate
        cy: center y-coordinate
        w: width of the bounding box
        h: height of the bounding box

    Returns:
    - A tuple (xmin, ymin, xmax, ymax) corresponding to the bounding box in 'xyxy' format.
    """
    cx, cy, w, h = box
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    
    return (xmin, ymin, xmax, ymax)

def convert_pil_to_cv2(pil_image):
    """
    Convert a PIL image (RGB) to an OpenCV image (BGR)

    Args:
    - pil_image: A PIL image

    Returns:
    - An OpenCV image
    """
    # Convert PIL Image to NumPy array
    img_array = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_bgr

def sort_by_scores(scores, preds):
    '''
    Sort predictions by scores in descending order
    '''
    scores = [score.cpu().item() for score in scores]
    scores_indices = np.argsort(-np.array(scores))

    if not scores_indices.tolist() == list(range(len(scores))):
        # print("* [INFO] Sorting by scores in descending order")
        preds = [[t[k] for k in scores_indices] for t in preds]

    return preds

def visualize_crops(crops):
    '''
    Visualize crops as a grid for detection refinement with VLM
    '''
    num_crops = len(crops)

    fig, axes = plt.subplots(
        1, num_crops, figsize=(num_crops * 10, 10), squeeze=False
    )

    for i in range(num_crops):
        axes[0, i].imshow(crops[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{i}", fontsize=100)
    plt.tight_layout()

    return fig

def transform_src_to_tgt(
    # Source point
    position_src: np.array,
    orientation_src: np.array,
    # Target perspective (New origin, +z axis)
    origin_tgt: np.array,
    z_axis_tgt: np.array,
    up_vector_global: np.array = np.array([0, -1, 0]),
):
    '''
    Transform a source point to a target perspective
    Input:
        position_src: np.array
        orientation_src: np.array
        origin_tgt: np.array
        z_axis_tgt: np.array
        up_vector_global: np.array

    Output:
        position_tgt: np.array
        orientation_tgt: np.array
    '''
    # Translation source point to origin
    position_src_translated = position_src - origin_tgt

    # Calculate right, up vectors for camera coord. system
    # Normalize the lookat (target) vector
    forward_vector = z_axis_tgt / np.linalg.norm(z_axis_tgt)
    # TODO: check this
    if np.all(up_vector_global == np.array([0, -1, 0])):
        right_vector = np.cross(forward_vector, up_vector_global)
    else:
        right_vector = np.cross(up_vector_global, forward_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # Recalculate the up vector
    up_vector_cam = np.cross(forward_vector, right_vector)
    up_vector_cam = up_vector_cam / np.linalg.norm(up_vector_cam)

    # Create rotation matrix
    rotation_matrix = np.vstack(
        (right_vector, up_vector_cam, forward_vector)
    )

    # Trasform source point to target coord. system (perspective)
    position_src_transformed = rotation_matrix @ position_src_translated
    orientation_src_transformed = rotation_matrix @ orientation_src

    # Round to 3 decimal places
    position_src_transformed = np.round(position_src_transformed, 3)
    orientation_src_transformed = np.round(orientation_src_transformed, 3)

    return position_src_transformed, orientation_src_transformed