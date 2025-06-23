import cv2
import numpy as np


def compute_output_size(pts):
    """
    Compute dynamic size of license plate.

    Args:
        pts (array): 4 points [top-left, top-right, bottom-right, bottom-left]
    Returns:
        maxWidth (int): Maximum Width
        maxHeight (int): Maximum Height
    """
    pts = [np.array(p) for p in pts]
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))

    return maxWidth, maxHeight

def warp_perspective(frame, pts):
    """
    Format the license plate perspective.


    Args:
        frame(numpy.ndarray): Frame with the license plate
        pts(array): 4 points [top-left, top-right, bottom-right, bottom-left]
    Returns:
        warped(array): Formatted license plate.
    """
    width, height = compute_output_size(pts)
    dst = np.array([[0,0],[width-1,0], [width-1, height-1], [0, height-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(np.array(pts, dtype='float32'), dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped