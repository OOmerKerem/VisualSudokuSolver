import numpy as np
import cv2

def roi_points(pts) :

    roi = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis=1)
    roi[0] = pts[np.argmin(s)] # top-left
    roi[2] = pts[np.argmax(s)] # bottom-right

    diff = np.diff(pts, axis=1)
    roi[1] = pts[np.argmin(diff)] # top-right
    roi[3] = pts[np.argmax(diff)] 

    return roi

def BirdsEyeView (image, pts):

    roi = roi_points(pts)
    (tl, tr, br, bl) = roi

    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width1), int(width2))

    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(height1), int(height2))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped