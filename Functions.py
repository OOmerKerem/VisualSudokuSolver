import numpy as np
import cv2
import imutils

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

    M = cv2.getPerspectiveTransform(roi, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def SudokuFinder (image):
    kernel = (7, 7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, kernel, 3)
    thresholded = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    thresholded = cv2.bitwise_not(thresholded)

    contours = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)

    Vertice = None

    for c in contours :
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            Vertice = approx
            break

    sudoku = BirdsEyeView(image, Vertice.reshape(4, 2))
    warped = BirdsEyeView(gray, Vertice.reshape(4, 2))

    return (puzzle, warped)