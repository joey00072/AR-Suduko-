import cv2
import numpy as np


def get_gray_and_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 8
    )
    return gray, thresh


def find_best_contour(threshold):
    contours, hierarchy = cv2.findContours(
        threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    max_area = 0
    c = 0
    best_cnt = None
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
        c += 1
    return best_cnt


def approximate_contour(best_contour):
    approx = cv2.approxPolyDP(
        best_contour, 0.1 * cv2.arcLength(best_contour, True), True
    )
    return approx


def draw_contour(image, contour, color=(200, 200, 0)):
    cv2.drawContours(image, [contour], 0, color, 2)


def project_grid(image, approx, pers_height=450, pers_width=450):
    try:
        m = abs(
            (approx[0][0][1] - approx[1][0][1]) / (approx[0][0][0] - approx[1][0][0])
        )
    except ZeroDivisionError:
        m = 1000
    pts1 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
    if m < 0.8:
        pts2 = np.float32(
            [[pers_width, 0], [0, 0], [0, pers_height], [pers_height, pers_width]]
        )
    else:
        pts2 = np.float32(
            [[0, 0], [0, pers_height], [pers_height, pers_width], [pers_width, 0]]
        )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    pers = cv2.warpPerspective(image, matrix, (pers_height, pers_width))
    return pers


def un_project_grid(
    pers, image, approx, is_print=False, pers_height=450, pers_width=450
):
    try:
        m = abs(
            (approx[0][0][1] - approx[1][0][1]) / (approx[0][0][0] - approx[1][0][0])
        )
    except ZeroDivisionError:
        m = 1000
    pts2 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
    if m < 0.8:
        pts1 = np.float32(
            [[pers_width, 0], [0, 0], [0, pers_height], [pers_height, pers_width]]
        )
    else:
        pts1 = np.float32(
            [[0, 0], [0, pers_height], [pers_height, pers_width], [pers_width, 0]]
        )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    un_pers = cv2.warpPerspective(pers, matrix, (image.shape[1], image.shape[0]))

    return un_pers


def is_image_empty(image, thresh=30):
    return np.average(image) < np.min(image) + thresh


def overlay(image, overlay_image):
    mask = np.sum(overlay_image, axis=2).astype(np.uint8)
    mask[mask > 0] = 255
    immask = 255 - mask
    im1 = cv2.bitwise_and(image, image, mask=immask)
    return im1 + overlay_image
