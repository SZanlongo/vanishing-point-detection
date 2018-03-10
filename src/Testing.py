import os

import cv2

from src.vanishing_point import hough_transform, find_intersections, sample_lines, find_vanishing_point

for subdir, dirs, files in os.walk('../pictures/input'):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".jpg"):
            print(filepath)
            img = cv2.imread(filepath)
            hough_lines = hough_transform(img)
            if hough_lines:
                random_sample = sample_lines(hough_lines, 100)
                intersections = find_intersections(random_sample)
                if intersections:
                    grid_size = min(img.shape[0], img.shape[1]) // 3
                    vanishing_point = find_vanishing_point(img, grid_size, intersections)
                    filename = '../pictures/output/' + os.path.splitext(file)[0] + '_center' + '.jpg'
                    cv2.imwrite(filename, img)
