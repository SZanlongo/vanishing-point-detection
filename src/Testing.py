import os

from VanishingPoint import *

i = 0
for subdir, dirs, files in os.walk('../pictures/input'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):
            i += 1
            img = cv2.imread(filepath)

            hough_lines = hough_transform(img)
            if hough_lines:
                random_sample = sample_lines(hough_lines, 100)
                intersections = find_intersections(random_sample, img)
                if intersections:
                    grid_size = min(img.shape[0], img.shape[1]) // 3
                    vanishing_point = find_vanishing_point(img, grid_size, intersections)

                    filename = '../pictures/output/center' + str(i) + '.jpg'

                    cv2.imwrite(filename, img)
