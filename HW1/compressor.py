import os
import cv2 as cv

# resize every image in the library to 1000x1000 and save back over it
for root, _, files in os.walk(os.path.join(os.getcwd(), 'library\in')):
        for file in files:
            if file.endswith('.jpeg'):
                image = cv.imread(os.path.join(root, file))
                resize_factor = 1000 / max(image.shape[0], image.shape[1])
                image = cv.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
                # save over the original image
                cv.imwrite(os.path.join(root, file), image)