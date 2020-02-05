from src.face_align import align
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import imageio

"""
Use 4 test image to apply all the available aligning method on them and show the difference as a gif animation.
"""

images = ["example_images/img1.jpg","example_images/img2.jpg","example_images/img3.jpg","example_images/img4.jpg"]
methods = ["skimage", "opencv_affine", "opencv_affine_partial"]
detector = MTCNN()

for image in images:
    img = cv2.imread(image)
    keypoints = detector.detect_faces(img)[0]["keypoints"]
    landmarks = [keypoints["left_eye"],
                 keypoints["right_eye"],
                 keypoints["nose"],
                 keypoints["mouth_left"],
                 keypoints["mouth_right"]]
    image_list = []
    for method in methods:
        aligned_image = align(method, img, landmarks)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        txt_image  =np.zeros((112, 96, 3))
        for i, txt in enumerate(method.split("_")):
            text = cv2.putText(img = txt_image,
                               text = txt,
                               org = (0, (i+1)*10),
                               fontFace = font,
                               fontScale = 0.50,
                               color = (255, 255, 255))
        new_image = np.concatenate((aligned_image, text), axis=1)
        image_list.append(cv2.cvtColor(new_image.astype("uint8"),cv2.COLOR_BGR2RGB))
    imageio.mimsave(image+".gif", image_list, duration=0.5)
