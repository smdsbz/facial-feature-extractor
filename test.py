# -*- coding: utf-8 -*-

import face_recognition as fr
from facial_feature_extractor import        \
    get_face_crop_coords, get_face_crops,   \
    get_landmarks_coords

from matplotlib import pyplot as plt


def disp_img(img, coords_list=[], width=None):
    imgcp = img.copy()
    h, w = imgcp.shape[0] - 1, imgcp.shape[1] - 1
    width = max(width // 2, 1) if width else int(max(min(h, w) / (2 * 1e2), 2))
    # mark-out rectangles specified by `coords_list`
    for coord in coords_list:
        top, right, bottom, left = coord
        imgcp[top:bottom, max(left - width, 0):min(left + width, w)] = (0, 255, 0)
        imgcp[top:bottom, max(right - width, 0):min(right + width, w)] = (0, 255, 0)
        imgcp[max(top - width, 0):min(top + width, h), left:right] = (0, 255, 0)
        imgcp[max(bottom - width, 0):min(bottom + width, h), left:right] = (0, 255, 0)

    # display result image
    plt.imshow(imgcp)
    plt.show()
    return


if __name__ == '__main__':

    img = fr.load_image_file('C:/Users/smdsbz/Desktop/test.jpg')
    print('image of size:', img.shape)
    disp_img(img)

    # crops = get_face_crops(img)
    # if not crops:
    #     print('Failed to find any faces in this image!')
    # else:
    #     print('Found', len(crops), 'face(s) in image!')
    #     for each in crops:
    #         disp_img(each)

    coords = get_face_crop_coords(img, fr_model='hog')
    print('Found', len(coords), 'face(s) in image!')
    disp_img(img, coords_list=coords)

    landmarks = get_landmarks_coords(img, output_format='list',
                                     fr_model='large')
    _coords_list = []
    for coords in landmarks:
        for x, y in coords:
            _coords_list.append((
                y - 1, x + 1, y + 1, x - 1
            ))
    disp_img(img, coords_list=_coords_list, width=2)
