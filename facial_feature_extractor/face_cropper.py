# -*- coding: utf-8 -*-

import face_recognition as fr


def get_face_crop_coords(img, fr_number_of_times_to_upsample=1,
                         fr_model='hog', **kwargs):
    '''
    Get the coordinate for cropped segments of faces from image.
    从图片中截取出包含人脸的部分。

    Args:
        img: `np.ndarray` Image in tensor
        fr_number_of_times_to_upsample: `int` See keyword argument
            `number_of_times_to_upsample` of `face_recognition.face_locations`
        fr_model: `str` See keyword argument `model` of
            `face_recognition.face_locations`

    Return:
        `list` List of quaternions of borders of the crop in CSS order. E.g.
        `[ ( top, right, bottom, left ), ... ]`
    '''
    return fr.face_locations(
        img,
        number_of_times_to_upsample=fr_number_of_times_to_upsample,
        model=fr_model
    )


def crop_image(img, css_ordered_coord):
    top, right, bottom, left = css_ordered_coord
    top, left = max(top - 1, 0), max(left - 1, 0)
    return img[top:bottom, left:right]


def get_face_crops(img, **kwargs):
    coords = get_face_crop_coords(img, **kwargs)
    ret = list(map(
        lambda coord: crop_image(img, coord),
        coords
    ))
    return ret
