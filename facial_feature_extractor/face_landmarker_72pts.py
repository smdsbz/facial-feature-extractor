# -*- coding: utf-8 -*-


import face_recognition as fr


def get_landmarks_coords(img, output_format='list',
                         fr_face_locations=None,
                         fr_model='large'):
    '''
    get 72-point face landmark coordinates feature

    Args:
        img: `np.ndarray` Image in 256 stages representation
        output_format: `'list' | 'dict'` Return type
        fr_face_locations: `None | list` See keyword argument `face_locations`
            of `face_recognition.face_landmarks`
        fr_model: `str` See keyword argument `model` of
            `face_recognition.face_landmarks` (choosing `'small'` will give a
            very limited landmark of 5 points per face)

    Return:
        `output_format == 'list'`: [ [ (x0, y0), ... ], second_face, ... ]
        `output_format == 'dict'`: [ { 'chin': [ (x0, y0), ... ], ... },
                                     second_face, ... ]
    '''
    # get raw api return
    result = fr.face_landmarks(
        img,
        face_locations=fr_face_locations,
        model=fr_model
    )

    # output formatting

    # original api return format, keys in dict are:
    #   chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye,
    #   right_eye, top_lip, bottom_lip
    if output_format == 'dict':
        pass

    # flattened
    elif output_format == 'list':
        for idx in range(len(result)):
            coord_dict = result[idx]
            flattened = []
            for key in coord_dict:
                flattened += coord_dict[key]
            result[idx] = flattened

    else:
        raise ValueError('argument `output_format`: {}'.format(output_format))

    return result
