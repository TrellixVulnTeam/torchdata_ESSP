import os

import PIL.Image
import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fake_mpii_data_dir(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp('mpii-data')

    image_indices = np.array([0, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6])
    person_indices = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0, 1])
    is_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    image_names = ['037454012.jpg', '095071431.jpg', '095071431.jpg', '073199394.jpg',
                   '059865848.jpg', '015601864.jpg', '015601864.jpg', '015599452.jpg',
                   '015599452.jpg', '015599452.jpg', '005808361.jpg', '005808361.jpg']
    imgnames = np.zeros((12, 16))
    for i, image_name in enumerate(image_names):
        imgnames[i, :len(image_name)] = np.frombuffer(image_name.encode('ascii'), dtype=np.uint8)
    subject_centres = np.array([[ 601,  380], [ 881,  394], [ 338,  210], [ 619,  350],
                                [ 684,  309], [ 594,  257], [ 952,  222], [ 619,  329],
                                [1010,  412], [ 133,  315], [ 966,  340], [ 489,  383]])
    subject_scales = np.array([3.88073395, 8.07816613, 8.90412938, 4.32666153,
                               4.92848050, 3.02104618, 2.47211650, 5.64127645,
                               6.07105131, 5.72816201, 4.71848789, 4.73408745])
    keypoints = np.zeros((12, 16, 2))
    keypoint_visibilities = np.ones((12, 16))
    head_lengths = np.array([  1.        ,   1.        ,   1.        ,   1.        ,
                               1.        ,  75.52615441,  61.80291255, 141.03191128,
                             151.77628273, 143.20405022, 117.96219733, 118.35218629])
    with h5py.File(data_dir.join('annot.h5'), 'w') as f:
        f.create_dataset('/index', dtype='<i8', data=image_indices)
        f.create_dataset('/person', dtype='<i8', data=person_indices)
        f.create_dataset('/istrain', dtype='<i8', data=is_train)
        f.create_dataset('/imgname', dtype='<f8', data=imgnames)
        f.create_dataset('/center', dtype='<i4', data=subject_centres)
        f.create_dataset('/scale', dtype='<f8', data=subject_scales)
        f.create_dataset('/part', dtype='<f8', data=keypoints)
        f.create_dataset('/visible', dtype='<f8', data=keypoint_visibilities)
        f.create_dataset('/normalize', dtype='<f8', data=head_lengths)

    with h5py.File(data_dir.join('valid.h5'), 'w') as f:
        f.create_dataset('/index', dtype='<f8', data=np.array([6]))
        f.create_dataset('/person', dtype='<f8', data=np.array([0]))

    image_dir = data_dir.mkdir('images')
    for image_name in set(image_names):
        with open(image_dir.join(image_name), 'wb') as f:
            PIL.Image.new('RGB', (1280, 720)).save(f)

    return data_dir


@pytest.fixture(scope="session")
def mpii_data_dir():
    data_dir = os.environ.get('TORCHDATA_MPII_DATA_DIR', None)
    if data_dir is not None and os.path.isdir(data_dir):
        return data_dir
    pytest.skip('requires MPII data')
