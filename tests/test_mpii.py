import numpy as np
import pytest

from torchdata import mpii


def test_validate_mpii_data_dir(mpii_data_dir, tmpdir):
    mpii.validate_mpii_data_dir(mpii_data_dir)
    with pytest.raises(AssertionError):
        mpii.validate_mpii_data_dir(tmpdir.join('non-existent'))


def test_subset_splits(mpii_data_dir):
    data = mpii.MpiiData(mpii_data_dir)
    assert [e for e in data.train_indices if e < 12] == [5, 6, 7, 8, 9, 11]
    assert [e for e in data.val_indices if e < 12] == [10]
    assert [e for e in data.test_indices if e < 12] == [0, 1, 2, 3, 4]


def test_get_bounding_box(mpii_data_dir):
    data = mpii.MpiiData(mpii_data_dir)
    bb = data.get_bounding_box(4)
    np.testing.assert_allclose(bb, (67.939938, -233.132855, 1300.060062, 998.987269))


def test_cropped_keypoints(mpii_data_dir):
    data = mpii.MpiiData(mpii_data_dir)
    index = 7
    matrix = data.get_crop_transform(index)
    keypoints = mpii.transform_keypoints(data.keypoints[index], matrix)
    assert keypoints.shape == (16, 2)
    np.testing.assert_allclose(keypoints[0], np.array([23.09556, 79.016413]))


def test_load_cropped_image(mpii_data_dir):
    data = mpii.MpiiData(mpii_data_dir)
    index = 0
    image = data.load_cropped_image(index, size=512)
    assert image.size == (512, 512)


def test_normalised_coordinate_transform():
    matrix = mpii.normalised_coordinate_transform(3)
    mapping = [
        ([-0.5, -0.5], [-1.0, -1.0]),
        ([2.5, 2.5], [1.0, 1.0]),
        ([1.0, 1.0], [0.0, 0.0])
    ]
    for p, q in mapping:
        np.testing.assert_almost_equal(mpii.transform_keypoints(p, matrix), q)


# def test_mpii_data():
#     import PIL.ImageDraw
#
#     data = mpii.MpiiData('/data/mpii')
#     index = 7
#     image = data.load_cropped_image(index)
#     keypoints = mpii.transform_keypoints(data.keypoints[index], data.get_crop_transform(index))
#     PIL.ImageDraw.Draw(image).line([*keypoints[mpii.MPII_Joint_Names.index('head_top')],
#                                     *keypoints[mpii.MPII_Joint_Names.index('neck')]])
#     image.show()


# def test_normalise():
#     size = 384
#     points = np.array([
#         [0, 0, 1],
#         [size - 1, size - 1, 1],
#     ])
#     matrix = np.array([
#         [2 / (size - 1),              0, -1],
#         [             0, 2 / (size - 1), -1]
#     ])
#     print(np.matmul(points, matrix.transpose()))
