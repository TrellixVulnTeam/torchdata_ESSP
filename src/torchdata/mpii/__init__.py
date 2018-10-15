import tarfile
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import PIL.Image
import h5py
import numpy as np
from tqdm import tqdm

from torchdata.logger import log
from torchdata.utils import download_file, remote_file, md5sum


MPII_Joint_Names = ['right_ankle', 'right_knee', 'right_hip', 'left_hip',
                    'left_knee', 'left_ankle', 'pelvis', 'spine',
                    'neck', 'head_top', 'right_wrist', 'right_elbow',
                    'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist']
MPII_Joint_Parents = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 8, 8, 13, 14]
MPII_Joint_Horizontal_Flips = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

MPII_Files = {
    # Archive file containing the images (12 GiB)
    'mpii_human_pose_v1.tar.gz': {
        'url': 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz',
        'md5': 'b6bc9c6869d3f035a5570b2e68ec84c4',
    },
    # All annotations (23 MiB)
    'annot.h5': {
        'url': 'https://github.com/princeton-vl/pose-hg-train/raw/4637618a1b162d80436bfd0b557833b5824cbb21/data/mpii/annot.h5',
        'md5': 'c0d0ba453709e37d632b4d4059e2799c',
    },
    # Validation set annotations (1 MiB)
    'valid.h5': {
        'url': 'https://github.com/princeton-vl/pose-hg-train/raw/4637618a1b162d80436bfd0b557833b5824cbb21/data/mpii/annot/valid.h5',
        'md5': 'd88b6828485168c1fb4c79a21995fdef',
    },
}


def validate_mpii_data_dir(data_dir, thorough=False):
    data_dir = Path(data_dir)
    assert data_dir.is_dir()
    assert (data_dir / 'images').is_dir()
    assert (data_dir / 'annot.h5').is_file()
    assert (data_dir / 'valid.h5').is_file()
    if thorough:
        assert len(list((data_dir / 'images').glob('*.jpg'))) == 24984
        assert md5sum(data_dir / 'annot.h5', quiet=True) == MPII_Files['annot.h5']['md5']
        assert md5sum(data_dir / 'valid.h5', quiet=True) == MPII_Files['valid.h5']['md5']


def install_mpii_dataset(data_dir, quiet=False, force=False):
    """Download and extract the MPII Human Pose dataset.

    Args:
        data_dir (str): The destination directory for installation.
        quiet (bool): If true, don't show progress bars. Other output may be suppressed by
                      configuring the log level on `torchdata.logger.log`.
        force (bool): If true, skip checking whether the dataset is already installed.
    """
    if not force:
        try:
            # Exit early if it looks like the dataset has already been downloaded and extracted
            validate_mpii_data_dir(data_dir, thorough=True)
            return
        except:
            pass
    start_time = perf_counter()
    data_dir = Path(data_dir).absolute()
    log.info('Installing the MPII Human Pose dataset in {}:'.format(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info('[1/3] Gathering files...')
    val_annots_file = data_dir / 'valid.h5'
    download_file(dest_path=val_annots_file, quiet=quiet, **MPII_Files['valid.h5'])
    all_annots_file = data_dir / 'annot.h5'
    download_file(dest_path=all_annots_file, quiet=quiet, **MPII_Files['annot.h5'])
    with remote_file(**MPII_Files['mpii_human_pose_v1.tar.gz'], quiet=quiet) as img_archive:
        with tarfile.open(img_archive, 'r:gz') as tar:
            log.info('[2/3] Loading archive metadata...')
            subdir_members = [member for member in tar.getmembers()
                              if member.name.startswith('./images/')]
            log.info('[3/3] Extracting images...')
            if quiet:
                progress_bar = None
            else:
                progress_bar = tqdm(iterable=subdir_members, ascii=True, leave=False)
                subdir_members = progress_bar
            tar.extractall(path=str(data_dir), members=subdir_members)
            if progress_bar:
                progress_bar.close()
    duration_seconds = round(perf_counter() - start_time)
    log.info('Installation finished in {}.'.format(str(timedelta(seconds=duration_seconds))))


def transform_keypoints(keypoints, matrix):
    """Transform 2D keypoints using the given 3x3 transformation matrix."""
    keypoints = np.pad(keypoints, (0, 1), 'constant', constant_values=1)
    transformed_keypoints = np.matmul(keypoints, np.transpose(matrix))[..., :2]
    return transformed_keypoints


class MpiiData:
    """A helper class for working with MPII Human Pose data.

    Args:
        data_dir (Path): The directory containing installed MPII data.
    """

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        validate_mpii_data_dir(data_dir)
        self.data_dir = data_dir

        with h5py.File(data_dir / 'annot.h5', 'r') as f:
            self.image_indices = f['/index'].value.astype(np.uint32)
            self.person_indices = f['/person'].value.astype(np.uint32)
            self.is_train = f['/istrain'].value.astype(np.bool)
            self.image_names = [imgname.tostring().decode('ascii').split('\0')[0]
                                for imgname in f['/imgname'].value.astype(np.uint8)]
            self.subject_centers = f['/center'].value.astype(np.int32)
            self.subject_scales = f['/scale'].value.astype(np.float64)
            self.keypoints = f['/part'].value.astype(np.float64)
            self.keypoint_masks = f['/visible'].value.astype(np.uint8)
            self.head_lengths = f['/normalize'].value.astype(np.float64)

        with h5py.File(data_dir / 'valid.h5', 'r') as f:
            val_image_indices = f['/index'].value.astype(np.uint32)
            val_person_indices = f['/person'].value.astype(np.uint32)

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        # Separate the example indices into train, validation, and test sets
        for i in range(len(self.image_indices)):
            if self.is_train[i]:
                val_pos = len(self.val_indices)
                if(
                    val_pos < len(val_image_indices)
                    and self.image_indices[i] == val_image_indices[val_pos]
                    and self.person_indices[i] == val_person_indices[val_pos]
                ):
                    self.val_indices.append(i)
                else:
                    self.train_indices.append(i)
            else:
                self.test_indices.append(i)

    def __len__(self):
        return len(self.image_indices)

    def load_image(self, index):
        """Load the full original image."""
        image_name = self.image_names[index]
        image = PIL.Image.open(self.data_dir / 'images' / image_name, 'r')
        return image

    def get_bounding_box(self, index):
        """Return a bounding box for the subject in image space.

        Based on the cropping scheme of A. Newell et al.

        Args:
            index (int): Example index.

        Returns:
            Bounding box coordinates as a (left, upper, right, lower) tuple.
        """
        scale = self.subject_scales[index]
        cx = self.subject_centers[index, 0]
        cy = self.subject_centers[index, 1] + scale * 15
        half_size = (scale * 125)  # = (scale * 1.25 * 200) / 2
        return (cx - half_size, cy - half_size, cx + half_size, cy + half_size)

    def load_cropped_image(self, index, size=384):
        """Load a cropped version of the image centred on the subject."""
        bb = self.get_bounding_box(index)
        image = self.load_image(index)
        image = image.crop(bb)
        image.thumbnail((size, size))
        if image.width != size:
            image = image.resize((size, size))
        return image

    def get_crop_transform(self, index, size=384):
        """Build the matrix which transforms points from original to cropped image space."""
        bb = self.get_bounding_box(index)
        k = size / (bb[2] - bb[0])
        transform_matrix = np.array([
            [k, 0, -bb[0] * k],
            [0, k, -bb[1] * k],
            [0, 0,          1],
        ])
        return transform_matrix
