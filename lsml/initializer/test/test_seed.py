import unittest

import numpy
from skimage.morphology import disk, ball

from lsml.core.datasets_handler import DatasetExample
from lsml.initializer.seed import center_of_mass_seeder


class TestSeed(unittest.TestCase):

    def test_center_of_mass_seeder_centered_circle(self):

        seg = numpy.pad(disk(10), 10, 'constant')

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 1.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20]

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_centered_circle_anisotropic(self):

        seg = numpy.pad(disk(10), 10, 'constant')

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 0.5],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 10]

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_centered_circle_anisotropic2(self):

        # Decimate in the j-axis direction, but counter with dx[1] = 2 below
        seg = numpy.pad(disk(10), 10, 'constant')[:, ::2]

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 2.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20]

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_translated_circle(self):

        translation = numpy.r_[5, -3]

        seg = numpy.pad(disk(10), 10, 'constant')
        seg = numpy.roll(
            numpy.roll(seg, translation[0], axis=0), translation[1], axis=1)

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 1.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20] + translation

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_translated_circle_anisotropic(self):

        translation = numpy.r_[5, -3]

        seg = numpy.pad(disk(10), 10, 'constant')
        seg = numpy.roll(
            numpy.roll(seg, translation[0], axis=0), translation[1], axis=1)

        dx = numpy.r_[1, 0.5]
        example = DatasetExample(
            seg=seg, dx=dx,
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 10] + translation * dx

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_centered_ball(self):

        seg = numpy.pad(ball(10), 10, 'constant')

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 1., 1.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20, 20]

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_centered_ball_anisotropic(self):

        # Down-sample the last axis and account for it in the delta term
        seg = numpy.pad(ball(10), 10, 'constant')[:, :, ::2]

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 1., 2.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20, 20]

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)

    def test_center_of_mass_seeder_translated_ball(self):

        translation = numpy.r_[5, -3, 4.]
        seg = numpy.pad(ball(10), 10, 'constant')

        # Translate the segmentation
        for itrans, trans in enumerate(translation):
            seg = numpy.roll(seg, int(trans), axis=itrans)

        example = DatasetExample(
            seg=seg, dx=numpy.r_[1., 1., 1.],
            # Provide blank values for unused parameters
            index=None, key=None, img=None, dist=None)

        seed = center_of_mass_seeder(dataset_example=example)
        expected_seed = numpy.r_[20, 20, 20] + translation

        self.assertLess(numpy.linalg.norm(seed - expected_seed), 1e-8)
