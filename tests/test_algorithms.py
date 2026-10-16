import unittest

import numpy as np

from awb_lab import (METHODS, apply_white_balance, estimate_illuminant,
                     recovery_angular_error, reproduction_angular_error)


class IlluminantTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:48, :64]
        surface = 0.1 + x / 200 + y / 300 + 0.02 * np.sin(x / 4)
        self.light = np.array([1.6, 1.0, 0.7])
        self.image = surface[..., None] * self.light
        self.expected = self.light / np.linalg.norm(self.light)

    def test_all_methods_recover_diagonal_light_on_neutral_texture(self):
        for method in METHODS:
            with self.subTest(method=method):
                estimated = estimate_illuminant(self.image, method)
                np.testing.assert_allclose(estimated, self.expected, rtol=1e-12, atol=1e-12)
                self.assertAlmostEqual(np.linalg.norm(estimated), 1)

    def test_gray_world_matches_hand_calculated_mean(self):
        image = np.array([[[1., 2., 3.], [3., 2., 1.]]])
        np.testing.assert_allclose(estimate_illuminant(image), np.ones(3) / np.sqrt(3))

    def test_white_patch_matches_channel_maxima(self):
        image = np.array([[[1., 2., 3.], [4., 1., 2.]]])
        expected = np.array([4., 2., 3.])
        np.testing.assert_allclose(estimate_illuminant(image, "white_patch"), expected / np.linalg.norm(expected))

    def test_shades_p_one_equals_gray_world(self):
        colored = self.image.copy()
        colored[:10, :10] *= [1.2, 0.5, 0.8]
        np.testing.assert_allclose(estimate_illuminant(colored), estimate_illuminant(colored, "shades_of_gray", p=1))

    def test_scale_invariance(self):
        for method in METHODS:
            with self.subTest(method=method):
                np.testing.assert_allclose(estimate_illuminant(self.image * 2000, method), self.expected)

    def test_mask_excludes_contamination_including_gray_edge_support(self):
        contaminated = self.image.copy()
        contaminated[15:28, 25:38] = [1e5, 2e2, 5e3]
        mask = np.ones(contaminated.shape[:2], dtype=bool)
        mask[15:28, 25:38] = False
        for method in METHODS:
            with self.subTest(method=method):
                np.testing.assert_allclose(estimate_illuminant(contaminated, method, mask), self.expected, atol=1e-12)

    def test_absolute_near_clip_excludes_any_channel(self):
        image = np.array([[[.3, .2, .1], [1., .1, .8]]])
        expected = np.array([3., 2., 1.]) / np.sqrt(14)
        np.testing.assert_allclose(estimate_illuminant(image, near_clip=.99), expected)

    def test_green_normalized_gains_preserve_green_and_do_not_clip(self):
        image = np.array([[[.8, .7, .6]]])
        result = apply_white_balance(image, [.4, .8, .2])
        np.testing.assert_allclose(result, [[[1.6, .7, 2.4]]])
        np.testing.assert_array_equal(image, [[[.8, .7, .6]]])

    def test_invalid_inputs_fail_explicitly(self):
        invalid = [np.zeros((3, 3)), np.zeros((0, 4, 3)), np.zeros((4, 4, 4)),
                   np.full((4, 4, 3), np.nan), np.full((4, 4, 3), -1)]
        for image in invalid:
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    estimate_illuminant(image)
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    estimate_illuminant(np.zeros((32, 32, 3)), method)
                with self.assertRaises(ValueError):
                    estimate_illuminant(self.image, method, np.zeros(self.image.shape[:2], bool))
        zero_channel = self.image.copy()
        zero_channel[..., 2] = 0
        with self.assertRaises(ValueError):
            estimate_illuminant(zero_channel)

    def test_bad_parameters_and_masks(self):
        kwargs_cases = [dict(method="unknown"), dict(p=0), dict(p=np.inf), dict(sigma=0),
                        dict(near_clip=-1), dict(mask=np.ones((2, 2))),
                        dict(mask=np.full(self.image.shape[:2], np.nan))]
        for kwargs in kwargs_cases:
            with self.subTest(kwargs=str(kwargs)):
                with self.assertRaises(ValueError):
                    estimate_illuminant(self.image, **kwargs)

    def test_gray_edge_rejects_no_derivative_support(self):
        with self.assertRaises(ValueError):
            estimate_illuminant(np.ones((32, 32, 3)), "gray_edge")
        with self.assertRaises(ValueError):
            estimate_illuminant(np.ones((8, 8, 3)), "gray_edge")


class MetricTests(unittest.TestCase):
    def test_identity_and_scale(self):
        for metric in (recovery_angular_error, reproduction_angular_error):
            self.assertEqual(metric([2, 1, 3], [20, 10, 30]), 0.0)

    def test_known_recovery_angle(self):
        expected = np.degrees(np.arccos(4 / np.sqrt(18)))
        self.assertAlmostEqual(recovery_angular_error([1, 1, 1], [1, 1, 2]), expected)

    def test_reproduction_formula(self):
        estimate = np.array([1., 2., 3.])
        truth = np.array([3., 1., 2.])
        expected = recovery_angular_error(truth / estimate, np.ones(3))
        self.assertAlmostEqual(reproduction_angular_error(estimate, truth), expected)

    def test_bad_vectors(self):
        for metric in (recovery_angular_error, reproduction_angular_error):
            for vector in ([0, 1, 1], [-1, 1, 1], [1, 1, np.nan], [1, 1]):
                with self.assertRaises(ValueError):
                    metric(vector, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
