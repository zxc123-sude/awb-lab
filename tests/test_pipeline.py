import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from awb_lab.cli import main
from awb_lab.demo import make_demo
from awb_lab.evaluation import _stats, evaluate
from awb_lab.image_io import linear_to_srgb, load_mask, load_rgb, save_rgb, srgb_to_linear


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_uint16_rgb_keeps_precision_and_original_level_units(self):
        rgb = np.array([[[1000, 2000, 3000], [512, 16383, 513]]], dtype=np.uint16)
        path = self.root / "线性16位.png"
        success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(success)
        encoded.tofile(str(path))
        loaded = load_rgb(path, "linear", black_level=512, white_level=16383)
        np.testing.assert_allclose(loaded, (rgb.astype(float) - 512) / (16383 - 512))
        self.assertGreater(loaded[0, 1, 2], 0)
        with self.assertRaises(ValueError):
            load_rgb(path, "linear")

    def test_srgb_transfer_roundtrip(self):
        values = np.array([0., .02, .04045, .1, .5, 1.])
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=3e-8)

    def test_linear_output_is_not_clipped(self):
        original = np.array([[[2.1, 0.1, .4]]])
        path = self.root / "out.npy"
        save_rgb(path, original)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), original)

    def test_jpeg_cannot_be_evaluated_as_linear(self):
        with self.assertRaisesRegex(ValueError, "JPEG"):
            load_rgb(self.root / "anything.jpg", "linear", white_level=255)

    def test_npy_pickle_disabled_and_mask_convention(self):
        path = self.root / "mask.npy"
        np.save(path, [[0, 2], [1, 0]], allow_pickle=False)
        np.testing.assert_array_equal(load_mask(path, (2, 2)), [[False, True], [True, False]])
        np.save(path, np.array([{"unsafe": "object"}], dtype=object))
        with self.assertRaises(ValueError):
            load_rgb(path, "linear", white_level=1)

    def test_demo_evaluation_end_to_end(self):
        manifest = make_demo(self.root / "data")
        summary = evaluate(manifest, self.root / "results")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["dataset_types"], ["synthetic_smoke_test"])
        self.assertEqual(len(summary["methods"]), 4)
        for result in summary["methods"].values():
            self.assertEqual(result["count"], 3)
            self.assertLess(result["recovery_error_deg"]["mean"], 1e-6)
            self.assertLess(result["reproduction_error_deg"]["mean"], 1e-6)
        saved_summary = json.loads((self.root / "results/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_summary, summary)
        with (self.root / "results/predictions.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)

    def test_manifest_error_reports_line_without_silently_dropping_sample(self):
        manifest = self.root / "bad.csv"
        manifest.write_text("image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level\nmissing.npy,1,1,1,0,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest line 2"):
            evaluate(manifest, self.root / "results")
        self.assertFalse((self.root / "results/predictions.csv").exists())

    def test_missing_white_level_not_guessed(self):
        manifest = self.root / "bad.csv"
        manifest.write_text("image,illuminant_r,illuminant_g,illuminant_b,black_level\na.png,1,1,1,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "white_level"):
            evaluate(manifest, self.root / "results")

    def test_missing_or_empty_black_level_not_guessed(self):
        for header, row in [
            ("image,illuminant_r,illuminant_g,illuminant_b,white_level", "a.npy,1,1,1,1"),
            ("image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level", "a.npy,1,1,1,,1"),
        ]:
            with self.subTest(header=header):
                manifest = self.root / "bad-black.csv"
                manifest.write_text(header + "\n" + row + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "black_level"):
                    evaluate(manifest, self.root / "results")

    def test_evaluation_excludes_saturation_by_default(self):
        image = np.array([[[.3, .2, .1], [1.2, .05, .8]]])
        np.save(self.root / "saturated.npy", image, allow_pickle=False)
        manifest = self.root / "saturated.csv"
        manifest.write_text(
            "image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level\n"
            "saturated.npy,3,2,1,0,1\n", encoding="utf-8")
        summary = evaluate(manifest, self.root / "results", methods=["white_patch"])
        self.assertEqual(summary["parameters"]["near_clip"], 1.0)
        self.assertEqual(summary["methods"]["white_patch"]["recovery_error_deg"]["mean"], 0.0)
        with self.assertRaisesRegex(ValueError, "near_clip"):
            evaluate(manifest, self.root / "bad", methods=["white_patch"], near_clip=1.1)

    def test_worst_quartile_uses_ceiling_sample_count(self):
        stats = _stats([1, 2, 3, 4, 5])
        self.assertEqual(stats["median"], 3)
        self.assertEqual(stats["worst_25_percent_mean"], 4.5)

    def test_cli_correction_and_failure_exit(self):
        source = self.root / "linear.npy"
        np.save(source, np.ones((16, 16, 3)) * [0.4, 0.2, 0.1], allow_pickle=False)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["correct", str(source), "--output", str(self.root / "corrected.npy"),
                         "--input-space", "linear", "--white-level", "1"])
        self.assertEqual(code, 0)
        info = json.loads(output.getvalue())
        np.testing.assert_allclose(info["green_normalized_gains"], [.5, 1, 2])
        np.testing.assert_allclose(np.load(self.root / "corrected.npy"), .2)
        with contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(["correct", str(source), "--output", str(self.root / "bad.npy"), "--input-space", "linear"])
        self.assertEqual(code, 2)
        self.assertIn("white_level", error.getvalue())


if __name__ == "__main__":
    unittest.main()
