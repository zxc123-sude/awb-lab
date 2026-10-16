"""Optional learned-AWB checks; cleanly skipped when PyTorch is absent."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from awb_lab.learning.__main__ import main
    from awb_lab.learning.checkpoint import load_checkpoint, save_checkpoint
    from awb_lab.learning.data import (check_disjoint, load_record, prepare_input,
                                       read_manifest)
    from awb_lab.learning.demo import make_demo
    from awb_lab.learning.model import TinyIlluminantNet, cosine_loss
    from awb_lab.learning.runner import evaluate, predict, train


@unittest.skipIf(torch is None, "optional PyTorch AI extra is not installed")
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.demo = make_demo(cls.root / "demo", train_count=8, val_count=3, test_count=3)
        cls.paths = {key: Path(value) for key, value in cls.demo["manifests"].items()}
        cls.report = train(cls.paths["train"], cls.paths["val"], cls.root / "model",
                           epochs=3, batch_size=4, input_size=32, seed=42, allow_synthetic=True)
        cls.checkpoint = cls.root / "model/best.pt"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_real_gradient_and_optimizer_update(self):
        torch.manual_seed(7)
        image = np.ones((32, 32, 3)) * [.3, .2, .1]
        inputs = prepare_input(image, input_size=32).unsqueeze(0)
        target = torch.tensor([[1.5, 1., .5]])
        model = TinyIlluminantNet()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        loss = cosine_loss(model(inputs), target)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertTrue(torch.isfinite(cosine_loss(model(inputs), target)))

    def test_masked_rgb_cannot_leak_through_resize_or_exposure_scale(self):
        image = np.ones((37, 51, 3)) * [.3, .2, .1]
        mask = np.ones(image.shape[:2], bool)
        mask[5:21, 11:30] = False
        altered = image.copy()
        altered[~mask] = [.95, .01, .85]
        first, second = prepare_input(image, mask, 32), prepare_input(altered, mask, 32)
        self.assertTrue(torch.equal(first, second))
        model, _, _ = load_checkpoint(self.checkpoint, allow_synthetic=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(first.unsqueeze(0)), model(second.unsqueeze(0)), rtol=0, atol=0)

    def test_degenerate_images_and_masks_rejected(self):
        for image, mask in ((np.zeros((32, 32, 3)), None),
                            (np.ones((32, 32, 3)), None),
                            (np.ones((32, 32, 3)) * [.1, .2, 0], None),
                            (np.ones((32, 32, 3)) * .2, np.zeros((32, 32)))):
            with self.assertRaises(ValueError):
                prepare_input(image, mask, 32)

    def test_synthetic_guard_for_data_checkpoint_and_cli(self):
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            read_manifest(self.paths["train"])
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            load_checkpoint(self.checkpoint)
        with contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(["predict", str(self.checkpoint), str(self.root / "demo/test_000.npy"),
                         "--black-level", "0", "--white-level", "1"])
        self.assertEqual(code, 2)
        self.assertIn("allow-synthetic", error.getvalue())

    def test_scene_group_and_canonical_path_split_overlap_rejected(self):
        first = read_manifest(self.paths["train"], True)
        second = read_manifest(self.paths["val"], True)
        check_disjoint(first, second)
        same_group = dict(second, groups=[first["groups"][0]])
        with self.assertRaisesRegex(ValueError, "group_id"):
            check_disjoint(first, same_group)
        same_path = dict(second, image_paths=[first["image_paths"][0]])
        with self.assertRaisesRegex(ValueError, "canonical image"):
            check_disjoint(first, same_path)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(self.checkpoint, self.paths["val"], self.root / "leaked", allow_synthetic=True)

    def test_duplicate_canonical_images_in_manifest_rejected(self):
        source = self.root / "demo"
        with self.paths["train"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        duplicate = dict(rows[0], image="./" + rows[0]["image"], group_id="another_group")
        path = source / "duplicate.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([rows[0], duplicate])
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            read_manifest(path, True)

    def test_checkpoint_roundtrip_and_strict_schema(self):
        model, payload, digest = load_checkpoint(self.checkpoint, True)
        self.assertEqual(len(digest), 64)
        self.assertTrue(payload["metadata"]["synthetic"])
        self.assertEqual(payload["metadata"]["epoch"], self.report["best_epoch"])
        self.assertEqual(payload["metadata"]["train_count"], 8)
        self.assertEqual(payload["metadata"]["val_count"], 3)
        copy = self.root / "copy.pt"
        save_checkpoint(copy, model, 32, payload["metadata"])
        other, _, _ = load_checkpoint(copy, True)
        record = read_manifest(self.paths["test"], True)["records"][0]
        image, mask = load_record(record)
        inputs = prepare_input(image, mask, 32).unsqueeze(0)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), other(inputs), rtol=0, atol=0)
        malformed = dict(payload, unexpected="refuse unknown schema fields")
        torch.save(malformed, self.root / "malformed.pt")
        with self.assertRaisesRegex(ValueError, "schema"):
            load_checkpoint(self.root / "malformed.pt", True)

    def test_training_history_and_validation_selection(self):
        history = self.report["history"]
        self.assertEqual(len(history), 3)
        best = min(history, key=lambda row: row["val_mean_recovery_deg"])
        self.assertEqual(best["epoch"], self.report["best_epoch"])
        self.assertTrue(all(np.isfinite(row["train_cosine_loss"]) for row in history))
        saved = json.loads((self.root / "model/training.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.report)

    def test_heldout_evaluation_cnn_plus_all_baselines_and_prediction(self):
        summary = evaluate(self.checkpoint, self.paths["test"], self.root / "eval", allow_synthetic=True)
        self.assertEqual(summary["sample_count"], 3)
        self.assertTrue(summary["heldout_groups_checked"])
        self.assertTrue(summary["synthetic"])
        self.assertEqual(set(summary["methods"]), {"tiny_cnn", "gray_world", "white_patch", "shades_of_gray", "gray_edge"})
        with (self.root / "eval/predictions.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(np.isfinite(float(row["recovery_error_deg"])) for row in rows))
        result = predict(self.checkpoint, self.root / "demo/test_000.npy", 0, 1,
                         self.root / "demo/test_000_mask.npy", self.root / "corrected.npy", allow_synthetic=True)
        self.assertTrue(result["synthetic"])
        self.assertFalse(result["seen_train_image"])
        self.assertAlmostEqual(np.linalg.norm(result["illuminant_l2_rgb"]), 1, places=6)
        self.assertEqual(result["green_normalized_gains"][1], 1)
        self.assertTrue((self.root / "corrected.npy").exists())


if __name__ == "__main__":
    unittest.main()
