"""CPU-only checks for download boundaries and the quickstart handoff."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_weights as downloads
import quickstart


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="turboclear test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def snapshot(self, **kwargs):
        root = Path(kwargs.get("local_dir") or self.root / "cache" / kwargs["repo_id"])
        for name in kwargs["allow_patterns"]:
            if "*" not in name:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        return str(root)

    def test_default_download_is_inference_only(self):
        with patch.object(downloads, "DEFAULT_WEIGHTS_DIR", self.root / "models"), \
                patch.object(downloads, "snapshot_download", side_effect=self.snapshot) as snapshot, \
                patch.object(downloads, "download_dataset") as dataset, \
                patch.object(sys, "argv", ["download_weights.py"]), \
                contextlib.redirect_stdout(io.StringIO()):
            downloads.main()
        dataset.assert_not_called()
        self.assertEqual(snapshot.call_count, 2)
        self.assertTrue((self.root / "models/TurboClear/sdxl/state_dict.pth").is_file())
        turbo_args = snapshot.call_args_list[1].kwargs
        self.assertEqual(set(turbo_args["allow_patterns"]),
                         {"sdxl/state_dict.pth", "fusion/fusion_module.pth"})

    def test_custom_directory_offline_and_missing_file(self):
        with patch.object(downloads, "snapshot_download", side_effect=self.snapshot):
            paths = downloads.download_weights(weights_dir=self.root)
        with patch.object(downloads, "snapshot_download") as network:
            self.assertEqual(paths, downloads.download_weights(self.root, local_files_only=True))
            Path(paths["FUSION_MODULE_PATH"]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "fusion_module.pth"):
                downloads.download_weights(self.root, local_files_only=True)
        network.assert_not_called()

    def test_cache_destination_and_partial_snapshot(self):
        with patch.object(downloads, "snapshot_download", side_effect=self.snapshot) as snapshot:
            downloads.download_weights(cache_dir=self.root / "hub")
        self.assertIsNone(snapshot.call_args.kwargs["local_dir"])
        self.assertEqual(snapshot.call_args.kwargs["cache_dir"], str(self.root / "hub"))
        with patch.object(downloads, "snapshot_download", return_value=str(self.root)):
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete weights"):
                downloads.download_weights(cache_dir=self.root, local_files_only=True)

    def test_dataset_requires_explicit_opt_in(self):
        with patch.object(downloads, "download_weights", return_value={}), \
                patch.object(downloads, "snapshot_download", side_effect=self.snapshot) as snapshot, \
                patch.object(sys, "argv", ["download_weights.py", "--with-dataset",
                                           "--datasets-dir", str(self.root)]), \
                contextlib.redirect_stdout(io.StringIO()):
            downloads.main()
        self.assertEqual(snapshot.call_args.kwargs["repo_type"], "dataset")
        self.assertEqual(len(list((self.root / "OBER/data").glob("*.parquet"))), 54)
        with patch.object(downloads, "snapshot_download") as network:
            downloads.download_dataset(self.root, local_files_only=True)
        network.assert_not_called()

    def test_quickstart_uses_active_python_and_resolved_paths(self):
        weights = {"BASE_MODEL_PATH": str(self.root / "ObjectClear"),
                   "WEIGHT_PATH": str(self.root / "TurboClear/sdxl"),
                   "FUSION_MODULE_PATH": str(self.root / "TurboClear/fusion/fusion_module.pth")}
        with patch.object(quickstart, "check_runtime"), \
                patch.object(quickstart, "download_weights", return_value=weights) as download, \
                patch.object(quickstart.subprocess, "run") as run, \
                patch.object(sys, "argv", ["quickstart.py", "--weights-dir", str(self.root),
                                           "--output-dir", str(self.root / "results"),
                                           "--local-files-only", "--max-samples", "1"]):
            quickstart.main()
        download.assert_called_once_with(self.root, None, True)
        self.assertEqual(run.call_args.args[0],
                         ["bash", str(quickstart.REPO_ROOT / "inference/inference_turboclear.sh")])
        settings = run.call_args.kwargs
        self.assertTrue(settings["check"])
        self.assertEqual(settings["cwd"], quickstart.REPO_ROOT)
        self.assertEqual(settings["env"]["PYTHON_BIN"], sys.executable)
        self.assertEqual(settings["env"]["INPUT_DIR"], str(quickstart.REPO_ROOT / "inputs/imgs"))
        self.assertEqual(settings["env"]["OUTPUT_DIR"], str(self.root / "results"))
        self.assertEqual(settings["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(settings["env"]["MAX_SAMPLES"], "1")

    def test_failed_preflight_does_not_download(self):
        with patch.object(quickstart, "check_runtime", side_effect=RuntimeError("No CUDA")), \
                patch.object(quickstart, "download_weights") as download, \
                patch.object(sys, "argv", ["quickstart.py"]):
            with self.assertRaisesRegex(RuntimeError, "No CUDA"):
                quickstart.main()
        download.assert_not_called()

    def test_inference_failure_propagates(self):
        with patch.object(quickstart, "check_runtime"), \
                patch.object(quickstart, "download_weights", return_value={}), \
                patch.object(quickstart.subprocess, "run",
                             side_effect=subprocess.CalledProcessError(1, "inference")), \
                patch.object(sys, "argv", ["quickstart.py"]):
            with self.assertRaises(subprocess.CalledProcessError):
                quickstart.main()


if __name__ == "__main__":
    unittest.main()
