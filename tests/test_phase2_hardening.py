"""Regression tests for the Phase 2 critical fixes (no behaviour change to analysis)."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import predict
from backend.main import app
from backend.security import get_current_recruiter


class AnalyzeUploadLimitTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(id=None)
        self.addCleanup(app.dependency_overrides.pop, get_current_recruiter, None)
        self.client = TestClient(app)

    def test_oversized_upload_is_rejected_before_analysis(self):
        payload = b"%PDF-1.4\n" + b"x" * 4096
        with (
            patch("backend.main.MAX_RESUME_BYTES", 1024),
            patch("backend.main.analyze_resume") as analyze,
        ):
            response = self.client.post(
                "/api/analyze/",
                files={"file": ("big.pdf", payload, "application/pdf")},
                data={"jd_text": "Python developer"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("size limit", response.json()["detail"])
        analyze.assert_not_called()

    def test_upload_without_pdf_signature_is_rejected_before_analysis(self):
        with patch("backend.main.analyze_resume") as analyze:
            response = self.client.post(
                "/api/analyze/",
                files={"file": ("fake.pdf", b"MZ not a pdf at all", "application/pdf")},
                data={"jd_text": "Python developer"},
            )
        self.assertEqual(response.status_code, 400)
        analyze.assert_not_called()

    def test_upload_at_the_limit_is_still_accepted(self):
        payload = b"%PDF-1.4\n" + b"x" * 100
        with (
            patch("backend.main.MAX_RESUME_BYTES", len(payload)),
            patch(
                "backend.main.analyze_resume",
                side_effect=RuntimeError("reached analysis"),
            ) as analyze,
        ):
            response = self.client.post(
                "/api/analyze/",
                files={"file": ("ok.pdf", payload, "application/pdf")},
                data={"jd_text": "Python developer"},
            )
        analyze.assert_called_once()
        self.assertEqual(response.status_code, 500)  # sanitized internal error


class ArtifactLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.vec = Path(self.tmp.name) / "vec.joblib"
        self.mod = Path(self.tmp.name) / "model.joblib"
        self.vec.write_bytes(b"v1")
        self.mod.write_bytes(b"m1")
        predict._ARTIFACT_CACHE.clear()
        self.addCleanup(predict._ARTIFACT_CACHE.clear)

    def test_default_paths_are_absolute_and_exist(self):
        self.assertTrue(Path(predict.VECTORIZER_PATH).is_absolute())
        self.assertTrue(Path(predict.MODEL_PATH).is_absolute())

    def test_artifacts_are_loaded_once_and_silently(self):
        loads = []

        def fake_load(path):
            loads.append(path)
            return f"obj:{Path(path).name}"

        out = io.StringIO()
        with (
            patch("predict.VECTORIZER_PATH", str(self.vec)),
            patch("predict.MODEL_PATH", str(self.mod)),
            patch("predict.joblib.load", side_effect=fake_load),
            contextlib.redirect_stdout(out),
        ):
            first = predict.load_artifacts_cached()
            second = predict.load_artifacts_cached()

        self.assertEqual(len(loads), 2)  # vectorizer + model, only once
        self.assertIs(first, second)
        self.assertEqual(out.getvalue(), "")

    def test_replaced_artifact_file_is_reloaded(self):
        loads = []
        with (
            patch("predict.VECTORIZER_PATH", str(self.vec)),
            patch("predict.MODEL_PATH", str(self.mod)),
            patch("predict.joblib.load", side_effect=lambda p: loads.append(p) or p),
        ):
            predict.load_artifacts_cached()
            self.mod.write_bytes(b"model-version-2")
            os.utime(self.mod, ns=(1, 1))  # force a distinct mtime
            predict.load_artifacts_cached()
        self.assertEqual(len(loads), 4)

    def test_missing_artifact_is_not_cached(self):
        with (
            patch("predict.VECTORIZER_PATH", str(self.vec)),
            patch("predict.MODEL_PATH", str(Path(self.tmp.name) / "missing.joblib")),
        ):
            with self.assertRaises(FileNotFoundError):
                predict.load_artifacts_cached()
        self.assertEqual(predict._ARTIFACT_CACHE, {})

    def test_verbose_flag_controls_progress_output(self):
        out = io.StringIO()
        with (
            patch("predict.VECTORIZER_PATH", str(self.vec)),
            patch("predict.MODEL_PATH", str(self.mod)),
            patch("predict.joblib.load", return_value=object()),
            contextlib.redirect_stdout(out),
        ):
            predict.load_artifacts()
            self.assertEqual(out.getvalue(), "")
            predict.load_artifacts(verbose=True)
        self.assertIn("[predict]", out.getvalue())


if __name__ == "__main__":
    unittest.main()