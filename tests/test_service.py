"""Offline contract and image regression checks: python -m unittest discover -s tests -v."""
import hashlib
import io
import asyncio
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

import hd_pipeline
import main


class IdentityModel:
    device = "cpu"
    epoch = 19
    model_version = "test-model"
    tile_size = 384

    def enhance(self, image):
        from PIL import ImageOps
        return ImageOps.exif_transpose(image).convert("RGB")


def image_bytes(fmt="PNG", size=(32, 24), mode="RGB", exif=None):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format=fmt, **({"exif": exif} if exif else {}))
    return buf.getvalue()


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(main, "model_instance", IdentityModel()),
                        patch.object(main, "API_KEY", "test-secret", create=True)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def post(self, data=None, path="/enhance", **kwargs):
        headers = {"X-API-Key": "test-secret", **kwargs.pop("headers", {})}
        return self.client.post(path, headers=headers,
                                files={"file": ("input", image_bytes() if data is None else data)}, **kwargs)

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()["error"], code)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_liveness_and_readiness(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 200)
        with patch.object(main, "model_instance", None):
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/ready").status_code, 503)
            self.assert_error(self.post(), 503, "model_unavailable")

    def test_authentication_fails_closed(self):
        self.assert_error(self.client.post("/enhance"), 401, "unauthorized")
        for key in ("", "wrong", "é"):
            # HTTP header values use Latin-1 in the transport.
            self.assert_error(self.post(headers={"X-API-Key": key.encode("utf-8")}), 401, "unauthorized")
        with patch.object(main, "API_KEY", ""):
            self.assertEqual(self.client.get("/ready").status_code, 503)
            self.assert_error(self.post(), 503, "service_unconfigured")

    def test_binary_output_and_versioned_alias(self):
        for fmt, path, mode in (("JPEG", "/enhance", "RGB"), ("PNG", "/v1/enhance", "RGBA"),
                                ("PNG", "/enhance", "L")):
            with self.subTest(fmt=fmt, path=path, mode=mode):
                response = self.post(image_bytes(fmt, mode=mode), path)
                self.assertEqual(response.status_code, 200, response.text[:100])
                self.assertEqual(response.headers["content-type"], "image/jpeg" if fmt == "JPEG" else "image/png")
                with Image.open(io.BytesIO(response.content)) as output:
                    self.assertEqual(output.size, (32, 24))
                    output.load()
                self.assertEqual(response.headers["X-Model-Version"], "test-model")
                self.assertGreaterEqual(float(response.headers["X-Inference-Ms"]), 0)
                self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_validation_and_checksums(self):
        self.assert_error(self.client.post("/enhance", headers={"X-API-Key": "test-secret"}), 400, "missing_file")
        for data, code in ((b"", "empty_file"), (b"invalid", "invalid_image"),
                           (image_bytes("GIF"), "unsupported_format")):
            self.assert_error(self.post(data), 400, code)
        self.assert_error(self.post(headers={"X-Model-Input-SHA256": "wrong"}), 400, "checksum_mismatch")
        self.assert_error(self.post(headers={"X-Model-Input-SHA256": "é".encode()}), 400, "checksum_mismatch")
        checksum = hashlib.sha256(image_bytes()).hexdigest()
        for header in ("X-Model-Input-SHA256", "X-SHA256-Checksum"):
            self.assertEqual(self.post(headers={header: checksum.upper()}).status_code, 200)

    def test_file_and_pixel_limits(self):
        with patch.object(main, "MAX_FILE_SIZE", 32):
            self.assert_error(self.post(b"x" * 33), 413, "file_too_large")
        with patch.object(main, "MAX_IMAGE_PIXELS", 10, create=True):
            self.assert_error(self.post(), 413, "image_too_large")
        with patch.object(main, "MAX_IMAGE_DIMENSION", 10, create=True):
            self.assert_error(self.post(), 413, "image_too_large")
        with patch.object(Image, "MAX_IMAGE_PIXELS", 1):
            self.assert_error(self.post(), 413, "image_too_large")

    def test_truncated_and_animated_images(self):
        self.assert_error(self.post(image_bytes()[:60]), 400, "invalid_image")
        output = io.BytesIO()
        with Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second:
            first.save(output, format="PNG", save_all=True, append_images=[second])
        self.assert_error(self.post(output.getvalue()), 400, "unsupported_format")

    def test_worker_remains_exclusive_when_request_slot_is_free(self):
        self.assertFalse(main.request_slot.locked())
        main.inference_slot.acquire()
        try:
            self.assert_error(self.post(), 503, "service_busy")
        finally:
            main.inference_slot.release()
        self.assertEqual(self.post().status_code, 200)

    def test_exif_orientation(self):
        exif = Image.Exif()
        exif[274] = 6
        response = self.post(image_bytes("JPEG", exif=exif))
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(response.content)) as output:
            self.assertEqual(output.size, (24, 32))

    def test_inference_errors_do_not_expose_exceptions(self):
        with self.assertLogs("aquavision", level="ERROR"), patch.object(main.model_instance, "enhance", side_effect=RuntimeError("private exception")):
            response = self.post()
            self.assert_error(response, 500, "inference_failed")
            self.assertNotIn("private exception", response.text)
        with self.assertLogs("aquavision", level="ERROR"), patch.object(main.model_instance, "enhance", side_effect=MemoryError()):
            self.assert_error(self.post(), 503, "compute_unavailable")
        self.assertEqual(self.post().status_code, 200)

    def test_health_remains_responsive_and_busy_request_is_rejected(self):
        started, release = threading.Event(), threading.Event()
        original = main.model_instance.enhance

        def blocking(image):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return original(image)

        with patch.object(main.model_instance, "enhance", side_effect=blocking), ThreadPoolExecutor(1) as executor:
            pending = executor.submit(self.post)
            try:
                self.assertTrue(started.wait(2))
                before = time.monotonic()
                self.assertEqual(self.client.get("/ready").status_code, 200)
                self.assertLess(time.monotonic() - before, 1)
                self.assert_error(self.post(), 503, "service_busy")
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=2).status_code, 200)

    def test_info_and_openapi_contract(self):
        info = self.client.get("/info").json()
        self.assertEqual(info["max_image_pixels"], main.MAX_IMAGE_PIXELS)
        self.assertEqual(info["model_version"], "test-model")
        schema = self.client.get("/openapi.json").json()
        for path in ("/enhance", "/v1/enhance"):
            operation = schema["paths"][path]["post"]
            self.assertIn("multipart/form-data", operation["requestBody"]["content"])
            self.assertIn("image/png", operation["responses"]["200"]["content"])
            self.assertTrue(operation["security"])

    def test_streaming_body_limit_without_content_length(self):
        async def run():
            chunks = iter([b"--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n\r\n",
                           b"x" * 66000])
            messages = []

            async def receive():
                return {"type": "http.request", "body": next(chunks), "more_body": True}

            async def send(message):
                messages.append(message)

            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                     "scheme": "http", "path": "/enhance", "raw_path": b"/enhance", "query_string": b"",
                     "headers": [(b"x-api-key", b"test-secret"),
                                 (b"content-type", b"multipart/form-data; boundary=boundary")],
                     "client": ("127.0.0.1", 1234), "server": ("test", 80)}
            await main.app(scope, receive, send)
            return messages

        with patch.object(main, "MAX_FILE_SIZE", 32):
            messages = asyncio.run(run())
        self.assertEqual(messages[0]["status"], 413)
        self.assertFalse(main.request_slot.locked())

    def test_startup_failure_is_visible_to_readiness(self):
        with patch.object(main, "AquaVisionModel", side_effect=RuntimeError("missing checkpoint")), self.assertLogs("aquavision", level="ERROR"):
            with TestClient(main.app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 503)


class TilingRegressionTests(unittest.TestCase):
    def test_bounded_tiling_matches_original_feathering(self):
        runner = getattr(hd_pipeline, "run_tiled_image", None)
        self.assertTrue(callable(runner), "PIL tiling must avoid full-image float arrays")
        rng = np.random.default_rng(42)
        # Awkward edges, narrow images, exact tile sizes and multiple rows.
        for width, height in ((1, 1), (8, 9), (384, 384), (385, 7), (17, 801), (800, 600), (641, 965)):
            with self.subTest(size=(width, height)):
                source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                shapes = []

                def predict(tile):
                    shapes.append(tile.shape)
                    # Position-dependent predictions expose blending errors at tile borders.
                    ramp = np.linspace(0, .1, tile.shape[1], dtype=np.float32)[None, :, None]
                    return np.clip(tile * .8 + ramp, 0, 1)

                expected = hd_pipeline.to_uint8(hd_pipeline.run_tiled(predict, source.astype(np.float32) / 255,
                                                                     tile=384, overlap=64))
                actual = np.asarray(runner(predict, Image.fromarray(source), tile=384, overlap=64))
                self.assertEqual(actual.shape, source.shape)
                self.assertLessEqual(np.abs(actual.astype(int) - expected.astype(int)).max(), 1)
                self.assertTrue(all(max(shape[:2]) <= 384 for shape in shapes))

    def test_low_memory_profile_uses_the_same_feathering(self):
        source = np.random.default_rng(2).integers(0, 256, (413, 321, 3), dtype=np.uint8)
        predictor = lambda tile: np.clip(tile * .7 + .1, 0, 1)
        expected = hd_pipeline.to_uint8(hd_pipeline.run_tiled(predictor, source.astype(np.float32) / 255,
                                                              tile=192, overlap=64))
        output = hd_pipeline.run_tiled_image(predictor, Image.fromarray(source), tile=192, overlap=64)
        np.testing.assert_array_equal(np.asarray(output), expected)


class ModelSmokeTests(unittest.TestCase):
    def test_weights_export_and_real_cpu_inference(self):
        import torch
        from export_weights import export_weights
        from inference import AquaVisionModel, CHECKPOINT_PATH

        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "inference.pt"
            export_weights(CHECKPOINT_PATH, weights)
            artifact = torch.load(weights, weights_only=True, map_location="cpu")
            self.assertEqual(set(artifact), {"model_state_dict", "epoch", "source_sha256"})
            self.assertLess(weights.stat().st_size, CHECKPOINT_PATH.stat().st_size / 2)
            original = AquaVisionModel(CHECKPOINT_PATH, device="cpu")
            model = AquaVisionModel(weights, device="cpu")
            self.assertEqual(model.model_version, original.model_version)
            self.assertEqual(sum(p.numel() for p in model.model.parameters()), 1928483)
            source = np.random.default_rng(1).integers(0, 256, (17, 385, 3), dtype=np.uint8)
            expected = hd_pipeline.to_uint8(hd_pipeline.run_tiled(original._predict_np,
                source.astype(np.float32) / 255, tile=384, overlap=64))
            with model.enhance(Image.fromarray(source)) as output:
                np.testing.assert_array_equal(np.asarray(output), expected)
            for size in ((1, 1), (7, 9), (33, 35)):
                with Image.new("RGB", size) as source_image, model.enhance(source_image) as output:
                    self.assertEqual(output.size, size)
            with patch.dict("os.environ", {"INFERENCE_TILE_SIZE": "192"}):
                small_tiles = AquaVisionModel(weights, device="cpu")
            self.assertEqual(small_tiles.tile_size, 192)
            self.assertNotEqual(small_tiles.model_version, model.model_version)
            with patch.object(main, "API_KEY", "integration-key"), patch.dict("os.environ", {
                "MODEL_PATH": str(weights), "AQUAVISION_DEVICE": "cpu", "INFERENCE_TILE_SIZE": "192"}):
                with TestClient(main.app) as client:
                    self.assertEqual(client.get("/ready").status_code, 200)
                    for fmt in ("JPEG", "PNG"):
                        response = client.post("/v1/enhance", headers={"X-API-Key": "integration-key"},
                                               files={"file": ("image", image_bytes(fmt))})
                        self.assertEqual(response.status_code, 200)
                        with Image.open(io.BytesIO(response.content)) as image:
                            self.assertEqual(image.size, (32, 24))
                            image.load()


if __name__ == "__main__":
    unittest.main()
