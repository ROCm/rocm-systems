#!/usr/bin/env python3
"""The base digest comes from the registry, without pulling the image."""

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_madengine as tm  # noqa: E402

DIGEST = "sha256:" + "a" * 64


class Response(io.BytesIO):
    def __init__(self, headers: dict | None = None, body: bytes = b""):
        super().__init__(body)
        self.headers = headers or {}


def unauthorized(scope: str) -> urllib.error.HTTPError:
    challenge = (f'Bearer realm="https://auth.docker.io/token",'
                 f'service="registry.docker.io",scope="{scope}"')
    return urllib.error.HTTPError(
        "url", 401, "Unauthorized", {"WWW-Authenticate": challenge}, None)


class RegistryDigestTest(unittest.TestCase):
    def test_docker_hub_token_flow(self):
        calls = []

        def urlopen(req, timeout):
            calls.append(req)
            if isinstance(req, str):
                return Response(body=json.dumps({"token": "t"}).encode())
            if "Authorization" not in req.headers:
                raise unauthorized("repository:rocm/primus:pull")
            return Response({"Docker-Content-Digest": DIGEST})

        with mock.patch.object(tm.urllib.request, "urlopen", urlopen):
            self.assertEqual(tm.registry_digest("rocm/primus:v26.4"),
                             f"rocm/primus@{DIGEST}")
        head = calls[0]
        self.assertEqual(head.get_method(), "HEAD")
        self.assertEqual(head.full_url,
                         "https://registry-1.docker.io/v2/rocm/primus/manifests/v26.4")
        self.assertIn("manifest.list.v2+json", head.headers["Accept"])
        self.assertIn("scope=repository%3Arocm%2Fprimus%3Apull", calls[1])
        self.assertEqual(calls[2].headers["Authorization"], "Bearer t")

    def test_reference_forms(self):
        urls = []

        def urlopen(req, timeout):
            urls.append(req.full_url)
            return Response({"Docker-Content-Digest": DIGEST})

        with mock.patch.object(tm.urllib.request, "urlopen", urlopen):
            self.assertEqual(tm.registry_digest("ubuntu"), f"ubuntu@{DIGEST}")
            tm.registry_digest("ghcr.io/rocm/x:1")
            tm.registry_digest("localhost:5000/x")
        self.assertEqual(urls, [
            "https://registry-1.docker.io/v2/library/ubuntu/manifests/latest",
            "https://ghcr.io/v2/rocm/x/manifests/1",
            "https://localhost:5000/v2/x/manifests/latest",
        ])
        self.assertEqual(tm.registry_digest(f"rocm/primus@{DIGEST}"),
                         f"rocm/primus@{DIGEST}")

    def test_failure_refuses_mutable_tag(self):
        def urlopen(req, timeout):
            raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        with mock.patch.object(tm.urllib.request, "urlopen", urlopen), \
                mock.patch.object(tm.subprocess, "run") as run:
            self.assertEqual(tm.registry_digest("rocm/primus:gone"), "")
            with self.assertRaisesRegex(RuntimeError, "immutable registry digest"):
                tm.resolve_base_digest("rocm/primus:gone")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
