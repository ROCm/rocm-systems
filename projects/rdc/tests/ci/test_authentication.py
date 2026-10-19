"""Checks of the certificate authority settings in authentication/openssl.cnf.

Runs the openssl commands of 01gen_root_cert.sh and 02gen_ssl_artifacts.sh in a
temporary directory, answering the signing prompts like the readme says. Needs only
the openssl command line tool (no GPU, no daemon).
"""

import os
import shutil
import subprocess

import pytest

CNF = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "authentication", "openssl.cnf")
)
# Must match the CA, see [ policy_match ] in openssl.cnf.
SUBJECT = "/O=MyCompany/OU=MyCompanyUnit/L=MyCity/ST=MyStateProvince/C=MC"
# A request that asks to become a signing certificate.
SIGNING_REQUEST_CNF = """\
[ req ]
string_mask = nombstr
distinguished_name = dn
req_extensions = ext
[ dn ]
[ ext ]
basicConstraints = critical, CA:TRUE
keyUsage = critical, keyCertSign, digitalSignature
"""

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not found")


def _openssl(cwd, *args, answers=None):
    return subprocess.run(
        ["openssl", *args],
        cwd=cwd,
        input=answers,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )


@pytest.fixture(scope="module")
def ca_dir(tmp_path_factory):
    """A CA directory set up the way 01gen_root_cert.sh sets it up."""
    root = tmp_path_factory.mktemp("rdc_ca")
    shutil.copy(CNF, root / "openssl.cnf")
    (root / "signing_request.cnf").write_text(SIGNING_REQUEST_CNF)
    ca = root / "CA"
    (ca / "private").mkdir(parents=True)
    (ca / "newcerts").mkdir()
    (ca / "serial").write_text("01\n")
    (ca / "index.txt").write_text("")
    _openssl(
        ca, "req", "-new", "-x509", "-nodes", "-newkey", "rsa:2048", "-extensions", "v3_ca",
        "-keyout", "private/rdc_cakey.pem", "-out", "rdc_cacert.pem", "-days", "30",
        "-config", "../openssl.cnf", "-subj", SUBJECT + "/CN=rdc-test-root",
    )  # fmt: skip
    return ca


def _issue(ca, name, request_cnf):
    """Create a request with request_cnf and sign it like 02gen_ssl_artifacts.sh."""
    _openssl(
        ca, "req", "-new", "-nodes", "-newkey", "rsa:2048", "-keyout", name + ".key",
        "-out", name + "_csr.pem", "-config", request_cnf, "-subj", SUBJECT + "/CN=" + name,
    )  # fmt: skip
    signing = _openssl(
        ca, "ca", "-out", name + ".pem", "-config", "../openssl.cnf", "-infiles", name + "_csr.pem",
        answers="y\ny\n",
    )  # fmt: skip
    cert = _openssl(ca, "x509", "-in", name + ".pem", "-noout", "-text").stdout
    return cert, signing.stdout + signing.stderr


def test_request_cannot_make_itself_a_signing_certificate(ca_dir):
    cert, _ = _issue(ca_dir, "asks-for-ca", "../signing_request.cnf")
    assert "CA:FALSE" in cert
    assert "CA:TRUE" not in cert
    assert "Certificate Sign" not in cert


def test_shipped_request_keeps_its_alternative_names(ca_dir):
    cert, _ = _issue(ca_dir, "shipped", "../openssl.cnf")
    assert "CA:FALSE" in cert
    assert "DNS:localhost" in cert


def test_issued_certificate_is_for_tls_servers_and_clients(ca_dir):
    cert, _ = _issue(ca_dir, "tls", "../openssl.cnf")
    assert "TLS Web Server Authentication" in cert
    assert "TLS Web Client Authentication" in cert


def test_signing_prompt_shows_the_extensions(ca_dir):
    _, prompt = _issue(ca_dir, "prompted", "../openssl.cnf")
    assert "X509v3 Subject Alternative Name" in prompt
    assert "Sign the certificate?" in prompt
