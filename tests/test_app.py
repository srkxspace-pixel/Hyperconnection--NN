"""The demo page's server (app/server.py) without PyTorch: model discovery, class info and the HTTP API with a stand-in model.

Runs without PyTorch:  python tests/test_app.py   (or through pytest). The real model code is covered by tests/test_predict.py.
"""
import contextlib
import http.client
import importlib.util
import io
import json
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "app" / "server.py"
if not SERVER.exists():                                       # the code snapshot that runs on Kaggle has no app/ folder
    import pytest
    pytest.skip("app/ is not part of this checkout", allow_module_level=True)

import numpy as np
from PIL import Image

_spec = importlib.util.spec_from_file_location("nn09_app_server", SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def make_run(base: Path, name: str, kind: str = "dhc", n: int = 2, acc: float = 0.667) -> None:
    d = base / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(f"seed: 0\nmodel: {{family: resnet, depth: 20}}\nconnection: {{kind: {kind}, n: {n}}}\n"
                                   "data: {name: cifar100, percent: 100}\n", encoding="utf-8")
    (d / "final.json").write_text(json.dumps({"epochs_done": 2, "final_acc": acc, "params": 279374}), encoding="utf-8")
    (d / "metrics.csv").write_text("epoch,step,train_loss,train_acc,test_loss,test_acc,lr\n"
                                   "1,10,4.0,0.1,3.9,0.12,0.1\n2,20,3.5,0.2,3.4,0.22,0.05\n", encoding="utf-8")
    (d / "model.pt").write_bytes(b"x")


def fake_run(run, image):                                     # same picture -> same answer; sums to 1
    logits = np.random.default_rng(abs(hash(image.tobytes())) % 2**32).normal(size=100) * 3
    p = np.exp(logits - logits.max())
    return (p / p.sum()).tolist(), image.convert("RGB").resize((32, 32))


@contextlib.contextmanager
def served(tmp_path):
    """The real server on a free port, with a stand-in for the PyTorch parts and two fake run folders."""
    runs = Path(tmp_path) / "demo"
    make_run(runs, "resnet20_residual_n1_cifar100p100_s0", "residual", 1, 0.65)
    make_run(runs, "resnet20_dhc_n2_cifar100p100_s0")
    (runs / "unfinished").mkdir()                             # no model.pt: not a model
    names = ("load_model", "run_model")
    saved = {k: getattr(server, k) for k in names}
    server.load_model, server.run_model = lambda d, dev: {"dir": str(d)}, fake_run
    srv = server.make_server(runs, port=0)                    # the class names and sample pictures are the real bundled ones
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        for k, v in saved.items():
            setattr(server, k, v)


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(url, body=b""):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST")) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def png(size=(50, 80), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_discover_lists_finished_runs_only(tmp_path):
    make_run(tmp_path, "b_dhc", "dhc", 2)
    make_run(tmp_path, "a_residual", "residual", 1)
    (tmp_path / "no_weights").mkdir()
    (tmp_path / "no_final").mkdir()
    (tmp_path / "no_final" / "model.pt").write_bytes(b"x")
    got = server.discover(tmp_path)
    assert [r["id"] for r in got] == ["a_residual", "b_dhc"]                         # standard shortcut first
    assert got[1]["n"] == 2 and got[1]["acc"] == 0.667 and got[1]["epochs"] == 2 and got[1]["family"] == "resnet"


def test_the_api_round_trip(tmp_path):
    with served(tmp_path) as (srv, url):
        init = json.loads(get(url + "/api/init")[1])
        assert [r["kind"] for r in init["runs"]] == ["residual", "dhc"] and len(init["classes"]) == 100
        rid = init["runs"][1]["id"]
        assert post(f"{url}/api/load?run={rid}")[1]["ok"]
        status, out = post(f"{url}/api/predict?run={rid}", png())
        p = [c["prob"] for c in out["top5"]]
        assert status == 200 and len(p) == 5 and p == sorted(p, reverse=True) and len(out["probs"]) == 100
        names = [c["name"] for c in init["classes"]]
        assert abs(sum(out["probs"]) - 1) < 1e-6 and p[0] == max(out["probs"]) and all(c["name"] == names[c["index"]] for c in out["top5"])
        assert out["seen"].startswith("data:image/png;base64,") and out["run"] == rid
        assert post(f"{url}/api/predict?run={rid}", png())[1]["top5"] == out["top5"]  # same picture, same model: same answer
        assert get(url + "/api/sample/7.png")[1][:8] == b"\x89PNG\r\n\x1a\n"
        assert json.loads(get(f"{url}/api/curve/{rid}")[1])["test_acc"] == [0.12, 0.22]
        assert get(url + "/")[0] == 200


def test_bad_requests_are_refused_with_a_reason(tmp_path):
    with served(tmp_path) as (srv, url):
        rid = srv.runs[0]["id"]
        assert "not an image" in post(f"{url}/api/predict?run={rid}", b"this is not an image")[1]["error"]
        assert post(f"{url}/api/predict?run=nope", png())[0] == 400
        assert post(f"{url}/api/predict?run={rid}", b"")[0] == 400
        assert get(url + "/api/sample/100.png")[0] == 404 and get(url + "/api/curve/nope")[0] == 404
        assert get(url + "/files/figures/../../README.md")[0] == 404                 # only the listed figures are served
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        c.putrequest("POST", f"/api/predict?run={rid}")
        c.putheader("Content-Length", str(server.MAX_UPLOAD + 1))
        c.endheaders()
        assert c.getresponse().status == 413                                         # refused before the body is read
        c.close()


def test_bundled_class_info_needs_no_dataset():
    info = server.class_info()
    names = [c["name"] for c in info["list"]]
    assert len(names) == 100 and names[0] == "apple" and names[34] == "fox" and all(c["super"] for c in info["list"])
    assert "dog" not in names and "cat" not in names                                 # those are CIFAR-10 names
    assert info["sprite"].size == (320, 320)
    sample = Image.open(io.BytesIO(server.sample_png(info, 5)))
    assert sample.size == (64, 64) and sample.getextrema() != ((0, 0), (0, 0), (0, 0))   # a real picture, not a blank cell


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                sub = Path(d) / name
                sub.mkdir()
                fn(sub) if fn.__code__.co_argcount else fn()
                print("ok  ", name)
