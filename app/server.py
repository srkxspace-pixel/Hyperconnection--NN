"""Live image-prediction page for the demonstration (D20): the "Image Prediction" screen of the project.

    python app/server.py                        # then open http://127.0.0.1:8000 (it opens by itself)
    python app/server.py --runs results/demo --port 8000 --device cpu

Serves app/index.html and a small JSON API on this computer only (127.0.0.1). A "model" is a run folder in --runs that holds
the model.pt written by a training run with train.save_weights=true (the four networks of the repository are in results/demo/;
README_DEMO.md, "Get the demo weights", re-creates them). The class names and sample pictures are bundled in app/ (classes.json,
classes.png), so no dataset is needed. PyTorch is imported when the first model is loaded, so the page opens without it. The
prediction history lives in the browser, not here.
"""
import argparse
import base64
import csv
import io
import json
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                  # so `python app/server.py` works without `pip install -e .`
MAX_UPLOAD = 10 * 2**20                                # bytes
REPORT = ROOT / "docs" / "PROJECT_REPORT.docx"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def discover(runs_dir: Path) -> list:
    """One record per run folder that holds model.pt, config.yaml and final.json (connection order: standard first)."""
    out = []
    for f in sorted(Path(runs_dir).glob("*/model.pt")):
        try:
            cfg = yaml.safe_load((f.parent / "config.yaml").read_text(encoding="utf-8"))
            fin = json.loads((f.parent / "final.json").read_text(encoding="utf-8"))
            out.append({"id": f.parent.name, "family": cfg["model"]["family"], "depth": cfg["model"]["depth"],
                        "kind": cfg["connection"]["kind"], "n": cfg["connection"]["n"], "dataset": cfg["data"]["name"],
                        "percent": cfg["data"]["percent"], "seed": cfg["seed"], "epochs": fin["epochs_done"],
                        "acc": fin["final_acc"], "params": fin["params"]})
        except (OSError, KeyError, ValueError, yaml.YAMLError):
            continue                                   # an unfinished or foreign folder: not a model
    return sorted(out, key=lambda r: (r["dataset"], r["family"], r["depth"], r["n"], r["kind"]))


def class_info() -> dict:
    """The 100 class names, their superclass names and one sample picture each (the first CIFAR-100 test image of the class),
    bundled in app/ by scripts/make_app_assets.py."""
    info = json.loads((ROOT / "app" / "classes.json").read_text(encoding="utf-8"))
    return {"list": [{"name": n, "super": s} for n, s in zip(info["names"], info["superclass"])],
            "sprite": Image.open(ROOT / "app" / "classes.png").convert("RGB")}


def sample_png(info: dict, idx: int) -> bytes:
    """A 64 x 64 PNG of the sample picture of class `idx`, cut out of the 10 x 10 sprite (32 x 32 each)."""
    x, y = 32 * (idx % 10), 32 * (idx // 10)
    buf = io.BytesIO()
    info["sprite"].crop((x, y, x + 32, y + 32)).resize((64, 64), Image.Resampling.BICUBIC).save(buf, "PNG")
    return buf.getvalue()


def curve(run_dir: Path) -> dict:
    """metrics.csv of a run as columns: epoch, train_loss, train_acc, test_loss, test_acc."""
    with (Path(run_dir) / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc")}


def figures() -> list:
    return sorted(p.name for p in (ROOT / "results" / "figures").glob("F*.png"))


def png_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_model(run_dir: Path, device: str) -> dict:
    from hcsmall.predict import load_run               # imports PyTorch, so only when a model is first needed
    return load_run(run_dir, device)


def run_model(run: dict, image: Image.Image) -> tuple:
    from hcsmall.predict import classify
    return classify(run, image)                        # (probabilities, the picture the network saw)


def model_for(srv, rec: dict) -> dict:
    with srv.lock:                                     # one load at a time; a loaded model stays in memory
        if rec["id"] not in srv.models:
            srv.models[rec["id"]] = load_model(srv.runs_dir / rec["id"], srv.device)
        return srv.models[rec["id"]]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                      # the console shows one line per prediction instead
        pass

    def reply(self, status: int, body: bytes, ctype: str, headers: dict = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {"Cache-Control": "no-store", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status: int = 200) -> None:
        self.reply(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        s, path = self.server, urlparse(self.path).path
        tail = path.rsplit("/", 1)[-1]
        if path in ("/", "/index.html"):
            return self.reply(200, (ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/init":
            return self.send_json({"runs": s.runs, "runs_dir": s.runs_rel, "classes": s.classes["list"],
                                   "figures": figures(), "report": REPORT.is_file()})
        if path.startswith("/api/sample/") and tail.endswith(".png") and tail[:-4].isdigit() and int(tail[:-4]) < len(s.classes["list"]):
            return self.reply(200, sample_png(s.classes, int(tail[:-4])), "image/png", {"Cache-Control": "max-age=3600"})
        if path.startswith("/api/curve/") and any(r["id"] == tail for r in s.runs):
            return self.send_json(curve(s.runs_dir / tail))
        if path.startswith("/files/figures/") and tail in figures():
            return self.reply(200, (ROOT / "results" / "figures" / tail).read_bytes(), "image/png")
        if path == "/files/report" and REPORT.is_file():
            return self.reply(200, REPORT.read_bytes(), DOCX, {"Content-Disposition": 'attachment; filename="PROJECT_REPORT.docx"'})
        self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        s, u = self.server, urlparse(self.path)
        rec = next((r for r in s.runs if r["id"] == parse_qs(u.query).get("run", [""])[0]), None)
        if rec is None or u.path not in ("/api/load", "/api/predict"):
            return self.send_json({"error": "unknown model or address"}, 400)
        try:
            if u.path == "/api/load":
                model_for(s, rec)
                return self.send_json({"ok": True, **rec})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self.send_json({"error": "no image received"}, 400)
            if n > MAX_UPLOAD:
                return self.send_json({"error": f"the image is larger than {MAX_UPLOAD // 2**20} MB"}, 413)
            try:
                image = Image.open(io.BytesIO(self.rfile.read(n)))
                image.load()
            except Exception:
                return self.send_json({"error": "that file is not an image this page can read (use JPG or PNG)"}, 400)
            t0 = time.perf_counter()
            probs, seen = run_model(model_for(s, rec), image)
            ms = (time.perf_counter() - t0) * 1000
            names = [c["name"] for c in s.classes["list"]]
            top = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:5]
            print(f"{rec['id']}: {names[top[0]]} {probs[top[0]]:.1%} ({ms:.0f} ms)", flush=True)
            self.send_json({"run": rec["id"], "ms": round(ms, 1), "seen": png_url(seen), "probs": probs,
                            "top5": [{"index": i, "name": names[i], "prob": probs[i]} for i in top]})
        except Exception as e:                         # show the reason in the page, keep the details in the console
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"              # on Windows this lets a second server share a busy port and answer in its place


def make_server(runs_dir: Path, port: int = 8000, device: str = "cpu") -> Server:
    srv = Server(("127.0.0.1", port), Handler)
    srv.runs_dir, srv.device, srv.models, srv.lock = Path(runs_dir), device, {}, threading.Lock()
    srv.runs, srv.classes = discover(srv.runs_dir), class_info()
    try:
        srv.runs_rel = str(srv.runs_dir.relative_to(ROOT))
    except ValueError:
        srv.runs_rel = str(srv.runs_dir)
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/demo", help="folder of run folders that hold a model.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu", help="cpu (default; plenty for one 32 x 32 image) or cuda")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    try:
        srv = make_server(ROOT / a.runs, a.port, a.device)
    except FileNotFoundError as e:                     # before OSError: it is a subclass
        sys.exit(f"A file the page needs is missing ({e.filename}). Is the repository complete (app/classes.json, app/classes.png)?")
    except OSError as e:
        sys.exit(f"Cannot start: {e}. If the port is busy, try --port {a.port + 1}.")
    url = f"http://127.0.0.1:{a.port}"
    print(f"Image Prediction page: {url}   (Ctrl+C stops it)")
    print(f"Models found in {srv.runs_rel}: {len(srv.runs)}" + ("" if srv.runs else '  -> see README_DEMO.md, "Get the demo weights"'), flush=True)
    if not a.no_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
