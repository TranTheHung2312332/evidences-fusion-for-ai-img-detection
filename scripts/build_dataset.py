from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import tarfile
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEED = 42
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "avif", "heic"}

FIVEK_META_URLS = [
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/training.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/validation.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/testing.json",
]
FODB_OFFICIAL_PAGE = "https://faui1-files.cs.fau.de/public/mmsec/datasets/fodb/"
FODB_PARTS = ["part01.zip", "part02.zip", "part03.zip"]
FODB_MEMBER_RE = re.compile(
    r"^(?P<device_dir>D(?P<device>\d{2})_[^/]+)/orig/"
    r"D(?P=device)_img_orig_(?P<scene>\d{4})\.jpg$",
    re.I,
)
UNSPLASH_HF_REPO = "1aurent/unsplash-lite"
UNSPLASH_HF_REVISION = "main"

AIGEN_REPO = "pthan12/AIGenImages2026"
AIGEN_REVISION = "073e1924d9d0d85ac97a53b07947b6ac95ce241c"
AIGEN_ARCHIVE_FILENAME = "aigenimages2026.tar.gz"
AIGEN_ARCHIVE_SHA256 = "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498"
AIGEN_ARCHIVE_BYTES = 11138511098

T2I_PROMPT_REPO = "lioooox/T2I-CoReBench"
T2I_IMAGE_REPO = "lioooox/T2I-CoReBench-Images"
T2I_PROMPT_REVISION = "3aed46abac251c5ecf43488203c5c65d7af863a3"
T2I_IMAGE_REVISION = "2350e0f664d849626df392cb93954b66019c18ee"
T2I_DIMENSIONS = [
    "C-MI", "C-MA", "C-MR", "C-TR",
    "R-LR", "R-BR", "R-HR", "R-PR",
    "R-GR", "R-AR", "R-CR", "R-RR",
]
T2I_ARCHIVE_MAP = {
    "FLUX.1-Krea-dev": "FLUX.1-Krea-dev.tar.gz",
    "Qwen-Image": "Qwen-Image.tar.gz",
    "Janus-Pro-7B": "Janus-Pro-7B.tar.gz",
    "GoT-R1-7B": "GoT-R1-7B.tar.gz",
    "Qwen-Image-2512": "Qwen-Image-2512.tar.gz",
    "HunyuanImage-3.0": "HunyuanImage-3.0.tar.gz",
    "Z-Image": "Z-Image.tar.gz",
    "Nano Banana 2": "Nano-Banana-2.tar.gz",
}

AIGEN_VAL_GENERATORS = [
    "FLUX.1 [pro] v1.1",
    "SD 3.5 Med",
    "Reve Img 1.0",
    "HiDream I1 Dev",
    "GPT-Img 1",
    "Ideogram 3",
]
AIGEN_TEST_GENERATORS = [
    "Gemini 2.5 Flash Img",
    "FLUX.2",
    "FLUX.2 [pro]",
    "Gemini 3 Pro Img",
    "GPT-Img 1.5",
    "Seedream 4.5",
]
T2I_VAL_GENERATORS = ["FLUX.1-Krea-dev", "Qwen-Image", "Janus-Pro-7B", "GoT-R1-7B"]
T2I_TEST_GENERATORS = ["Qwen-Image-2512", "HunyuanImage-3.0", "Z-Image", "Nano Banana 2"]

GENERATOR_FAMILY = {
    "FLUX.1 [pro] v1.1": "FLUX",
    "SD 3.5 Med": "Stable Diffusion",
    "Reve Img 1.0": "Reve",
    "HiDream I1 Dev": "HiDream",
    "GPT-Img 1": "GPT-Image",
    "Ideogram 3": "Ideogram",
    "Gemini 2.5 Flash Img": "Gemini",
    "FLUX.2": "FLUX",
    "FLUX.2 [pro]": "FLUX",
    "Gemini 3 Pro Img": "Gemini",
    "GPT-Img 1.5": "GPT-Image",
    "Seedream 4.5": "Seedream",
    "FLUX.1-Krea-dev": "FLUX",
    "Qwen-Image": "Qwen-Image",
    "Janus-Pro-7B": "Janus-Pro",
    "GoT-R1-7B": "GoT-R1",
    "Qwen-Image-2512": "Qwen-Image",
    "HunyuanImage-3.0": "HunyuanImage",
    "Z-Image": "Z-Image",
    "Nano Banana 2": "Nano Banana",
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    cache: Path

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @property
    def test(self) -> Path:
        return self.root / "test"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    def create(self) -> None:
        for path in (self.root, self.cache, self.validation, self.test, self.manifests, self.metadata):
            path.mkdir(parents=True, exist_ok=True)


def configure(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).strip())


def prompt_key(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def normalize_name(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def deterministic_rank(seed: int, *parts: object) -> bytes:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def choose_variant(seed: int, generator: str, prompt_id: str, n_variants: int = 4) -> int:
    value = f"{seed}|{generator}|{prompt_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest(), "big") % n_variants


def safe_component(text: str, max_len: int = 180) -> str:
    value = str(text).replace("/", "__").replace("\\", "__")
    value = re.sub(r"[^A-Za-z0-9._+\-\[\]()]+", "_", value).strip("._")
    if len(value) > max_len:
        suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        value = value[: max_len - 13] + "_" + suffix
    return value or "unnamed"


def sanitize_stem(name: str) -> str:
    value = Path(str(name)).name
    if "." in value:
        value = value.rsplit(".", 1)[0]
    value = "".join(c if c.isalnum() or c in "-_." else "_" for c in value).strip("._")
    if not value:
        raise ValueError(f"Invalid filename stem: {name!r}")
    return value


def output_path(paths: DatasetPaths, source: str, stem: str, ext: str, split: str, label: str) -> Path:
    target = paths.validation if split == "validation" else paths.test
    suffix = ext.lower().lstrip(".")
    return target / f"{source}_{safe_component(stem)}_{label}.{suffix}"


def save_manifest(paths: DatasetPaths, frame: pd.DataFrame, name: str) -> None:
    csv_path = paths.manifests / f"{name}.csv"
    jsonl_path = paths.manifests / f"{name}.jsonl"
    frame.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    logging.info("Saved %s", csv_path)


def stream_download(session: requests.Session, url: str, dst: Path, verify: bool = True) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return {"path": str(dst), "sha256": sha256_file(dst), "bytes": dst.stat().st_size, "resolved_url": None, "status": "existing"}

    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total = 0
    with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True, verify=verify) as response:
        response.raise_for_status()
        resolved_url = response.url
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    os.replace(tmp, dst)
    return {"path": str(dst), "sha256": digest.hexdigest(), "bytes": total, "resolved_url": resolved_url, "status": "downloaded"}


def download_resumable(session: requests.Session, url: str, dst: Path, verify: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    offset = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True, verify=verify, headers=headers) as response:
        if offset and response.status_code == 416:
            return
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        with dst.open(mode) as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def detect_image_extension(prefix: bytes) -> str:
    if prefix.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if prefix[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "webp"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        brand = prefix[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "heic"
        if brand in {b"avif", b"avis"}:
            return "avif"
    raise ValueError(f"Unknown image magic: {prefix[:16]!r}")


def download_dynamic_image(session: requests.Session, url: str, paths: DatasetPaths, source: str, original_name: str, split: str) -> dict:
    with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
        response.raise_for_status()
        iterator = response.iter_content(1024 * 1024)
        first = next(iterator, b"")
        if not first:
            raise RuntimeError(f"Empty response: {url}")
        ext = detect_image_extension(first[:32])
        dst = output_path(paths, source, original_name, ext, split, "real")
        if dst.exists():
            return {"path": str(dst), "sha256": sha256_file(dst), "bytes": dst.stat().st_size, "resolved_url": response.url, "extension": ext, "status": "existing"}
        tmp = dst.with_name(dst.name + ".part")
        digest = hashlib.sha256()
        total = 0
        with tmp.open("wb") as handle:
            handle.write(first)
            digest.update(first)
            total += len(first)
            for chunk in iterator:
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        os.replace(tmp, dst)
        return {"path": str(dst), "sha256": digest.hexdigest(), "bytes": total, "resolved_url": response.url, "extension": ext, "status": "downloaded"}


def write_exact_bytes(data: bytes, dst: Path) -> tuple[str, int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    new_hash = hashlib.sha256(data).hexdigest()
    if dst.exists():
        existing_hash = sha256_file(dst)
        if existing_hash != new_hash:
            raise RuntimeError(f"Existing output has different bytes: {dst}")
        return existing_hash, dst.stat().st_size, "existing"
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return new_hash, len(data), "written"


def assert_unique_output_paths(frame: pd.DataFrame) -> None:
    duplicate = frame["output_path"].duplicated(keep=False)
    if duplicate.any():
        raise RuntimeError(frame.loc[duplicate, ["source_id", "output_path"]].to_string(index=False))


def build_fivek(paths: DatasetPaths, session: requests.Session, seed: int, workers: int) -> pd.DataFrame:
    merged: dict[str, dict] = {}
    for url in FIVEK_META_URLS:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        overlap = set(merged).intersection(payload)
        if overlap:
            raise RuntimeError(f"Duplicate FiveK basename: {next(iter(overlap))}")
        merged.update(payload)

    rows = []
    for basename, item in merged.items():
        categories = item.get("categories") or {}
        camera = item.get("camera") or {}
        dng_url = (item.get("urls") or {}).get("dng") or f"https://data.csail.mit.edu/graphics/fivek/img/dng/{basename}.dng"
        rows.append({
            "basename": basename,
            "source_id": str(item.get("id", basename)),
            "dng_url": dng_url.replace("http://data.csail.mit.edu/", "https://data.csail.mit.edu/"),
            "subject": categories.get("subject", "unknown"),
            "location": categories.get("location", "unknown"),
            "time": categories.get("time", "unknown"),
            "light": categories.get("light", "unknown"),
            "camera_make": camera.get("make", "unknown"),
            "camera_model": camera.get("model", "unknown"),
        })
    catalog = pd.DataFrame(rows).sort_values("basename").reset_index(drop=True)
    catalog["stratum_raw"] = catalog["subject"].astype(str) + "|" + catalog["location"].astype(str) + "|" + catalog["camera_make"].astype(str)
    freq = catalog["stratum_raw"].value_counts()
    catalog["stratum"] = catalog["stratum_raw"].where(catalog["stratum_raw"].map(freq) >= 20, "OTHER")
    selected, _ = train_test_split(catalog, train_size=1000, random_state=seed, shuffle=True, stratify=catalog["stratum"])
    val, test = train_test_split(selected, train_size=500, test_size=500, random_state=seed, shuffle=True, stratify=selected["stratum"])
    val, test = val.copy(), test.copy()
    val["split"], test["split"] = "validation", "test"

    def fetch(row: dict) -> dict:
        dst = output_path(paths, "FiveK", row["basename"], "dng", row["split"], "real")
        info = stream_download(session, row["dng_url"], dst)
        return {
            "source": "FiveK", "label": "real", "split": row["split"], "source_id": row["source_id"],
            "original_name": row["basename"] + ".dng", "output_path": info["path"], "extension": "dng",
            "bytes": info["bytes"], "sha256": info["sha256"], "source_url": row["dng_url"],
            "resolved_url": info["resolved_url"], "download_status": info["status"], "subject": row["subject"],
            "location": row["location"], "time": row["time"], "light": row["light"],
            "camera_make": row["camera_make"], "camera_model": row["camera_model"],
            "photographer_id": None, "device_id": None, "scene_id": None,
        }

    jobs = pd.concat([val, test], ignore_index=True).to_dict(orient="records")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="FiveK"):
            records.append(future.result())
    manifest = pd.DataFrame(records).sort_values(["split", "source_id"]).reset_index(drop=True)
    assert_unique_output_paths(manifest)
    save_manifest(paths, manifest, "real_fivek")
    return manifest


def fodb_zip_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.infolist()[:1]
        return True
    except zipfile.BadZipFile:
        return False


def build_fodb(paths: DatasetPaths, session: requests.Session, seed: int) -> pd.DataFrame:
    archive_dir = paths.cache / "fodb_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if any(not (archive_dir / name).exists() for name in FODB_PARTS):
        if shutil.disk_usage(archive_dir).free < 24 * 1024**3:
            raise RuntimeError("FODB requires at least 24 GiB of free cache space")

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    for part in FODB_PARTS:
        dst = archive_dir / part
        if not fodb_zip_valid(dst):
            logging.info("Downloading %s", part)
            download_resumable(session, FODB_OFFICIAL_PAGE.rstrip("/") + "/" + part, dst, verify=False)
        if not fodb_zip_valid(dst):
            raise RuntimeError(f"Incomplete or invalid FODB archive: {dst}")

    archive_rows = [{"archive_name": p.name, "archive_path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in (archive_dir / name for name in FODB_PARTS)]
    archive_manifest = pd.DataFrame(archive_rows)
    archive_manifest.to_csv(paths.manifests / "fodb_archive_hashes.csv", index=False)
    archive_sha = dict(zip(archive_manifest["archive_path"], archive_manifest["sha256"]))

    rows, seen = [], set()
    for part in FODB_PARTS:
        zpath = archive_dir / part
        with zipfile.ZipFile(zpath) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                match = FODB_MEMBER_RE.match(info.filename.lstrip("/"))
                if not match:
                    continue
                device_id = f"D{match.group('device')}"
                scene_id = match.group("scene")
                key = (device_id, scene_id)
                if key in seen:
                    raise RuntimeError(f"Duplicate FODB original: {key}")
                seen.add(key)
                rows.append({
                    "device_id": device_id, "scene_id": scene_id, "original_name": Path(info.filename).name,
                    "archive_name": zpath.name, "archive_path": str(zpath), "zip_member": info.filename,
                    "zip_crc32": f"{info.CRC:08x}", "jpeg_bytes": info.file_size, "zip_compressed_bytes": info.compress_size,
                })
    catalog = pd.DataFrame(rows).sort_values(["device_id", "scene_id"]).reset_index(drop=True)
    if catalog["device_id"].nunique() != 27:
        raise RuntimeError(f"Expected 27 FODB devices, found {catalog['device_id'].nunique()}")

    devices = sorted(catalog["device_id"].unique())
    scenes = sorted(catalog["scene_id"].unique())
    present = set(zip(catalog["device_id"], catalog["scene_id"]))
    rng = random.Random(seed)
    rng.shuffle(devices)
    rng.shuffle(scenes)
    best = None
    for offset in range(len(devices)):
        rotated = devices[offset:] + devices[:offset]
        val_devices, test_devices = tuple(rotated[:10]), tuple(rotated[10:20])
        val_candidates = [scene for scene in scenes if all((device, scene) in present for device in val_devices)]
        test_candidates = [scene for scene in scenes if all((device, scene) in present for device in test_devices)]
        val_scenes = val_candidates[:40]
        test_scenes = [scene for scene in test_candidates if scene not in set(val_scenes)][:40]
        score = (len(val_scenes), len(test_scenes))
        if best is None or score > best["score"]:
            best = {"score": score, "val_devices": val_devices, "test_devices": test_devices, "val_scenes": tuple(val_scenes), "test_scenes": tuple(test_scenes)}
        if score == (40, 40):
            break
    if not best or best["score"] != (40, 40):
        raise RuntimeError(f"Could not construct FODB split; best coverage={best['score'] if best else None}")

    val = catalog[catalog["device_id"].isin(best["val_devices"]) & catalog["scene_id"].isin(best["val_scenes"])].copy()
    test = catalog[catalog["device_id"].isin(best["test_devices"]) & catalog["scene_id"].isin(best["test_scenes"])].copy()
    val["split"], test["split"] = "validation", "test"
    split_info = {
        "seed": seed, "val_devices": sorted(best["val_devices"]), "test_devices": sorted(best["test_devices"]),
        "val_scenes": sorted(best["val_scenes"]), "test_scenes": sorted(best["test_scenes"]),
    }
    (paths.manifests / f"fodb_split_seed{seed}.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    def extract(row: dict) -> dict:
        dst = output_path(paths, "FODB", Path(row["original_name"]).stem, "jpg", row["split"], "real")
        if dst.exists():
            digest, size, status = sha256_file(dst), dst.stat().st_size, "existing"
        else:
            tmp = dst.with_name(dst.name + ".part")
            digest = hashlib.sha256()
            size = 0
            with zipfile.ZipFile(row["archive_path"]) as archive, archive.open(row["zip_member"]) as src, tmp.open("wb") as out:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            os.replace(tmp, dst)
            digest, status = digest.hexdigest(), "extracted_from_official_zip"
        return {
            "source": "FODB", "label": "real", "split": row["split"], "source_id": f"{row['device_id']}:{row['scene_id']}",
            "original_name": row["original_name"], "output_path": str(dst), "extension": "jpg", "bytes": size,
            "sha256": digest, "source_url": FODB_OFFICIAL_PAGE.rstrip("/") + "/" + row["archive_name"],
            "archive_name": row["archive_name"], "archive_path": row["archive_path"], "archive_sha256": archive_sha[row["archive_path"]],
            "zip_member": row["zip_member"], "zip_crc32": row["zip_crc32"], "jpeg_bytes_catalog": row["jpeg_bytes"],
            "zip_compressed_bytes": row["zip_compressed_bytes"], "download_status": status, "tls_certificate_verified": False,
            "photographer_id": None, "device_id": row["device_id"], "scene_id": row["scene_id"],
        }

    records = [extract(row) for row in tqdm(pd.concat([val, test], ignore_index=True).to_dict(orient="records"), desc="FODB")]
    manifest = pd.DataFrame(records).sort_values(["split", "device_id", "scene_id"]).reset_index(drop=True)
    if len(manifest[manifest.split == "validation"]) != 400 or len(manifest[manifest.split == "test"]) != 400:
        raise RuntimeError("FODB split count mismatch")
    if not (manifest["bytes"] == manifest["jpeg_bytes_catalog"]).all():
        raise RuntimeError("FODB extracted byte-size mismatch")
    assert_unique_output_paths(manifest)
    save_manifest(paths, manifest, "real_fodb")
    return manifest


def clean_hf_scalar(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "null"} else value


def build_unsplash(paths: DatasetPaths, session: requests.Session, seed: int, workers: int, hf_token: str | None) -> pd.DataFrame:
    from datasets import load_dataset

    cache = paths.cache / "hf" / "unsplash_lite"
    dataset = load_dataset(UNSPLASH_HF_REPO, split="train", revision=UNSPLASH_HF_REVISION, cache_dir=str(cache), streaming=True, token=hf_token)
    rows = []
    for item in tqdm(dataset, desc="Unsplash metadata"):
        photo, photographer, exif, location = item.get("photo") or {}, item.get("photographer") or {}, item.get("exif") or {}, item.get("location") or {}
        rows.append({
            "photo_id": clean_hf_scalar(photo.get("id")), "photo_url": clean_hf_scalar(photo.get("url")),
            "photo_image_url": clean_hf_scalar(photo.get("image_url")), "photo_width": photo.get("width"), "photo_height": photo.get("height"),
            "photo_submitted_at": clean_hf_scalar(photo.get("submitted_at")), "photographer_username": clean_hf_scalar(photographer.get("username")),
            "exif_camera_make": clean_hf_scalar(exif.get("camera_make")), "exif_camera_model": clean_hf_scalar(exif.get("camera_model")),
            "exif_iso": clean_hf_scalar(exif.get("iso")), "exif_aperture_value": clean_hf_scalar(exif.get("aperture_value")),
            "exif_focal_length": clean_hf_scalar(exif.get("focal_length")), "exif_exposure_time": clean_hf_scalar(exif.get("exposure_time")),
            "photo_location_name": clean_hf_scalar(location.get("name")),
        })
    photos = pd.DataFrame(rows)
    if len(photos) < 24000:
        raise RuntimeError(f"Unexpectedly small Unsplash catalog: {len(photos)} rows")
    eligible = photos.copy()
    for col in ["photo_id", "photo_image_url", "photographer_username", "exif_camera_make", "exif_camera_model"]:
        eligible = eligible[eligible[col].notna() & eligible[col].astype(str).str.strip().ne("")]
    eligible["megapixels"] = pd.to_numeric(eligible["photo_width"], errors="coerce") * pd.to_numeric(eligible["photo_height"], errors="coerce") / 1_000_000
    eligible = eligible[eligible["megapixels"] >= 5.0].drop_duplicates("photo_id").reset_index(drop=True)

    counts = eligible.groupby("photographer_username").size()
    users = sorted(counts.index)
    rng = random.Random(seed)
    rng.shuffle(users)
    val_users, test_users, val_cap, test_cap = [], [], 0, 0
    for user in users:
        cap = min(2, int(counts[user]))
        if val_cap < 350 and (val_cap <= test_cap or test_cap >= 350):
            val_users.append(user)
            val_cap += cap
        elif test_cap < 350:
            test_users.append(user)
            test_cap += cap
        if val_cap >= 350 and test_cap >= 350:
            break
    if val_cap < 350 or test_cap < 350:
        raise RuntimeError("Insufficient Unsplash photographer-disjoint capacity")

    def select(user_list: list[str]) -> pd.DataFrame:
        local_rng = random.Random(seed)
        indices = []
        for user in user_list:
            candidates = eligible.index[eligible["photographer_username"] == user].tolist()
            local_rng.shuffle(candidates)
            indices.extend(candidates[:2])
        local_rng.shuffle(indices)
        return eligible.loc[indices[:350]].copy()

    val, test = select(val_users), select(test_users)
    val["split"], test["split"] = "validation", "test"

    def fetch(row: dict) -> dict:
        original_name = str(row["photo_id"])
        info = download_dynamic_image(session, str(row["photo_image_url"]), paths, "Unsplash", original_name, row["split"])
        return {
            "source": "Unsplash", "label": "real", "split": row["split"], "source_id": original_name,
            "original_name": original_name, "output_path": info["path"], "extension": info["extension"], "bytes": info["bytes"],
            "sha256": info["sha256"], "source_url": row["photo_image_url"], "resolved_url": info["resolved_url"],
            "download_status": info["status"], "location": row.get("photo_location_name"), "time": row.get("photo_submitted_at"),
            "camera_make": row.get("exif_camera_make"), "camera_model": row.get("exif_camera_model"),
            "photographer_id": row.get("photographer_username"), "device_id": None, "scene_id": None,
            "photo_width_catalog": row.get("photo_width"), "photo_height_catalog": row.get("photo_height"), "megapixels_catalog": row.get("megapixels"),
            "exif_iso_catalog": row.get("exif_iso"), "exif_aperture_value_catalog": row.get("exif_aperture_value"),
            "exif_focal_length_catalog": row.get("exif_focal_length"), "exif_exposure_time_catalog": row.get("exif_exposure_time"),
            "photo_url": row.get("photo_url"), "metadata_hf_repo": UNSPLASH_HF_REPO, "metadata_hf_revision": UNSPLASH_HF_REVISION,
        }

    jobs = pd.concat([val, test], ignore_index=True).to_dict(orient="records")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Unsplash"):
            records.append(future.result())
    manifest = pd.DataFrame(records).sort_values(["split", "source_id"]).reset_index(drop=True)
    assert_unique_output_paths(manifest)
    if not set(manifest.loc[manifest.split == "validation", "photographer_id"]).isdisjoint(set(manifest.loc[manifest.split == "test", "photographer_id"])):
        raise RuntimeError("Unsplash photographer leakage")
    save_manifest(paths, manifest, "real_unsplash")
    return manifest


def audit_real(paths: DatasetPaths, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    all_real = pd.concat(list(frames), ignore_index=True, sort=False)
    expected = {
        ("validation", "FiveK"): 500, ("validation", "FODB"): 400, ("validation", "Unsplash"): 350,
        ("test", "FiveK"): 500, ("test", "FODB"): 400, ("test", "Unsplash"): 350,
    }
    actual = all_real.groupby(["split", "source"]).size().to_dict()
    if actual != expected:
        raise RuntimeError(f"REAL composition mismatch: {actual}")
    if not all_real["output_path"].is_unique:
        raise RuntimeError("Duplicate REAL output paths")
    failures = []
    for row in tqdm(all_real.to_dict(orient="records"), desc="REAL integrity"):
        path = Path(row["output_path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"REAL integrity failures: {failures[:10]}")
    val_hashes = set(all_real.loc[all_real.split == "validation", "sha256"])
    test_hashes = set(all_real.loc[all_real.split == "test", "sha256"])
    if val_hashes & test_hashes:
        raise RuntimeError("Byte-identical REAL files across validation/test")
    fodb = all_real[all_real.source == "FODB"]
    if not set(fodb.loc[fodb.split == "validation", "device_id"]).isdisjoint(set(fodb.loc[fodb.split == "test", "device_id"])):
        raise RuntimeError("FODB device leakage")
    if not set(fodb.loc[fodb.split == "validation", "scene_id"]).isdisjoint(set(fodb.loc[fodb.split == "test", "scene_id"])):
        raise RuntimeError("FODB scene leakage")
    unsplash = all_real[all_real.source == "Unsplash"]
    if not set(unsplash.loc[unsplash.split == "validation", "photographer_id"]).isdisjoint(set(unsplash.loc[unsplash.split == "test", "photographer_id"])):
        raise RuntimeError("Unsplash photographer leakage")
    fivek = all_real[all_real.source == "FiveK"]
    if not set(fivek.loc[fivek.split == "validation", "source_id"].astype(str)).isdisjoint(set(fivek.loc[fivek.split == "test", "source_id"].astype(str))):
        raise RuntimeError("FiveK source-ID leakage")
    save_manifest(paths, all_real[all_real.split == "validation"].copy(), "real_validation")
    save_manifest(paths, all_real[all_real.split == "test"].copy(), "real_test")
    save_manifest(paths, all_real, "real_all")
    return all_real


AIGEN_RAW_TO_CANONICAL = {
    normalize_name("flux-pro-v1.1"): "FLUX.1 [pro] v1.1",
    normalize_name("stable-diffusion-v35-medium"): "SD 3.5 Med",
    normalize_name("reve"): "Reve Img 1.0",
    normalize_name("hidream-i1-dev"): "HiDream I1 Dev",
    normalize_name("gpt-image-1"): "GPT-Img 1",
    normalize_name("ideogram-v3"): "Ideogram 3",
    normalize_name("gemini-25-flash-image"): "Gemini 2.5 Flash Img",
    normalize_name("flux-2"): "FLUX.2",
    normalize_name("flux-2-pro"): "FLUX.2 [pro]",
    normalize_name("gemini-3-pro-image-preview"): "Gemini 3 Pro Img",
    normalize_name("gpt-image-1.5"): "GPT-Img 1.5",
    normalize_name("seedream-v4.5"): "Seedream 4.5",
    normalize_name("seedreamv4.5"): "Seedream 4.5",
}


def infer_aigen_raw_generator(member_name: str) -> str:
    path = PurePosixPath(member_name)
    parts = list(path.parts)
    fake_idx = next((i for i, part in enumerate(parts) if part.lower() == "1_fake"), None)
    if fake_idx is None:
        raise ValueError(f"Not an AIGen fake image: {member_name}")
    if fake_idx + 1 < len(parts) - 1:
        return parts[fake_idx + 1]
    stem = path.stem
    if stem.startswith("Firefly_"):
        return "firefly-image5"
    if stem.startswith("image_midjourneyv7_"):
        return "midjourney-v7"
    match = re.match(r"^TEST_(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", stem, re.I)
    if match:
        return match.group(1)
    match = re.match(r"^fal-ai_(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+$", stem, re.I)
    if not match:
        raise ValueError(f"Cannot infer AIGen generator from {member_name}")
    raw = match.group(1)
    aliases = {
        "bytedance_seedream_v4.5_text-to-image": "seedream-v4.5", "firefly_image5": "firefly-image5",
        "flux_dev": "flux-dev", "flux-pro_v1.1": "flux-pro-v1.1", "gpt-image-1_text-to-image": "gpt-image-1",
        "ideogram_v3": "ideogram-v3", "imagen4_preview": "imagen4", "reve_text-to-image": "reve", "z-image_turbo": "z-image-turbo",
    }
    return aliases.get(raw, raw.replace("_", "-"))


def parse_aigen_metadata(payload: bytes, member_name: str) -> dict | None:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if "\x00" in text[:4096]:
        return None
    required = {"image_id", "filename", "caption", "caption_id", "split"}
    for delimiter in [",", "\t", ";", "|"]:
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            continue
        normalized = {str(name).replace("\ufeff", "").strip().lower(): name for name in reader.fieldnames}
        if not required.issubset(normalized):
            continue
        rows = []
        for source in reader:
            split = str(source[normalized["split"]]).strip().lower()
            split = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val"}.get(split)
            if split is None:
                rows = []
                break
            rows.append({field: str(source[normalized[field]] or "").strip() for field in required} | {"split": split})
        if rows:
            return {"member": member_name, "rows": rows, "split_counts": dict(Counter(row["split"] for row in rows))}
    return None


def resolve_aigen_archive(paths: DatasetPaths, hf_token: str | None) -> Path:
    cache = paths.cache / "hf" / "aigenimages2026"
    cache.mkdir(parents=True, exist_ok=True)
    for candidate in paths.cache.rglob(AIGEN_ARCHIVE_FILENAME):
        if candidate.is_file() and candidate.stat().st_size == AIGEN_ARCHIVE_BYTES and sha256_file(candidate) == AIGEN_ARCHIVE_SHA256:
            return candidate
    if shutil.disk_usage(cache).free < 14 * 1024**3:
        raise RuntimeError("AIGenImages2026 requires at least 14 GiB of free cache space")
    archive = Path(hf_hub_download(repo_id=AIGEN_REPO, filename=AIGEN_ARCHIVE_FILENAME, revision=AIGEN_REVISION, repo_type="dataset", cache_dir=str(cache), token=hf_token))
    if archive.stat().st_size != AIGEN_ARCHIVE_BYTES or sha256_file(archive) != AIGEN_ARCHIVE_SHA256:
        raise RuntimeError("AIGenImages2026 archive verification failed")
    return archive


def build_aigen(paths: DatasetPaths, seed: int, hf_token: str | None) -> pd.DataFrame:
    archive = resolve_aigen_archive(paths, hf_token)
    image_rows, metadata_tables = [], []
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    with tarfile.open(archive, "r:*") as handle:
        for info in tqdm(handle.getmembers(), desc="AIGen scan"):
            if not info.isfile():
                continue
            path = PurePosixPath(info.name)
            parts = [part.lower() for part in path.parts]
            if path.suffix.lower() in image_suffixes and "1_fake" in parts:
                split = next((part for part in parts if part in {"train", "val"}), None)
                if split:
                    raw = infer_aigen_raw_generator(info.name)
                    image_rows.append({
                        "tar_member": info.name, "publisher_split": split, "filename": path.name,
                        "image_ext": path.suffix.lower().lstrip("."), "member_bytes": info.size,
                        "raw_generator": raw, "generator": AIGEN_RAW_TO_CANONICAL.get(normalize_name(raw)),
                    })
                continue
            if 0 < info.size <= 64 * 1024 * 1024 and path.suffix.lower() in {".csv", ".tsv", ".txt", ""}:
                src = handle.extractfile(info)
                if src:
                    parsed = parse_aigen_metadata(src.read(), info.name)
                    if parsed:
                        metadata_tables.append(parsed)
    if len(image_rows) != 5438:
        raise RuntimeError(f"Expected 5438 AIGen synthetic image members, found {len(image_rows)}")

    tables = {}
    for split, expected in {"train": 4880, "val": 559}.items():
        candidates = []
        for table in metadata_tables:
            rows = [row for row in table["rows"] if row["split"] == split]
            if len(rows) == expected:
                candidates.append(rows)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one AIGen metadata table for {split}, found {len(candidates)}")
        tables[split] = candidates[0]

    meta_lookup = {}
    for split, rows in tables.items():
        for row in rows:
            key = (split, PurePosixPath(row["filename"]).name)
            if key in meta_lookup:
                raise RuntimeError(f"Duplicate AIGen metadata filename: {key}")
            meta_lookup[key] = row
    image_keys = {(row["publisher_split"], row["filename"]) for row in image_rows}
    missing = sorted(set(meta_lookup) - image_keys)
    if missing != [("train", "image_midjourneyv7_300.png")]:
        raise RuntimeError(f"Unexpected AIGen metadata/member mismatch: {missing[:20]}")

    catalog = []
    for image in image_rows:
        meta = meta_lookup[(image["publisher_split"], image["filename"])]
        catalog.append(image | {"image_id": meta["image_id"], "caption_id": meta["caption_id"], "prompt": meta["caption"], "prompt_key": prompt_key(meta["caption"])})
    catalog = pd.DataFrame(catalog)
    targets = set(AIGEN_VAL_GENERATORS + AIGEN_TEST_GENERATORS)
    selected = catalog[catalog["generator"].isin(targets)].copy()
    expected_counts = {
        "FLUX.1 [pro] v1.1": 305, "SD 3.5 Med": 305, "Reve Img 1.0": 305, "HiDream I1 Dev": 305,
        "GPT-Img 1": 305, "Ideogram 3": 305, "Gemini 2.5 Flash Img": 305, "FLUX.2": 306,
        "FLUX.2 [pro]": 305, "Gemini 3 Pro Img": 307, "GPT-Img 1.5": 305, "Seedream 4.5": 305,
    }
    if selected.groupby("generator").size().to_dict() != expected_counts:
        raise RuntimeError("AIGen generator census mismatch")

    unique_rows = []
    for (_, _), group in selected.groupby(["generator", "prompt_key"], sort=True):
        if len(group) == 1:
            row = group.iloc[0].copy()
            row["duplicate_group_size"] = 1
            row["duplicate_selection_rule"] = "only_member"
        else:
            ranked = group.copy()
            ranked["_rank"] = ranked.apply(lambda r: hashlib.sha256(f"{seed}|AIGen-duplicate|{r['generator']}|{r['prompt_key']}|{r['tar_member']}".encode()).hexdigest(), axis=1)
            ranked = ranked.sort_values(["_rank", "tar_member"], kind="mergesort")
            row = ranked.iloc[0].drop(labels=["_rank"]).copy()
            row["duplicate_group_size"] = len(group)
            row["duplicate_selection_rule"] = f"seed{seed}_sha256_generator_prompt_member"
        unique_rows.append(row)
    unique = pd.DataFrame(unique_rows).reset_index(drop=True)
    if any(unique.groupby("generator")["prompt_key"].nunique() != 305):
        raise RuntimeError("AIGen unique prompt census mismatch")

    prompt_sets = {g: set(unique.loc[unique.generator == g, "prompt_key"]) for g in targets}
    common_test = set.intersection(*(prompt_sets[g] for g in AIGEN_TEST_GENERATORS))
    common_val = set.intersection(*(prompt_sets[g] for g in AIGEN_VAL_GENERATORS))
    test_keys = sorted(common_test, key=lambda key: deterministic_rank(seed, "AIGen", "test", key))[:125]
    val_keys = sorted(common_val - set(test_keys), key=lambda key: deterministic_rank(seed, "AIGen", "validation", key))[:125]
    if len(test_keys) != 125 or len(val_keys) != 125:
        raise RuntimeError("Insufficient common AIGen prompts")

    plan_parts = []
    for split, generators, keys in [("validation", AIGEN_VAL_GENERATORS, val_keys), ("test", AIGEN_TEST_GENERATORS, test_keys)]:
        for generator in generators:
            part = unique[(unique.generator == generator) & unique.prompt_key.isin(keys)].copy()
            if len(part) != 125 or part["prompt_key"].duplicated().any():
                raise RuntimeError(f"AIGen selection mismatch for {generator} {split}")
            part["split"] = split
            plan_parts.append(part)
    plan = pd.concat(plan_parts, ignore_index=True)
    save_manifest(paths, plan, "ai_aigen2026_selection_plan")

    selected_by_member = {row["tar_member"]: row for row in plan.to_dict(orient="records")}
    remaining = set(selected_by_member)
    records = []
    with tarfile.open(archive, "r:*") as handle:
        for info in tqdm(handle, desc="AIGen extract"):
            if not info.isfile() or info.name not in remaining:
                continue
            src = handle.extractfile(info)
            if src is None:
                raise RuntimeError(f"Cannot read AIGen member: {info.name}")
            data = src.read()
            if len(data) != info.size:
                raise RuntimeError(f"AIGen member size mismatch: {info.name}")
            row = selected_by_member[info.name]
            stem = f"{safe_component(row['generator'])}__{Path(row['filename']).stem}"
            dst = output_path(paths, "AIGen2026", stem, row["image_ext"], row["split"], "ai")
            digest, size, status = write_exact_bytes(data, dst)
            records.append({
                "source": "AIGenImages2026", "label": "ai", "split": row["split"], "generator": row["generator"],
                "raw_generator": row["raw_generator"], "generator_family": GENERATOR_FAMILY[row["generator"]],
                "prompt": row["prompt"], "prompt_key": row["prompt_key"], "prompt_id": row["caption_id"], "image_id": row["image_id"],
                "publisher_split": row["publisher_split"], "duplicate_group_size": int(row["duplicate_group_size"]),
                "duplicate_selection_rule": row["duplicate_selection_rule"], "variant_id": None, "tar_member": info.name,
                "original_name": row["filename"], "output_path": str(dst), "extension": row["image_ext"], "bytes": size,
                "sha256": digest, "source_repo": AIGEN_REPO, "source_revision": AIGEN_REVISION,
                "source_archive": AIGEN_ARCHIVE_FILENAME, "source_archive_sha256": AIGEN_ARCHIVE_SHA256, "write_status": status,
            })
            remaining.remove(info.name)
    if remaining:
        raise RuntimeError(f"Missing selected AIGen members: {len(remaining)}")
    manifest = pd.DataFrame(records).sort_values(["split", "generator", "prompt_key"]).reset_index(drop=True)
    if len(manifest) != 1500:
        raise RuntimeError(f"Expected 1500 AIGen outputs, found {len(manifest)}")
    save_manifest(paths, manifest, "ai_aigen2026")
    return manifest


def load_t2i_prompts(paths: DatasetPaths, hf_token: str | None) -> pd.DataFrame:
    cache = paths.cache / "hf" / "t2i_prompts"
    rows = []
    for dim in T2I_DIMENSIONS:
        relpath = f"data/{dim}.json"
        local = Path(hf_hub_download(repo_id=T2I_PROMPT_REPO, filename=relpath, revision=T2I_PROMPT_REVISION, repo_type="dataset", cache_dir=str(cache), token=hf_token))
        data = json.loads(local.read_text(encoding="utf-8"))
        expected_ids = [f"{dim}-{i:03d}" for i in range(1, 91)]
        if set(data) != set(expected_ids):
            raise RuntimeError(f"T2I prompt-ID census changed for {dim}")
        expected_main = "Composition" if dim.startswith("C-") else "Reasoning"
        for pid in expected_ids:
            entry = data[pid]
            if entry.get("Main Class") != expected_main:
                raise RuntimeError(f"Unexpected T2I main class for {pid}")
            prompt = str(entry["Prompt"])
            rows.append({
                "prompt_id": pid, "prompt": prompt, "prompt_key": prompt_key(prompt), "dimension": dim,
                "main_class": entry["Main Class"], "sub_class": entry["Sub Class"], "remark": entry["Remark"],
                "prompt_source_file": relpath, "prompt_source_revision": T2I_PROMPT_REVISION,
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 1080 or frame["prompt_id"].nunique() != 1080 or frame["prompt_key"].nunique() != 1080:
        raise RuntimeError("T2I prompt census mismatch")
    return frame


def make_t2i_plan(paths: DatasetPaths, prompts: pd.DataFrame, aigen_manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    aigen_keys = set(aigen_manifest["prompt_key"])
    if len(aigen_keys) != 250:
        raise RuntimeError(f"Expected 250 unique AIGen prompts, found {len(aigen_keys)}")
    candidates = prompts[~prompts["prompt_key"].isin(aigen_keys)].copy().reset_index(drop=True)
    dimensions = T2I_DIMENSIONS.copy()
    ordered_dims = sorted(dimensions, key=lambda dim: deterministic_rank(seed, "T2I-dimension", dim))
    val_extra, test_extra = set(ordered_dims[:4]), set(ordered_dims[4:8])
    val_parts, test_parts = [], []
    for dim in dimensions:
        part = candidates[candidates.dimension == dim].copy()
        part["_rank"] = part["prompt_id"].map(lambda pid: hashlib.sha256(f"{seed}|T2I-prompt|{dim}|{pid}".encode()).hexdigest())
        part = part.sort_values(["_rank", "prompt_id"], kind="mergesort").drop(columns="_rank").reset_index(drop=True)
        n_val, n_test = (34 if dim in val_extra else 33), (34 if dim in test_extra else 33)
        if len(part) < n_val + n_test:
            raise RuntimeError(f"Insufficient T2I prompts in {dim}")
        val_parts.append(part.iloc[:n_val].copy())
        test_parts.append(part.iloc[n_val:n_val + n_test].copy())
    val_prompts, test_prompts = pd.concat(val_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)
    if len(val_prompts) != 400 or len(test_prompts) != 400 or not set(val_prompts.prompt_key).isdisjoint(set(test_prompts.prompt_key)):
        raise RuntimeError("T2I prompt split mismatch")

    def assignments(prompt_df: pd.DataFrame, generators: list[str], split: str) -> pd.DataFrame:
        rows = sorted(prompt_df.to_dict(orient="records"), key=lambda row: deterministic_rank(seed, "T2I-assignment", split, row["prompt_id"]))
        output = []
        for i, row in enumerate(rows):
            output.append(row | {"split": split, "generator": generators[i % 4], "assignment_role": "primary"})
        for i, row in enumerate(rows[:100]):
            output.append(row | {"split": split, "generator": generators[(i % 4 + 1) % 4], "assignment_role": "secondary"})
        frame = pd.DataFrame(output)
        frame["requested_variant"] = frame.apply(lambda r: choose_variant(seed, r["generator"], r["prompt_id"]), axis=1)
        if frame.groupby("generator").size().to_dict() != {g: 125 for g in generators}:
            raise RuntimeError(f"T2I generator allocation mismatch for {split}")
        if frame.groupby("prompt_id").size().value_counts().to_dict() != {1: 300, 2: 100}:
            raise RuntimeError(f"T2I prompt multiplicity mismatch for {split}")
        return frame

    plan = pd.concat([assignments(val_prompts, T2I_VAL_GENERATORS, "validation"), assignments(test_prompts, T2I_TEST_GENERATORS, "test")], ignore_index=True)
    save_manifest(paths, plan, "ai_t2icorebench_selection_plan")
    return plan


def parse_t2i_member(member_name: str) -> dict | None:
    path = PurePosixPath(member_name)
    ext = path.suffix.lower().lstrip(".")
    if ext not in IMAGE_EXTS:
        return None
    match = re.fullmatch(r"(?P<prompt_id>[CR]-[A-Z]{2}-\d{3})-(?P<variant>[0-3])", path.stem)
    if not match:
        return None
    pid, variant = match.group("prompt_id"), int(match.group("variant"))
    dimension = "-".join(pid.split("-")[:2])
    if path.parent.name != dimension:
        raise RuntimeError(f"T2I path/dimension mismatch: {member_name}")
    return {"prompt_id": pid, "variant": variant, "extension": ext, "original_name": path.name}


def stage_tar_member(handle: tarfile.TarFile, member: tarfile.TarInfo, dst: Path) -> dict:
    src = handle.extractfile(member)
    if src is None:
        raise RuntimeError(f"Cannot extract TAR member: {member.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    digest, total = hashlib.sha256(), 0
    with src, tmp.open("wb") as out:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            out.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    if total != member.size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"TAR member size mismatch: {member.name}")
    os.replace(tmp, dst)
    return {"stage_path": str(dst), "bytes": total, "sha256": digest.hexdigest()}


def finalize_stage(stage_path: Path, dst: Path, expected_hash: str, expected_bytes: int) -> str:
    if stage_path.stat().st_size != expected_bytes or sha256_file(stage_path) != expected_hash:
        raise RuntimeError(f"Staged file verification failed: {stage_path}")
    if dst.exists():
        if dst.stat().st_size != expected_bytes or sha256_file(dst) != expected_hash:
            raise RuntimeError(f"Existing output mismatch: {dst}")
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(stage_path, tmp)
    if tmp.stat().st_size != expected_bytes or sha256_file(tmp) != expected_hash:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Final copy verification failed: {dst}")
    os.replace(tmp, dst)
    return "written"


def build_t2i(paths: DatasetPaths, seed: int, hf_token: str | None, aigen_manifest: pd.DataFrame, keep_cache: bool) -> pd.DataFrame:
    prompts = load_t2i_prompts(paths, hf_token)
    plan = make_t2i_plan(paths, prompts, aigen_manifest, seed)
    api = HfApi(token=hf_token)
    repo_info = api.dataset_info(T2I_IMAGE_REPO, revision=T2I_IMAGE_REVISION, files_metadata=True)
    if repo_info.sha != T2I_IMAGE_REVISION:
        raise RuntimeError("T2I image repository revision mismatch")
    siblings = {s.rfilename: s for s in repo_info.siblings}
    generators = T2I_VAL_GENERATORS + T2I_TEST_GENERATORS
    archive_spec = {}
    inventory = []
    for generator in generators:
        archive_name = T2I_ARCHIVE_MAP[generator]
        sibling = siblings.get(archive_name)
        size = getattr(sibling, "size", None) if sibling else None
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Missing T2I archive metadata: {archive_name}")
        archive_spec[generator] = {"archive_name": archive_name, "expected_bytes": size}
        inventory.append({"generator": generator, "archive_name": archive_name, "expected_bytes": size, "repo_revision": T2I_IMAGE_REVISION})
    (paths.manifests / f"t2i_source_inventory_seed{seed}.json").write_text(json.dumps({"seed": seed, "prompt_repo": T2I_PROMPT_REPO, "prompt_revision": T2I_PROMPT_REVISION, "image_repo": T2I_IMAGE_REPO, "image_revision": T2I_IMAGE_REVISION, "archives": inventory}, indent=2), encoding="utf-8")

    cache_root = paths.cache / "hf" / "t2i_models"
    stage_root = paths.cache / "t2i_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    official_ids = set(prompts["prompt_id"])
    records = []
    for generator in generators:
        model_plan = plan[plan.generator == generator].copy().reset_index(drop=True)
        wanted = {str(row["prompt_id"]): row for row in model_plan.to_dict(orient="records")}
        spec = archive_spec[generator]
        required_free = int(spec["expected_bytes"]) + 4 * 1024**3
        if shutil.disk_usage(paths.cache).free < required_free:
            raise RuntimeError(f"Insufficient cache space for {spec['archive_name']}")
        model_cache = cache_root / safe_component(generator)
        stage_dir = stage_root / safe_component(generator)
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        archive_path = Path(hf_hub_download(repo_id=T2I_IMAGE_REPO, filename=spec["archive_name"], revision=T2I_IMAGE_REVISION, repo_type="dataset", cache_dir=str(model_cache), token=hf_token))
        if archive_path.stat().st_size != spec["expected_bytes"]:
            raise RuntimeError(f"T2I archive size mismatch: {generator}")
        archive_sha = sha256_file(archive_path)
        variants, members, selected_stage = defaultdict(set), {}, {}
        parsed_count = 0
        with tarfile.open(archive_path, "r|*") as handle:
            for member in tqdm(handle, desc=f"T2I {generator}", unit="member"):
                if not member.isfile():
                    continue
                parsed = parse_t2i_member(member.name)
                if parsed is None:
                    continue
                parsed_count += 1
                pid, variant = parsed["prompt_id"], parsed["variant"]
                if pid not in official_ids:
                    raise RuntimeError(f"Unexpected T2I prompt ID: {pid}")
                key = (pid, variant)
                if key in members:
                    raise RuntimeError(f"Duplicate T2I member for {pid}-{variant}")
                members[key] = member.name
                variants[pid].add(variant)
                row = wanted.get(pid)
                if row and variant == int(row["requested_variant"]):
                    staged = stage_tar_member(handle, member, stage_dir / f"{pid}-{variant}.{parsed['extension']}")
                    selected_stage[pid] = staged | {"variant": variant, "tar_member": member.name, "original_name": parsed["original_name"], "extension": parsed["extension"]}
        if parsed_count != 4320 or set(variants) != official_ids or any(v != {0, 1, 2, 3} for v in variants.values()):
            raise RuntimeError(f"T2I archive census mismatch: {generator}")
        if len(selected_stage) != 125:
            raise RuntimeError(f"T2I selected member count mismatch: {generator}")
        for pid, row in wanted.items():
            staged = selected_stage[pid]
            stem = f"{safe_component(generator)}__{Path(staged['original_name']).stem}"
            dst = output_path(paths, "T2ICoReBench", stem, staged["extension"], row["split"], "ai")
            status = finalize_stage(Path(staged["stage_path"]), dst, staged["sha256"], int(staged["bytes"]))
            records.append({
                "source": "T2I-CoReBench-Images", "label": "ai", "split": row["split"], "generator": generator,
                "generator_family": GENERATOR_FAMILY[generator], "prompt": row["prompt"], "prompt_key": row["prompt_key"],
                "prompt_id": pid, "dimension": row["dimension"], "main_class": row["main_class"], "sub_class": row["sub_class"],
                "remark": row.get("remark"), "assignment_role": row["assignment_role"], "requested_variant": int(row["requested_variant"]),
                "variant_id": int(staged["variant"]), "variant_fallback_used": False, "tar_member": staged["tar_member"],
                "original_name": staged["original_name"], "output_path": str(dst), "extension": staged["extension"],
                "bytes": int(staged["bytes"]), "sha256": staged["sha256"], "prompt_source_repo": T2I_PROMPT_REPO,
                "prompt_source_revision": T2I_PROMPT_REVISION, "image_source_repo": T2I_IMAGE_REPO,
                "image_source_revision": T2I_IMAGE_REVISION, "source_archive": spec["archive_name"],
                "source_archive_expected_bytes": int(spec["expected_bytes"]), "source_archive_sha256": archive_sha, "write_status": status,
            })
        shutil.rmtree(stage_dir, ignore_errors=True)
        if not keep_cache:
            shutil.rmtree(model_cache, ignore_errors=True)
    manifest = pd.DataFrame(records).sort_values(["split", "generator", "prompt_id"]).reset_index(drop=True)
    if len(manifest) != 1000:
        raise RuntimeError(f"Expected 1000 T2I outputs, found {len(manifest)}")
    save_manifest(paths, manifest, "ai_t2icorebench")
    return manifest


def audit_ai(paths: DatasetPaths, aigen: pd.DataFrame, t2i: pd.DataFrame) -> pd.DataFrame:
    all_ai = pd.concat([aigen, t2i], ignore_index=True, sort=False)
    expected = {
        ("validation", "AIGenImages2026"): 750, ("validation", "T2I-CoReBench-Images"): 500,
        ("test", "AIGenImages2026"): 750, ("test", "T2I-CoReBench-Images"): 500,
    }
    if all_ai.groupby(["split", "source"]).size().to_dict() != expected:
        raise RuntimeError("AI composition mismatch")
    val_generators = set(all_ai.loc[all_ai.split == "validation", "generator"])
    test_generators = set(all_ai.loc[all_ai.split == "test", "generator"])
    if val_generators & test_generators:
        raise RuntimeError("Generator leakage across validation/test")
    if set(all_ai.loc[all_ai.split == "validation", "prompt_key"]) & set(all_ai.loc[all_ai.split == "test", "prompt_key"]):
        raise RuntimeError("Prompt leakage across validation/test")
    if all_ai["output_path"].duplicated().any():
        raise RuntimeError("Duplicate AI output paths")
    failures = []
    for row in tqdm(all_ai.to_dict(orient="records"), desc="AI integrity"):
        path = Path(row["output_path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"AI integrity failures: {failures[:10]}")
    if set(all_ai.loc[all_ai.split == "validation", "sha256"]) & set(all_ai.loc[all_ai.split == "test", "sha256"]):
        raise RuntimeError("Byte-identical AI files across validation/test")
    real_manifest = paths.manifests / "real_all.csv"
    if real_manifest.exists():
        real_hashes = set(pd.read_csv(real_manifest)["sha256"].dropna())
        if real_hashes & set(all_ai["sha256"].dropna()):
            raise RuntimeError("Byte-identical REAL/AI files detected")
    save_manifest(paths, all_ai[all_ai.split == "validation"].copy(), "ai_validation")
    save_manifest(paths, all_ai[all_ai.split == "test"].copy(), "ai_test")
    save_manifest(paths, all_ai, "ai_all")
    return all_ai


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "<na>"}


def first_value(row: pd.Series, candidates: list[str]):
    for column in candidates:
        if column in row.index and not is_missing(row[column]):
            value = row[column]
            return value.strip() if isinstance(value, str) else value
    return None


def normalize_source(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if "fivek" in normalized:
        return "FiveK"
    if "fodb" in normalized:
        return "FODB"
    if "unsplash" in normalized:
        return "Unsplash"
    if "aigenimages2026" in normalized or "aigen2026" in normalized:
        return "AIGenImages2026"
    if "t2icorebench" in normalized:
        return "T2I-CoReBench"
    return text


def load_manifest_group(paths: DatasetPaths, kind: str) -> pd.DataFrame:
    all_path = paths.manifests / f"{kind}_all.csv"
    if all_path.exists():
        return pd.read_csv(all_path)
    if kind == "real":
        names = ["real_validation.csv", "real_test.csv"]
    else:
        source_names = ["ai_aigen2026.csv", "ai_t2icorebench.csv"]
        if all((paths.manifests / name).exists() for name in source_names):
            return pd.concat([pd.read_csv(paths.manifests / name) for name in source_names], ignore_index=True, sort=False)
        names = ["ai_validation.csv", "ai_test.csv"]
    files = [paths.manifests / name for name in names]
    if not all(path.exists() for path in files):
        raise RuntimeError(f"Missing {kind} manifests")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "split" not in frame.columns:
            frame["split"] = "validation" if "validation" in path.stem else "test"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_metadata(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_raw = load_manifest_group(paths, "real")
    ai_raw = load_manifest_group(paths, "ai")
    columns = ["image_name", "source", "generator", "generator_family", "label", "split", "source_id", "original_name", "sha256", "prompt_id", "prompt", "variant_id"]

    def canonicalize(row: pd.Series, label: int) -> dict:
        path_value = first_value(row, ["output_path", "filepath", "file_path", "path", "saved_path"])
        name_value = first_value(row, ["image_name", "output_filename", "output_name", "filename", "file_name"])
        image_name = Path(str(path_value or name_value)).name if (path_value or name_value) else None
        if not image_name:
            raise RuntimeError("Cannot determine image_name")
        source = normalize_source(first_value(row, ["source", "source_dataset", "dataset", "dataset_name"]))
        if source is None:
            raise RuntimeError(f"Cannot determine source for {image_name}")
        split_raw = str(first_value(row, ["split"])).strip().lower()
        split = {"val": "validation", "valid": "validation", "validation": "validation", "test": "test"}.get(split_raw)
        if split is None:
            raise RuntimeError(f"Unsupported split: {split_raw}")
        sha = str(first_value(row, ["sha256", "file_sha256", "image_sha256"]) or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise RuntimeError(f"Invalid SHA-256 for {image_name}")
        generator = None if label == 0 else first_value(row, ["generator", "model", "model_name"])
        family = None if label == 0 else GENERATOR_FAMILY.get(str(generator))
        if label == 1 and family is None:
            raise RuntimeError(f"Unknown generator family: {generator}")
        prompt_id = None if label == 0 else first_value(row, ["prompt_id", "caption_id"])
        prompt = None if label == 0 else first_value(row, ["prompt", "caption"])
        variant = None if label == 0 else first_value(row, ["variant_id", "requested_variant"])
        original = first_value(row, ["original_name", "original_filename", "source_filename", "archive_member", "tar_member", "zip_member", "member"])
        original = Path(str(original)).name if original is not None else None
        native_id = first_value(row, ["source_id", "image_id", "photo_id", "id", "uid"])
        if source == "T2I-CoReBench":
            source_id = f"T2I-CoReBench::{generator}::{prompt_id}::{int(float(variant))}"
        elif native_id is not None:
            source_id = str(native_id)
        else:
            source_id = f"{source}::{original}" if original else None
        return {
            "image_name": image_name, "source": source, "generator": generator, "generator_family": family,
            "label": label, "split": split, "source_id": source_id, "original_name": original, "sha256": sha,
            "prompt_id": prompt_id, "prompt": prompt, "variant_id": variant,
        }

    real = pd.DataFrame([canonicalize(row, 0) for _, row in real_raw.iterrows()], columns=columns)
    ai = pd.DataFrame([canonicalize(row, 1) for _, row in ai_raw.iterrows()], columns=columns)
    metadata = pd.concat([real, ai], ignore_index=True)
    metadata["label"] = metadata["label"].astype("int8")
    metadata["variant_id"] = pd.to_numeric(metadata["variant_id"], errors="coerce").astype("Int64")
    if len(metadata) != 5000 or metadata["image_name"].duplicated().any():
        raise RuntimeError("Metadata row or image-name census mismatch")
    expected = {("validation", 0): 1250, ("validation", 1): 1250, ("test", 0): 1250, ("test", 1): 1250}
    if metadata.groupby(["split", "label"]).size().to_dict() != expected:
        raise RuntimeError("Metadata split/label census mismatch")
    if set(metadata.loc[metadata.split == "validation", "sha256"]) & set(metadata.loc[metadata.split == "test", "sha256"]):
        raise RuntimeError("Metadata contains cross-split byte duplicates")
    validation = metadata[metadata.split == "validation"].sort_values(["label", "source", "generator", "image_name"], na_position="first").reset_index(drop=True)
    test = metadata[metadata.split == "test"].sort_values(["label", "source", "generator", "image_name"], na_position="first").reset_index(drop=True)
    validation.to_csv(paths.metadata / "validation.csv", index=False)
    test.to_csv(paths.metadata / "test.csv", index=False)
    return validation, test


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "GProject-DatasetBuilder/1.0"})
    return session


def build_real(paths: DatasetPaths, seed: int, workers: int, hf_token: str | None) -> pd.DataFrame:
    session = make_session()
    fivek = build_fivek(paths, session, seed, workers)
    fodb = build_fodb(paths, session, seed)
    unsplash = build_unsplash(paths, session, seed, workers, hf_token)
    return audit_real(paths, [fivek, fodb, unsplash])


def build_ai(paths: DatasetPaths, seed: int, hf_token: str | None, keep_cache: bool) -> pd.DataFrame:
    aigen = build_aigen(paths, seed, hf_token)
    t2i = build_t2i(paths, seed, hf_token, aigen, keep_cache)
    result = audit_ai(paths, aigen, t2i)
    if not keep_cache:
        shutil.rmtree(paths.cache / "hf" / "aigenimages2026", ignore_errors=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the validation/test image dataset and canonical metadata.")
    parser.add_argument("command", choices=["real", "ai", "metadata", "all"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("GPROJECT_CACHE", "~/.cache/gproject-datasets")).expanduser())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.hf_token:
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
    configure(args.seed)
    paths = DatasetPaths(args.output_dir.expanduser().resolve(), args.cache_dir.expanduser().resolve())
    paths.create()
    if args.command in {"real", "all"}:
        build_real(paths, args.seed, args.workers, args.hf_token)
    if args.command in {"ai", "all"}:
        build_ai(paths, args.seed, args.hf_token, args.keep_cache)
    if args.command in {"metadata", "all"}:
        build_metadata(paths)


if __name__ == "__main__":
    main()
