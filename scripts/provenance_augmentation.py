from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("provenance")
PROVENANCE_VERSION = "prov-valtest-v2"
PROTOCOL_PROFILE = "natural-c2pa-audit-mask+fixed-generator-quota"
NATURAL_C2PA_POLICY = "audit_and_mask"
SCENARIO_SALT = "scenario"
MASK_SALT = "mask"
SUBVARIANT_SALT = "subvariant"
SPLITS = ("validation", "test")

@dataclass(frozen=True)
class BuildConfig:
    dataset_root: Path
    image_dirs: dict[str, Path]
    metadata_dir: Path
    unsplash_manifest: Path
    output_root: Path
    exiftool: str = "exiftool"
    exiftool_batch_size: int = 80
    workers: int = 4
    seed: int = 42
    dry_run: bool = False
    replace_existing: bool = False

    @property
    def output_metadata_dir(self) -> Path:
        return self.output_root / "metadata"

    @property
    def provenance_dir(self) -> Path:
        return self.output_root / "provenance"

PROVENANCE_COLUMNS = [
    'provenance_id',
    'provenance_scenario',
    'provenance_available',
    'provenance_mask_profile',
]

REQUIRED_METADATA_COLUMNS = [
    'image_name', 'source', 'generator', 'generator_family', 'label', 'split',
    'source_id', 'original_name', 'sha256', 'prompt_id', 'prompt', 'variant_id',
]

REAL_SOURCES = ('FiveK', 'FODB', 'Unsplash')
AI_SOURCES = ('AIGenImages2026', 'T2I-CoReBench')
ALL_SOURCES = REAL_SOURCES + AI_SOURCES

EXPECTED_SOURCE_COUNTS = {
    'FiveK': 500,
    'FODB': 400,
    'Unsplash': 350,
    'AIGenImages2026': 750,
    'T2I-CoReBench': 500,
}
EXPECTED_SOURCE_LABEL = {
    'FiveK': 0,
    'FODB': 0,
    'Unsplash': 0,
    'AIGenImages2026': 1,
    'T2I-CoReBench': 1,
}
EXPECTED_GENERATOR_COUNTS = {
    'AIGenImages2026': 6,
    'T2I-CoReBench': 4,
}

SCENARIO_ORDER = ('P0', 'P1', 'P2', 'P3', 'P4')
SOURCE_SCENARIO_QUOTAS = {
    'FiveK': {'P0': 300, 'P1': 100, 'P2': 52, 'P3': 28, 'P4': 20},
    'FODB': {'P0': 240, 'P1': 80, 'P2': 42, 'P3': 22, 'P4': 16},
    'Unsplash': {'P0': 210, 'P1': 70, 'P2': 36, 'P3': 20, 'P4': 14},
    'AIGenImages2026': {'P0': 450, 'P1': 150, 'P2': 78, 'P3': 42, 'P4': 30},
    'T2I-CoReBench': {'P0': 300, 'P1': 100, 'P2': 52, 'P3': 28, 'P4': 20},
}
AI_SCENARIO_PATTERN = {'P0': 75, 'P1': 25, 'P2': 13, 'P3': 7, 'P4': 5}

MASK_ORDER = ('minimal', 'standard', 'leaky')
SOURCE_MASK_QUOTAS = {
    'FiveK': {'minimal': 140, 'standard': 52, 'leaky': 8},
    'FODB': {'minimal': 112, 'standard': 42, 'leaky': 6},
    'Unsplash': {'minimal': 98, 'standard': 36, 'leaky': 6},
    'AIGenImages2026': {'minimal': 210, 'standard': 78, 'leaky': 12},
    'T2I-CoReBench': {'minimal': 140, 'standard': 52, 'leaky': 8},
}
AI_MASK_PATTERN = {'minimal': 35, 'standard': 13, 'leaky': 2}

EFFECTIVE_METADATA_FIELDS = [
    'metadata_present',
    'exif_present',
    'xmp_present',
    'camera_identity_present',
    'capture_settings_present',
    'software_metadata_present',
    'camera_make_present',
    'camera_model_present',
    'lens_model_present',
    'iso',
    'f_number',
    'exposure_time_s',
    'focal_length_mm',
    'datetime_present',
    'gps_present',
    'dng_present',
    'dng_color_matrix_present',
    'dng_camera_calibration_present',
    'dng_measured_rggb_present',
    'dng_black_level_present',
    'dng_white_level_present',
    'dng_cfa_present',
    'dng_version_present',
]

C2PA_FIELDS = [
    'present',
    'validation_state',
    'active_action',
    'action_created',
    'action_opened',
    'action_edited',
    'origin_resolved',
    'origin_source_type',
    'has_parent',
    'parent_manifest_available',
    'unknown_provenance',
    'manifest_count',
    'ingredient_count',
    'chain_depth',
]

MINIMAL_METADATA_FIELDS = {
    'metadata_present', 'exif_present', 'xmp_present',
    'camera_identity_present', 'capture_settings_present', 'software_metadata_present',
}
STANDARD_METADATA_FIELDS = MINIMAL_METADATA_FIELDS | {
    'camera_make_present', 'camera_model_present', 'lens_model_present',
    'iso', 'f_number', 'exposure_time_s', 'focal_length_mm',
    'datetime_present', 'gps_present',
}
LEAKY_METADATA_FIELDS = STANDARD_METADATA_FIELDS | {
    'dng_present', 'dng_color_matrix_present', 'dng_camera_calibration_present',
    'dng_measured_rggb_present', 'dng_black_level_present',
    'dng_white_level_present', 'dng_cfa_present', 'dng_version_present',
}

MINIMAL_C2PA_FIELDS = {
    'present', 'validation_state', 'active_action', 'origin_resolved', 'origin_source_type',
}
STANDARD_C2PA_FIELDS = set(C2PA_FIELDS)

FORBIDDEN_EFFECTIVE_KEYS = {
    'label', 'split', 'source', 'source_id', 'image_name', 'original_name', 'sha256',
    'generator', 'generator_family', 'prompt', 'prompt_id', 'variant_id',
    'file_path', 'directory', 'url', 'repository', 'hf_repo', 'dataset_name',
    'exact_gps_latitude', 'exact_gps_longitude', 'exact_datetime',
    'camera_make_value', 'camera_model_value', 'lens_model_value',
    'software_value', 'creator_tool_value',
    'c2pa_signer_identity', 'c2pa_software_agent_name', 'c2pa_model_name',
    'c2pa_model_identifier', 'provider_name',
}

PROVENANCE_NUMERIC_FIELDS_V1 = [
    'available',
    'c2pa_present', 'c2pa_valid', 'c2pa_origin_resolved',
    'c2pa_origin_digital_capture', 'c2pa_origin_trained_algorithmic_media',
    'c2pa_action_created', 'c2pa_action_opened', 'c2pa_action_edited',
    'c2pa_has_parent', 'c2pa_parent_manifest_available', 'c2pa_unknown_provenance',
    'c2pa_manifest_count', 'c2pa_ingredient_count', 'c2pa_chain_depth',
    'metadata_present', 'exif_present', 'xmp_present', 'camera_identity_present',
    'capture_settings_present', 'software_metadata_present',
    'camera_make_present', 'camera_model_present', 'lens_model_present',
    'iso', 'f_number', 'exposure_time_s', 'focal_length_mm',
    'datetime_present', 'gps_present',
    'dng_present', 'dng_color_matrix_present', 'dng_camera_calibration_present',
    'dng_measured_rggb_present', 'dng_black_level_present', 'dng_white_level_present',
    'dng_cfa_present', 'dng_version_present',
]



def rank_key(*parts: str) -> str:
    return hashlib.sha256('|'.join(str(x) for x in parts).encode('utf-8')).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def image_path(split: str, image_name: str, config: BuildConfig) -> Path:
    return config.image_dirs[split] / image_name


def load_split_metadata(split: str, config: BuildConfig) -> pd.DataFrame:
    path = config.metadata_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f'{split}: missing metadata columns: {missing}')

    existing_prov = [c for c in PROVENANCE_COLUMNS if c in df.columns]
    if existing_prov:
        df = df.drop(columns=existing_prov)

    df['label'] = pd.to_numeric(df['label'], errors='raise').astype('int8')

    if len(df) != 2500:
        raise RuntimeError(f'{split}: expected 2500 rows, found {len(df)}')
    if df['image_name'].duplicated().any():
        dup = df.loc[df['image_name'].duplicated(keep=False), 'image_name'].tolist()[:10]
        raise RuntimeError(f'{split}: duplicate image_name: {dup}')
    unsafe = [name for name in df['image_name'] if Path(name).name != name or name in {'.', '..'}]
    if unsafe:
        raise ValueError(f'{split}: image_name must be a basename: {unsafe[:5]}')
    if set(df['split']) != {split}:
        raise RuntimeError(f'{split}: unexpected split values: {sorted(set(df["split"]))}')
    if (df['label'] == 0).sum() != 1250 or (df['label'] == 1).sum() != 1250:
        raise RuntimeError(f'{split}: label balance mismatch')

    actual_sources = df.groupby('source').size().to_dict()
    if actual_sources != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f'{split}: source composition mismatch: {actual_sources}')

    for source, expected_label in EXPECTED_SOURCE_LABEL.items():
        labels = set(df.loc[df['source'] == source, 'label'].astype(int))
        if labels != {expected_label}:
            raise RuntimeError(f'{split}: source {source} has labels {labels}, expected {expected_label}')

    for source, n_generators in EXPECTED_GENERATOR_COUNTS.items():
        part = df[df['source'] == source]
        counts = part.groupby('generator').size().to_dict()
        if len(counts) != n_generators or set(counts.values()) != {125}:
            raise RuntimeError(f'{split}: generator census mismatch for {source}: {counts}')

    bad_sha = df[~df['sha256'].str.fullmatch(r'[0-9a-fA-F]{64}')]
    if len(bad_sha):
        raise RuntimeError(f'{split}: invalid SHA-256 rows: {bad_sha["image_name"].head().tolist()}')

    missing_files = [name for name in df['image_name'] if not image_path(split, name, config).is_file()]
    if missing_files:
        raise RuntimeError(f'{split}: missing image files: {missing_files[:10]}')

    return df.reset_index(drop=True)


def verify_image_hashes(
    metadata_by_split: dict[str, pd.DataFrame], config: BuildConfig
) -> dict[tuple[str, str], str]:
    jobs = []
    for split, df in metadata_by_split.items():
        for row in df[['image_name', 'sha256']].to_dict(orient='records'):
            jobs.append((split, row['image_name'], row['sha256'].lower()))

    out: dict[tuple[str, str], str] = {}
    failures = []

    def one(job):
        split, name, expected = job
        got = sha256_file(image_path(split, name, config))
        return split, name, expected, got

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for fut in as_completed(futures):
            split, name, expected, got = fut.result()
            out[(split, name)] = got
            if got.lower() != expected:
                failures.append((split, name, expected, got))

    if failures:
        raise RuntimeError(f'Image hash verification failed; first failures: {failures[:5]}')
    return out


def norm_token(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def leaf_key(key: str) -> str:
    return str(key).rsplit(':', 1)[-1]


def valid_scalar(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and value.strip().lower() in {'', 'none', 'null', 'nan', '-'}:
        return False
    return True


def find_tag(tags: dict, aliases: list[str]) -> Any:
    by_leaf = defaultdict(list)
    for key, value in tags.items():
        by_leaf[norm_token(leaf_key(key))].append(value)
    for alias in aliases:
        for value in by_leaf.get(norm_token(alias), []):
            if valid_scalar(value):
                return value
    return None


def to_float(value: Any) -> float | None:
    if not valid_scalar(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip()
    try:
        if re.fullmatch(r'[+-]?\d+\s*/\s*\d+', text):
            return float(Fraction(text.replace(' ', '')))
        result = float(text)
        return result if math.isfinite(result) else None
    except Exception:
        match = re.search(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', text)
        if match:
            try:
                result = float(match.group(0))
                return result if math.isfinite(result) else None
            except Exception:
                pass
    return None


def exif_group(key: str) -> str:
    return str(key).split(':', 1)[0]


def normalize_exif(tags: dict) -> dict:
    make = find_tag(tags, ['Make'])
    model = find_tag(tags, ['Model', 'CameraModelName'])
    lens = find_tag(tags, ['LensModel', 'LensID', 'LensType'])
    iso = to_float(find_tag(tags, ['ISO', 'PhotographicSensitivity', 'ISOSetting']))
    f_number = to_float(find_tag(tags, ['FNumber', 'Aperture']))
    exposure = to_float(find_tag(tags, ['ExposureTime']))
    focal = to_float(find_tag(tags, ['FocalLength']))
    software = find_tag(tags, ['Software'])
    dt = find_tag(tags, ['DateTimeOriginal'])

    gps_leaves = {
        'gpslatitude', 'gpslongitude', 'gpsposition', 'gpslatituderef',
        'gpslongituderef', 'gpsaltitude', 'gpsdestlatitude', 'gpsdestlongitude',
    }
    gps_present = any(
        norm_token(leaf_key(k)) in gps_leaves and valid_scalar(v)
        for k, v in tags.items()
    )

    canonical_field_present = any([
        make is not None, model is not None, lens is not None,
        iso is not None, f_number is not None, exposure is not None, focal is not None,
        software is not None, dt is not None, gps_present,
    ])
    explicit_exif_group = any(
        exif_group(k).lower() in {'exif', 'exififd', 'gps', 'interopifd'}
        for k in tags
    )
    present = bool(canonical_field_present or explicit_exif_group)

    return {
        'present': present,
        'make_present': make is not None,
        'model_present': model is not None,
        'lens_model_present': lens is not None,
        'iso': iso,
        'f_number': f_number,
        'exposure_time_s': exposure,
        'focal_length_mm': focal,
        'software_present': software is not None,
        'software_raw': None if software is None else str(software),
        'datetime_original_present': dt is not None,
        'gps_present': bool(gps_present),
        'make_raw': None if make is None else str(make),
        'model_raw': None if model is None else str(model),
        'lens_model_raw': None if lens is None else str(lens),
        'natural_origin': 'embedded' if present else 'none',
    }


def normalize_xmp(tags: dict) -> dict:
    xmp_keys = [k for k in tags if exif_group(k).lower().startswith('xmp')]
    creator = find_tag(tags, ['CreatorTool'])
    return {
        'present': bool(xmp_keys),
        'creator_tool_present': creator is not None,
        'creator_tool_raw': None if creator is None else str(creator),
    }


def any_tag_present(tags: dict, aliases: list[str]) -> bool:
    alias_tokens = {norm_token(a) for a in aliases}
    return any(norm_token(leaf_key(k)) in alias_tokens and valid_scalar(v) for k, v in tags.items())


def normalize_dng(tags: dict, path: Path) -> dict:
    dng_version_present = any_tag_present(tags, ['DNGVersion'])
    present = path.suffix.lower() == '.dng' or dng_version_present
    return {
        'present': bool(present),
        'color_matrix_present': any_tag_present(tags, ['ColorMatrix1', 'ColorMatrix2', 'ColorMatrix3']),
        'camera_calibration_present': any_tag_present(
            tags, ['CameraCalibration1', 'CameraCalibration2', 'CameraCalibration3']
        ),
        'measured_rggb_present': any_tag_present(
            tags, ['WB_RGGBLevelsMeasured', 'MeasuredRGGB', 'MeasuredRGGBLevels']
        ),
        'black_level_present': any_tag_present(tags, ['BlackLevel', 'BlackLevelRepeatDim']),
        'white_level_present': any_tag_present(tags, ['WhiteLevel']),
        'cfa_present': any_tag_present(tags, ['CFAPattern', 'CFAPlaneColor', 'CFARepeatPatternDim']),
        'dng_version_present': bool(dng_version_present),
    }


def exiftool_c2pa_signal(tags: dict) -> bool:
    needles = ('c2pa', 'jumbf', 'contentcredential')
    for key, value in tags.items():
        if leaf_key(key).lower() in {'sourcefile', 'filename', 'directory'}:
            continue
        key_norm = norm_token(key)
        if any(n in key_norm for n in needles):
            return True
        if isinstance(value, str):
            value_norm = norm_token(value[:4096])
            if any(n in value_norm for n in needles):
                return True
    return False


def run_exiftool(paths: list[Path], *, batch_size: int, exiftool: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        cmd = [exiftool, '-json', '-G1', '-n', '-api', 'LargeFileSupport=1', *[str(p) for p in batch]]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'ExifTool failed:\n{proc.stderr[-4000:]}')
        try:
            rows = json.loads(proc.stdout)
        except Exception as exc:
            raise RuntimeError(f'Invalid ExifTool JSON: {exc}') from exc
        if len(rows) != len(batch):
            raise RuntimeError(f'ExifTool returned {len(rows)} rows for {len(batch)} files')
        for item in rows:
            source_file = str(item.get('SourceFile', ''))
            if not source_file:
                raise RuntimeError('ExifTool result missing SourceFile')
            errors = [v for k, v in item.items() if norm_token(leaf_key(k)) == 'error' and valid_scalar(v)]
            if errors:
                raise RuntimeError(f'ExifTool error for {source_file}: {errors}')
            result[source_file] = item
    return result


def load_unsplash_catalog(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        'source_id', 'camera_make', 'camera_model',
        'exif_iso_catalog', 'exif_aperture_value_catalog',
        'exif_focal_length_catalog', 'exif_exposure_time_catalog',
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Unsplash manifest missing columns: {sorted(missing)}')
    if df['source_id'].duplicated().any():
        raise RuntimeError('Unsplash manifest contains duplicate source_id')
    return {str(row['source_id']): row for row in df.to_dict(orient='records')}


def catalog_exif(row: dict) -> dict:
    make = row.get('camera_make') or None
    model = row.get('camera_model') or None
    iso = to_float(row.get('exif_iso_catalog'))
    f_number = to_float(row.get('exif_aperture_value_catalog'))
    focal = to_float(row.get('exif_focal_length_catalog'))
    exposure = to_float(row.get('exif_exposure_time_catalog'))
    present = any(x is not None for x in [make, model, iso, f_number, focal, exposure])
    return {
        'present': bool(present),
        'make_present': make is not None,
        'model_present': model is not None,
        'lens_model_present': False,
        'iso': iso,
        'f_number': f_number,
        'exposure_time_s': exposure,
        'focal_length_mm': focal,
        'software_present': False,
        'software_raw': None,
        'datetime_original_present': False,
        'gps_present': False,
        'make_raw': None if make is None else str(make),
        'model_raw': None if model is None else str(model),
        'lens_model_raw': None,
        'natural_origin': 'catalog' if present else 'none',
    }


def merge_embedded_and_catalog_exif(embedded: dict, catalog: dict) -> dict:
    out = copy.deepcopy(embedded)
    embedded_present = bool(embedded.get('present'))
    catalog_present = bool(catalog.get('present'))

    for present_key in ['make_present', 'model_present', 'lens_model_present', 'software_present',
                        'datetime_original_present', 'gps_present']:
        out[present_key] = bool(embedded.get(present_key) or catalog.get(present_key))

    for value_key in ['iso', 'f_number', 'exposure_time_s', 'focal_length_mm',
                      'software_raw', 'make_raw', 'model_raw', 'lens_model_raw']:
        if out.get(value_key) is None and catalog.get(value_key) is not None:
            out[value_key] = catalog[value_key]

    out['present'] = bool(embedded_present or catalog_present)
    if embedded_present and catalog_present:
        out['natural_origin'] = 'embedded+catalog'
    elif catalog_present:
        out['natural_origin'] = 'catalog'
    elif embedded_present:
        out['natural_origin'] = 'embedded'
    else:
        out['natural_origin'] = 'none'
    return out


def make_c2pa_context(c2pa_module):
    return c2pa_module.Context.from_dict({
        'verify': {
            'remote_manifest_fetch': False,
            'ocsp_fetch': False,
        }
    })


def audit_natural_c2pa(
    path: Path, *, exiftool_signal: bool, context, c2pa_module
) -> dict:
    """Record C2PA presence without treating manifest discovery as trust validation."""
    absent = {
        'present': False,
        'parser': 'c2pa-python',
        'parser_status': 'no_manifest_or_unsupported',
        'embedded': False,
        'remote_manifest_reference_present': False,
    }
    entered = False
    try:
        with c2pa_module.Reader(str(path), context=context) as reader:
            entered = True
            payload = json.loads(reader.json())
            if not isinstance(payload, dict):
                raise ValueError('C2PA SDK returned a non-object payload')
            active = payload.get('active_manifest')
            manifests = payload.get('manifests') or {}
            embedded = bool(reader.is_embedded())
            remote_ref = reader.get_remote_url() is not None
            present = bool(active or manifests or embedded or remote_ref)
            return {
                'present': present,
                'parser': 'c2pa-python',
                'parser_status': 'manifest_present' if present else 'no_manifest',
                'embedded': embedded,
                'remote_manifest_reference_present': bool(remote_ref),
            }
    except AttributeError as exc:
        raise RuntimeError(f'C2PA SDK API is incompatible: {path}') from exc
    except Exception as exc:
        if entered or exiftool_signal:
            raise RuntimeError(f'C2PA could not be read or validated for audit: {path}') from exc
        return absent


def extract_natural_split(
    df: pd.DataFrame,
    split: str,
    unsplash_catalog: dict[str, dict],
    config: BuildConfig,
    c2pa_module,
) -> dict[str, dict]:
    ordered = df.sort_values('image_name').reset_index(drop=True)
    paths = [image_path(split, name, config) for name in ordered['image_name']]
    tags_by_path = run_exiftool(paths, batch_size=config.exiftool_batch_size, exiftool=config.exiftool)

    naturals: dict[str, dict] = {}
    context = make_c2pa_context(c2pa_module)
    try:
        iterator = ordered.to_dict(orient='records')
        for row in iterator:
            path = image_path(split, row['image_name'], config)
            if str(path) not in tags_by_path:
                raise RuntimeError(f'ExifTool output missing image: {path}')
            tags = tags_by_path[str(path)]
            exif = normalize_exif(tags)
            xmp = normalize_xmp(tags)
            dng = normalize_dng(tags, path)

            if row['source'] == 'Unsplash':
                source_id = str(row['source_id'])
                if source_id not in unsplash_catalog:
                    raise RuntimeError(f'Unsplash catalog join missing source_id={source_id}')
                exif = merge_embedded_and_catalog_exif(exif, catalog_exif(unsplash_catalog[source_id]))

            c2pa_info = audit_natural_c2pa(
                path,
                exiftool_signal=exiftool_c2pa_signal(tags),
                context=context,
                c2pa_module=c2pa_module,
            )

            naturals[row['image_name']] = {
                'c2pa': c2pa_info,
                'exif': exif,
                'xmp': xmp,
                'dng': dng,
            }
    finally:
        close = getattr(context, 'close', None)
        if callable(close):
            close()

    if len(naturals) != len(df):
        raise RuntimeError(f'{split}: natural provenance join mismatch')

    return naturals


def assign_slice(
    items: list[dict], quotas: dict[str, int], order: tuple[str, ...], key_name: str
) -> dict[str, str]:
    if sum(quotas.values()) != len(items):
        raise RuntimeError(f'Quota sum {sum(quotas.values())} != group size {len(items)}')
    out = {}
    pos = 0
    for category in order:
        n = quotas[category]
        for item in items[pos:pos+n]:
            out[item[key_name]] = category
        pos += n
    if pos != len(items):
        raise RuntimeError('Slice assignment did not consume all rows')
    return out


def assign_scenarios(df: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    assignments = {}
    rank_keys = {}

    for row in df.to_dict(orient='records'):
        rank_keys[row['image_name']] = rank_key(
            PROVENANCE_VERSION, seed, SCENARIO_SALT,
            split, row['source'], row['image_name'],
        )

    for source in REAL_SOURCES:
        rows = df[df['source'] == source].to_dict(orient='records')
        rows = sorted(rows, key=lambda r: (rank_keys[r['image_name']], r['image_name']))
        assigned = assign_slice(rows, SOURCE_SCENARIO_QUOTAS[source], SCENARIO_ORDER, 'image_name')
        assignments.update(assigned)

    for source in AI_SOURCES:
        generators = sorted(set(df.loc[df['source'] == source, 'generator']))
        for generator in generators:
            rows = df[(df['source'] == source) & (df['generator'] == generator)].to_dict(orient='records')
            if len(rows) != 125:
                raise RuntimeError(f'{split}/{source}/{generator}: expected 125 rows')
            rows = sorted(rows, key=lambda r: (rank_keys[r['image_name']], r['image_name']))
            assignments.update(assign_slice(rows, AI_SCENARIO_PATTERN, SCENARIO_ORDER, 'image_name'))

    if len(assignments) != len(df):
        raise RuntimeError(f'{split}: scenario assignment coverage mismatch')

    out = df[['image_name', 'source', 'generator', 'label']].copy()
    out['scenario'] = out['image_name'].map(assignments)
    out['scenario_rank_key'] = out['image_name'].map(rank_keys)
    return out


def assign_masks(df: pd.DataFrame, split: str, scenario_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    work = scenario_df.copy()
    work['mask_profile'] = 'none'
    work['mask_rank_key'] = None
    assigned_profiles = {}

    mask_rank = {}
    for row in df.to_dict(orient='records'):
        mask_rank[row['image_name']] = rank_key(
            PROVENANCE_VERSION, seed, MASK_SALT,
            split, row['source'], row['image_name'],
        )

    for source in REAL_SOURCES:
        names = set(work[(work['source'] == source) & (work['scenario'] != 'P0')]['image_name'])
        rows = [r for r in df[df['source'] == source].to_dict(orient='records') if r['image_name'] in names]
        rows = sorted(rows, key=lambda r: (mask_rank[r['image_name']], r['image_name']))
        assigned = assign_slice(rows, SOURCE_MASK_QUOTAS[source], MASK_ORDER, 'image_name')
        for name, profile in assigned.items():
            assigned_profiles[name] = profile

    for source in AI_SOURCES:
        generators = sorted(set(df.loc[df['source'] == source, 'generator']))
        for generator in generators:
            available_names = set(work[
                (work['source'] == source) &
                (work['generator'] == generator) &
                (work['scenario'] != 'P0')
            ]['image_name'])
            rows = [r for r in df[
                (df['source'] == source) & (df['generator'] == generator)
            ].to_dict(orient='records') if r['image_name'] in available_names]
            if len(rows) != 50:
                raise RuntimeError(f'{split}/{source}/{generator}: expected 50 available records')
            rows = sorted(rows, key=lambda r: (mask_rank[r['image_name']], r['image_name']))
            assigned = assign_slice(rows, AI_MASK_PATTERN, MASK_ORDER, 'image_name')
            for name, profile in assigned.items():
                assigned_profiles[name] = profile

    work['mask_profile'] = work['image_name'].map(assigned_profiles).fillna('none')
    work['mask_rank_key'] = work['image_name'].map(mask_rank).where(work['scenario'] != 'P0', None)
    if ((work['scenario'] == 'P0') & (work['mask_profile'] != 'none')).any():
        raise RuntimeError(f'{split}: P0 has non-none mask profile')
    if ((work['scenario'] != 'P0') & (~work['mask_profile'].isin(MASK_ORDER))).any():
        raise RuntimeError(f'{split}: available record missing mask profile')
    return work


def empty_c2pa() -> dict:
    return {k: None for k in C2PA_FIELDS}


def c2pa_absent() -> dict:
    out = empty_c2pa()
    out['present'] = False
    return out


def build_synthetic_c2pa(scenario: str, label: int) -> dict | None:
    if scenario not in {'P2', 'P3', 'P4'}:
        return None

    origin = 'digitalCapture' if int(label) == 0 else 'trainedAlgorithmicMedia'
    out = empty_c2pa()
    out['present'] = True
    out['validation_state'] = 'valid'

    if scenario == 'P2':
        out.update({
            'active_action': 'created',
            'action_created': True,
            'action_opened': False,
            'action_edited': False,
            'origin_resolved': True,
            'origin_source_type': origin,
            'has_parent': False,
            'parent_manifest_available': False,
            'unknown_provenance': False,
            'manifest_count': 1,
            'ingredient_count': 0,
            'chain_depth': 1,
        })
    elif scenario == 'P3':
        out.update({
            'active_action': 'opened',
            'action_created': False,
            'action_opened': True,
            'action_edited': True,
            'origin_resolved': True,
            'origin_source_type': origin,
            'has_parent': True,
            'parent_manifest_available': True,
            'unknown_provenance': False,
            'manifest_count': 2,
            'ingredient_count': 1,
            'chain_depth': 2,
        })
    else:
        out.update({
            'active_action': 'opened',
            'action_created': False,
            'action_opened': True,
            'action_edited': True,
            'origin_resolved': False,
            'origin_source_type': None,
            'has_parent': True,
            'parent_manifest_available': False,
            'unknown_provenance': True,
            'manifest_count': 1,
            'ingredient_count': 1,
            'chain_depth': 1,
        })
    return out


def ai_p1_neutral_metadata(row: dict, seed: int) -> dict:
    variant = int(rank_key(
        PROVENANCE_VERSION, seed, SUBVARIANT_SALT,
        row['split'], row['source'], row['image_name'],
    ), 16) % 2
    if variant == 0:
        return {
            'template': 'A',
            'exif_present': True,
            'xmp_present': True,
            'camera_identity_present': False,
            'capture_settings_present': False,
            'software_metadata_present': True,
            'datetime_present': True,
            'gps_present': False,
        }
    return {
        'template': 'B',
        'exif_present': False,
        'xmp_present': True,
        'camera_identity_present': False,
        'capture_settings_present': False,
        'software_metadata_present': True,
        'datetime_present': False,
        'gps_present': False,
    }


def build_synthetic_provenance(scenario: str, row: dict, seed: int) -> dict:
    label = int(row['label'])
    if scenario == 'P0':
        return {'applied': False, 'kind': None, 'c2pa': None, 'metadata': None}
    if scenario == 'P1':
        if label == 1:
            return {
                'applied': True,
                'kind': 'neutral_non_c2pa_metadata',
                'c2pa': None,
                'metadata': ai_p1_neutral_metadata(row, seed),
            }
        return {'applied': False, 'kind': None, 'c2pa': None, 'metadata': None}
    return {
        'applied': True,
        'kind': 'canonical_c2pa_simulation',
        'c2pa': build_synthetic_c2pa(scenario, label),
        'metadata': None,
    }


def natural_to_candidate_metadata(natural: dict) -> dict:
    exif = natural['exif']
    xmp = natural['xmp']
    dng = natural['dng']
    numeric = [
        exif.get('iso'), exif.get('f_number'),
        exif.get('exposure_time_s'), exif.get('focal_length_mm'),
    ]
    out = {
        'metadata_present': bool(exif.get('present') or xmp.get('present') or dng.get('present')),
        'exif_present': bool(exif.get('present')),
        'xmp_present': bool(xmp.get('present')),
        'camera_identity_present': bool(
            exif.get('make_present') or exif.get('model_present')
            or exif.get('lens_model_present')
        ),
        'capture_settings_present': any(v is not None for v in numeric),
        'software_metadata_present': bool(exif.get('software_present') or xmp.get('creator_tool_present')),
        'camera_make_present': bool(exif.get('make_present')),
        'camera_model_present': bool(exif.get('model_present')),
        'lens_model_present': bool(exif.get('lens_model_present')),
        'iso': exif.get('iso'),
        'f_number': exif.get('f_number'),
        'exposure_time_s': exif.get('exposure_time_s'),
        'focal_length_mm': exif.get('focal_length_mm'),
        'datetime_present': bool(exif.get('datetime_original_present')),
        'gps_present': bool(exif.get('gps_present')),
        'dng_present': bool(dng.get('present')),
        'dng_color_matrix_present': bool(dng.get('color_matrix_present')),
        'dng_camera_calibration_present': bool(dng.get('camera_calibration_present')),
        'dng_measured_rggb_present': bool(dng.get('measured_rggb_present')),
        'dng_black_level_present': bool(dng.get('black_level_present')),
        'dng_white_level_present': bool(dng.get('white_level_present')),
        'dng_cfa_present': bool(dng.get('cfa_present')),
        'dng_version_present': bool(dng.get('dng_version_present')),
    }
    if list(out.keys()) != EFFECTIVE_METADATA_FIELDS:
        raise AssertionError('Candidate metadata schema/order drift')
    return out


def merge_neutral_p1_ai(candidate: dict, neutral: dict) -> dict:
    out = copy.deepcopy(candidate)
    for key in [
        'exif_present', 'xmp_present', 'camera_identity_present',
        'capture_settings_present', 'software_metadata_present',
        'datetime_present', 'gps_present',
    ]:
        out[key] = bool(out.get(key) or neutral.get(key))
    out['metadata_present'] = bool(
        out['metadata_present'] or out['exif_present'] or out['xmp_present'] or
        out['camera_identity_present'] or out['capture_settings_present'] or
        out['software_metadata_present'] or out['dng_present']
    )
    return out


def merge_natural_and_synthetic(natural: dict, synthetic: dict, scenario: str, row: dict) -> dict:
    metadata = natural_to_candidate_metadata(natural)
    if scenario == 'P1' and int(row['label']) == 1:
        metadata = merge_neutral_p1_ai(metadata, synthetic['metadata'])

    if scenario == 'P0':
        c2pa_view = None
    elif scenario == 'P1':
        c2pa_view = c2pa_absent()
    else:
        c2pa_view = copy.deepcopy(synthetic['c2pa'])

    return {
        'available': scenario != 'P0',
        'c2pa': c2pa_view,
        'metadata': metadata,
    }


def mask_dict(candidate: dict, fields: list[str], exposed: set[str]) -> dict:
    return {key: (candidate.get(key) if key in exposed else None) for key in fields}


def apply_mask_profile(candidate: dict, scenario: str, profile: str) -> dict:
    if scenario == 'P0':
        if profile != 'none':
            raise RuntimeError('P0 must use mask_profile=none')
        return {'available': False, 'c2pa': None, 'metadata': None}

    if profile not in MASK_ORDER:
        raise RuntimeError(f'Invalid available mask profile: {profile}')

    metadata_exposed = {
        'minimal': MINIMAL_METADATA_FIELDS,
        'standard': STANDARD_METADATA_FIELDS,
        'leaky': LEAKY_METADATA_FIELDS,
    }[profile]
    c2pa_exposed = MINIMAL_C2PA_FIELDS if profile == 'minimal' else STANDARD_C2PA_FIELDS

    return {
        'available': True,
        'c2pa': mask_dict(candidate['c2pa'], C2PA_FIELDS, c2pa_exposed),
        'metadata': mask_dict(candidate['metadata'], EFFECTIVE_METADATA_FIELDS, metadata_exposed),
    }


def normalize_forbidden_key(key: str) -> str:
    return str(key).strip().lower()


def build_global_forbidden_literals(
    metadata_by_split: dict[str, pd.DataFrame], config: BuildConfig
) -> set[str]:
    values = set(ALL_SOURCES) | {'validation', 'test', str(config.dataset_root)}
    for df in metadata_by_split.values():
        for column in ['generator', 'generator_family']:
            values.update(v for v in df[column].astype(str) if v.strip())
    return {v.casefold() for v in values if v and len(v) >= 3}


def forbidden_literals_for_row(row: dict, global_literals: set[str]) -> set[str]:
    vals = set(global_literals)
    for key in [
        'image_name', 'source', 'generator', 'generator_family', 'split', 'source_id',
        'original_name', 'sha256', 'prompt_id', 'prompt', 'variant_id',
    ]:
        value = str(row.get(key, '')).strip()
        if len(value) >= 3:
            vals.add(value.casefold())
    return vals


def scan_effective_forbidden(effective: Any, row: dict, global_literals: set[str]) -> list[str]:
    violations = []
    literals = forbidden_literals_for_row(row, global_literals)

    def walk(obj: Any, path: str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if normalize_forbidden_key(key) in FORBIDDEN_EFFECTIVE_KEYS:
                    violations.append(f'{path}.{key}: forbidden key')
                walk(value, f'{path}.{key}')
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f'{path}[{i}]')
        elif isinstance(obj, str):
            text = obj.casefold()
            for needle in literals:
                if needle and needle in text:
                    violations.append(f'{path}: forbidden literal {needle!r}')
                    break

    walk(effective, 'effective')
    return violations


def assert_mask_semantics(effective: dict, profile: str):
    if not effective['available']:
        return
    metadata = effective['metadata']
    c2 = effective['c2pa']

    expected_meta = {
        'minimal': MINIMAL_METADATA_FIELDS,
        'standard': STANDARD_METADATA_FIELDS,
        'leaky': LEAKY_METADATA_FIELDS,
    }[profile]
    expected_c2 = MINIMAL_C2PA_FIELDS if profile == 'minimal' else STANDARD_C2PA_FIELDS

    for key in EFFECTIVE_METADATA_FIELDS:
        if key not in expected_meta and metadata[key] is not None:
            raise AssertionError(f'{profile}: masked metadata field leaked: {key}')
    for key in C2PA_FIELDS:
        if key not in expected_c2 and c2[key] is not None:
            raise AssertionError(f'{profile}: masked C2PA field leaked: {key}')

    if profile != 'leaky':
        dng_fields = [x for x in EFFECTIVE_METADATA_FIELDS if x.startswith('dng_')]
        if any(metadata[k] is not None for k in dng_fields):
            raise AssertionError(f'{profile}: high-risk DNG presence leaked')


def validate_record(record: dict, row: dict, global_literals: set[str]) -> None:
    scenario = record['assignment']['scenario']
    profile = record['assignment']['mask_profile']
    synthetic = record['synthetic']
    effective = record['effective']
    if scenario == 'P0':
        if not effective['available'] is False:
            raise AssertionError("validate_record: effective['available'] is False")
        if not effective['c2pa'] is None:
            raise AssertionError("validate_record: effective['c2pa'] is None")
        if not effective['metadata'] is None:
            raise AssertionError("validate_record: effective['metadata'] is None")
        if not profile == 'none':
            raise AssertionError("validate_record: profile == 'none'")
    else:
        if not effective['available'] is True:
            raise AssertionError("validate_record: effective['available'] is True")
        if not profile in MASK_ORDER:
            raise AssertionError('validate_record: profile in MASK_ORDER')
        if not set(effective['metadata']) == set(EFFECTIVE_METADATA_FIELDS):
            raise AssertionError('Effective metadata schema mismatch')
        if not set(effective['c2pa']) == set(C2PA_FIELDS):
            raise AssertionError("validate_record: set(effective['c2pa']) == set(C2PA_FIELDS)")
        assert_mask_semantics(effective, profile)
    if scenario == 'P1':
        if not effective['c2pa']['present'] is False:
            raise AssertionError("validate_record: effective['c2pa']['present'] is False")
        if not effective['metadata']['metadata_present'] is True:
            raise AssertionError("validate_record: effective['metadata']['metadata_present'] is True")
    if record['natural']['c2pa'].get('present', False):
        if scenario == 'P0':
            if not effective['c2pa'] is None:
                raise AssertionError("validate_record: effective['c2pa'] is None")
        elif scenario == 'P1':
            if not effective['c2pa']['present'] is False:
                raise AssertionError("validate_record: effective['c2pa']['present'] is False")
        elif not synthetic['kind'] == 'canonical_c2pa_simulation':
            raise AssertionError("validate_record: synthetic['kind'] == 'canonical_c2pa_simulation'")
    if scenario in {'P2', 'P3', 'P4'}:
        c2 = synthetic['c2pa']
        if not c2['present'] is True:
            raise AssertionError("validate_record: c2['present'] is True")
        if not c2['validation_state'] == 'valid':
            raise AssertionError("validate_record: c2['validation_state'] == 'valid'")
        if scenario == 'P2':
            if not c2['active_action'] == 'created':
                raise AssertionError("validate_record: c2['active_action'] == 'created'")
            if not c2['origin_resolved'] is True:
                raise AssertionError("validate_record: c2['origin_resolved'] is True")
            if not c2['has_parent'] is False:
                raise AssertionError("validate_record: c2['has_parent'] is False")
            if not c2['manifest_count'] == 1:
                raise AssertionError("validate_record: c2['manifest_count'] == 1")
            if not c2['ingredient_count'] == 0:
                raise AssertionError("validate_record: c2['ingredient_count'] == 0")
            if not c2['chain_depth'] == 1:
                raise AssertionError("validate_record: c2['chain_depth'] == 1")
        elif scenario == 'P3':
            if not c2['active_action'] == 'opened':
                raise AssertionError("validate_record: c2['active_action'] == 'opened'")
            if not c2['action_edited'] is True:
                raise AssertionError("validate_record: c2['action_edited'] is True")
            if not c2['has_parent'] is True:
                raise AssertionError("validate_record: c2['has_parent'] is True")
            if not c2['parent_manifest_available'] is True:
                raise AssertionError("validate_record: c2['parent_manifest_available'] is True")
            if not c2['origin_resolved'] is True:
                raise AssertionError("validate_record: c2['origin_resolved'] is True")
            if not c2['manifest_count'] == 2:
                raise AssertionError("validate_record: c2['manifest_count'] == 2")
            if not c2['ingredient_count'] == 1:
                raise AssertionError("validate_record: c2['ingredient_count'] == 1")
            if not c2['chain_depth'] == 2:
                raise AssertionError("validate_record: c2['chain_depth'] == 2")
        else:
            if not c2['active_action'] == 'opened':
                raise AssertionError("validate_record: c2['active_action'] == 'opened'")
            if not c2['has_parent'] is True:
                raise AssertionError("validate_record: c2['has_parent'] is True")
            if not c2['parent_manifest_available'] is False:
                raise AssertionError("validate_record: c2['parent_manifest_available'] is False")
            if not c2['unknown_provenance'] is True:
                raise AssertionError("validate_record: c2['unknown_provenance'] is True")
            if not c2['origin_resolved'] is False:
                raise AssertionError("validate_record: c2['origin_resolved'] is False")
            if not c2['origin_source_type'] is None:
                raise AssertionError("validate_record: c2['origin_source_type'] is None")
            if not c2['manifest_count'] == 1:
                raise AssertionError("validate_record: c2['manifest_count'] == 1")
            if not c2['ingredient_count'] == 1:
                raise AssertionError("validate_record: c2['ingredient_count'] == 1")
            if not c2['chain_depth'] == 1:
                raise AssertionError("validate_record: c2['chain_depth'] == 1")
        if scenario in {'P2', 'P3'}:
            expected_origin = 'digitalCapture' if int(row['label']) == 0 else 'trainedAlgorithmicMedia'
            if not c2['origin_source_type'] == expected_origin:
                raise AssertionError("validate_record: c2['origin_source_type'] == expected_origin")
    if scenario in {'P2', 'P3', 'P4'}:
        exposed = MINIMAL_C2PA_FIELDS if profile == 'minimal' else STANDARD_C2PA_FIELDS
        expected = mask_dict(synthetic['c2pa'], C2PA_FIELDS, exposed)
        if effective['c2pa'] != expected:
            raise ValueError(f"{row['image_name']}: effective C2PA differs from the assigned scenario")

    violations = scan_effective_forbidden(effective, row, global_literals)
    if violations:
        raise RuntimeError(f"{row['image_name']}: effective provenance leakage: {violations[:10]}")


def effective_to_numeric(effective: dict) -> tuple[list[float], list[float], list[float]]:
    c2 = effective['c2pa'] if effective['c2pa'] is not None else {}
    md = effective['metadata'] if effective['metadata'] is not None else {}
    origin = c2.get('origin_source_type')

    canonical = {
        'available': effective['available'],
        'c2pa_present': c2.get('present'),
        'c2pa_valid': None if c2.get('validation_state') is None else (c2.get('validation_state') == 'valid'),
        'c2pa_origin_resolved': c2.get('origin_resolved'),
        'c2pa_origin_digital_capture': None if origin is None else (origin == 'digitalCapture'),
        'c2pa_origin_trained_algorithmic_media': (
            None if origin is None else origin == 'trainedAlgorithmicMedia'
        ),
        'c2pa_action_created': c2.get('action_created'),
        'c2pa_action_opened': c2.get('action_opened'),
        'c2pa_action_edited': c2.get('action_edited'),
        'c2pa_has_parent': c2.get('has_parent'),
        'c2pa_parent_manifest_available': c2.get('parent_manifest_available'),
        'c2pa_unknown_provenance': c2.get('unknown_provenance'),
        'c2pa_manifest_count': c2.get('manifest_count'),
        'c2pa_ingredient_count': c2.get('ingredient_count'),
        'c2pa_chain_depth': c2.get('chain_depth'),
    }
    for key in EFFECTIVE_METADATA_FIELDS:
        canonical[key] = md.get(key)

    values = []
    masks = []
    for i, field in enumerate(PROVENANCE_NUMERIC_FIELDS_V1):
        value = canonical[field]
        if field == 'available':
            values.append(1.0 if bool(value) else 0.0)
            continue
        if value is None:
            values.append(0.0)
            masks.append(0.0)
        elif isinstance(value, bool):
            values.append(1.0 if value else 0.0)
            masks.append(1.0)
        else:
            values.append(float(value))
            masks.append(1.0)
    return values, masks, values + masks


def effective_to_text(effective: dict) -> str:
    if not effective['available']:
        return '[PROVENANCE]\nEvidence unavailable.'

    lines = ['[PROVENANCE]', 'Evidence available.']
    c2 = effective['c2pa']
    md = effective['metadata']

    if c2.get('present') is not None:
        lines.append('C2PA metadata is present.' if c2['present'] else 'C2PA metadata is absent.')
    if c2.get('validation_state') is not None:
        lines.append(f"C2PA validation state: {c2['validation_state']}.")
    if c2.get('active_action') is not None:
        lines.append(f"Active C2PA action: {c2['active_action']}.")
    if c2.get('action_created') is not None:
        lines.append(
            'The asset has a created action.' if c2['action_created']
            else 'The asset does not have a created action.'
        )
    if c2.get('action_opened') is not None:
        lines.append(
            'The asset has an opened action.' if c2['action_opened']
            else 'The asset does not have an opened action.'
        )
    if c2.get('action_edited') is not None:
        lines.append(
            'The asset has an edited action.' if c2['action_edited']
            else 'The asset does not have an edited action.'
        )
    if c2.get('has_parent') is not None:
        lines.append(
            'A parent ingredient is present.' if c2['has_parent']
            else 'No parent ingredient is present.'
        )
    if c2.get('parent_manifest_available') is not None:
        lines.append(
            'The parent manifest is available.' if c2['parent_manifest_available']
            else 'The parent manifest is unavailable.'
        )
    if c2.get('unknown_provenance') is not None:
        lines.append(
            'The parent provenance is unresolved.' if c2['unknown_provenance']
            else 'Unknown-provenance status is absent.'
        )
    if c2.get('origin_resolved') is not None:
        lines.append('The origin is resolved.' if c2['origin_resolved'] else 'The origin is unresolved.')
    if c2.get('origin_source_type') is not None:
        lines.append(f"Origin digital source type: {c2['origin_source_type']}.")
    if c2.get('manifest_count') is not None:
        lines.append(f"Manifest count: {c2['manifest_count']}.")
    if c2.get('ingredient_count') is not None:
        lines.append(f"Ingredient count: {c2['ingredient_count']}.")
    if c2.get('chain_depth') is not None:
        lines.append(f"Provenance chain depth: {c2['chain_depth']}.")

    bool_text = [
        ('exif_present', 'EXIF metadata'),
        ('xmp_present', 'XMP metadata'),
        ('camera_identity_present', 'Camera identity metadata'),
        ('capture_settings_present', 'Capture-settings metadata'),
        ('software_metadata_present', 'Software metadata'),
        ('camera_make_present', 'Camera make information'),
        ('camera_model_present', 'Camera model information'),
        ('lens_model_present', 'Lens model information'),
        ('datetime_present', 'Date/time metadata'),
        ('gps_present', 'GPS metadata'),
        ('dng_present', 'DNG metadata'),
        ('dng_color_matrix_present', 'DNG color-matrix metadata'),
        ('dng_camera_calibration_present', 'DNG camera-calibration metadata'),
        ('dng_measured_rggb_present', 'DNG measured-RGGB metadata'),
        ('dng_black_level_present', 'DNG black-level metadata'),
        ('dng_white_level_present', 'DNG white-level metadata'),
        ('dng_cfa_present', 'DNG CFA metadata'),
        ('dng_version_present', 'DNG version metadata'),
    ]
    for key, phrase in bool_text:
        if md.get(key) is not None:
            lines.append(f'{phrase} is present.' if md[key] else f'{phrase} is absent.')

    for key, label in [
        ('iso', 'ISO'), ('f_number', 'F-number'),
        ('exposure_time_s', 'Exposure time (s)'), ('focal_length_mm', 'Focal length (mm)'),
    ]:
        if md.get(key) is not None:
            lines.append(f'{label}: {md[key]}.')

    return '\n'.join(lines)


def canonical_record_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def records_sha256(records: list[dict]) -> str:
    ordered = sorted(records, key=lambda r: r['provenance_id'])
    payload = '\n'.join(canonical_record_line(r) for r in ordered).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def counts_by(df: pd.DataFrame, group: str, category: str, categories: list[str]) -> dict:
    output = {}
    for group_value, part in df.groupby(group):
        counts = part[category].value_counts().to_dict()
        output[str(group_value)] = {c: int(counts.get(c, 0)) for c in categories}
    return output


def validate_pass1(extended: pd.DataFrame, split: str) -> None:
    if not len(extended) == 2500:
        raise AssertionError('validate_pass1: len(extended) == 2500')
    if not (extended['label'] == 0).sum() == 1250:
        raise AssertionError("validate_pass1: (extended['label'] == 0).sum() == 1250")
    if not (extended['label'] == 1).sum() == 1250:
        raise AssertionError("validate_pass1: (extended['label'] == 1).sum() == 1250")
    if not extended['provenance_id'].is_unique:
        raise AssertionError("validate_pass1: extended['provenance_id'].is_unique")
    if not extended['image_name'].is_unique:
        raise AssertionError("validate_pass1: extended['image_name'].is_unique")
    for source, expected in SOURCE_SCENARIO_QUOTAS.items():
        part = extended[extended['source'] == source]
        actual = {s: int((part['provenance_scenario'] == s).sum()) for s in SCENARIO_ORDER}
        if not actual == expected:
            raise AssertionError((split, source, actual, expected))
    class_expected = {'P0': 750, 'P1': 250, 'P2': 130, 'P3': 70, 'P4': 50}
    for label in [0, 1]:
        part = extended[extended['label'] == label]
        actual = {s: int((part['provenance_scenario'] == s).sum()) for s in SCENARIO_ORDER}
        if not actual == class_expected:
            raise AssertionError((split, label, actual))
    for source, expected in SOURCE_MASK_QUOTAS.items():
        part = extended[(extended['source'] == source) & (extended['provenance_available'] == 1)]
        actual = {m: int((part['provenance_mask_profile'] == m).sum()) for m in MASK_ORDER}
        if not actual == expected:
            raise AssertionError((split, source, actual, expected))
    class_mask_expected = {'minimal': 350, 'standard': 130, 'leaky': 20}
    for label in [0, 1]:
        part = extended[(extended['label'] == label) & (extended['provenance_available'] == 1)]
        actual = {m: int((part['provenance_mask_profile'] == m).sum()) for m in MASK_ORDER}
        if not actual == class_mask_expected:
            raise AssertionError((split, label, actual))
    for source in AI_SOURCES:
        for generator, part in extended[extended['source'] == source].groupby('generator'):
            scenario = {s: int((part['provenance_scenario'] == s).sum()) for s in SCENARIO_ORDER}
            if not scenario == AI_SCENARIO_PATTERN:
                raise AssertionError((split, source, generator, scenario))
            available = part[part['provenance_available'] == 1]
            masks = {m: int((available['provenance_mask_profile'] == m).sum()) for m in MASK_ORDER}
            if not masks == AI_MASK_PATTERN:
                raise AssertionError((split, source, generator, masks))
    if not int(extended['provenance_available'].sum()) == 1000:
        raise AssertionError("validate_pass1: int(extended['provenance_available'].sum()) == 1000")
    if not int(extended['provenance_scenario'].isin(['P2', 'P3', 'P4']).sum()) == 500:
        raise AssertionError('Expected 500 records with effective C2PA')


def build_summary(records: list[dict], extended: pd.DataFrame, split: str, assignment_hash: str) -> dict:
    audit = extended.copy()
    audit['label_name'] = audit['label'].map({0: 'Real', 1: 'AI'})
    available = audit[audit['provenance_available'] == 1]
    ai = audit[audit['label'] == 1]
    ai_available = available[available['label'] == 1]

    natural_present_names = {
        r['image_name'] for r in records
        if r['natural']['c2pa'].get('present', False)
    }
    natural_audit = audit[audit['image_name'].isin(natural_present_names)].copy()
    natural_c2pa_count_by_source = {
        str(k): int(v)
        for k, v in natural_audit.groupby('source', dropna=False).size().to_dict().items()
    }
    natural_c2pa_count_by_generator = {}
    if not natural_audit.empty:
        for (source, generator), part in natural_audit.groupby(['source', 'generator'], dropna=False):
            gen_name = '<none>' if pd.isna(generator) or str(generator).strip() == '' else str(generator)
            natural_c2pa_count_by_generator[f'{source}::{gen_name}'] = int(len(part))

    scenario_counts_by_generator = {}
    mask_counts_by_generator = {}
    for (source, generator), part in ai.groupby(['source', 'generator']):
        key = f'{source}::{generator}'
        scenario_counts_by_generator[key] = {
            s: int((part['provenance_scenario'] == s).sum()) for s in SCENARIO_ORDER
        }
    for (source, generator), part in ai_available.groupby(['source', 'generator']):
        key = f'{source}::{generator}'
        mask_counts_by_generator[key] = {
            m: int((part['provenance_mask_profile'] == m).sum()) for m in MASK_ORDER
        }

    natural_c2pa_count = sum(int(r['natural']['c2pa'].get('present', False)) for r in records)
    effective_c2pa_count = sum(
        int(bool(r['effective']['c2pa'] and r['effective']['c2pa'].get('present') is True))
        for r in records
    )

    return {
        'schema_version': PROVENANCE_VERSION,
        'split': split,
        'n_records': len(records),
        'scenario_counts_by_source': counts_by(audit, 'source', 'provenance_scenario', list(SCENARIO_ORDER)),
        'scenario_counts_by_label': counts_by(
            audit, 'label_name', 'provenance_scenario', list(SCENARIO_ORDER)
        ),
        'mask_counts_by_source': counts_by(available, 'source', 'provenance_mask_profile', list(MASK_ORDER)),
        'mask_counts_by_label': counts_by(
            available, 'label_name', 'provenance_mask_profile', list(MASK_ORDER)
        ),
        'scenario_counts_by_generator': scenario_counts_by_generator,
        'mask_counts_by_generator': mask_counts_by_generator,
        'natural_c2pa_policy': NATURAL_C2PA_POLICY,
        'protocol_profile': PROTOCOL_PROFILE,
        'natural_c2pa_count': int(natural_c2pa_count),
        'natural_c2pa_count_by_source': natural_c2pa_count_by_source,
        'natural_c2pa_count_by_generator': natural_c2pa_count_by_generator,
        'effective_c2pa_count': int(effective_c2pa_count),
        'effective_available_count': int(extended['provenance_available'].sum()),
        'effective_unavailable_count': int((extended['provenance_available'] == 0).sum()),
        'forbidden_field_violations': 0,
        'semantic_validation_errors': 0,
        'assignment_sha256': assignment_hash,
    }


def build_split_records(
    df: pd.DataFrame,
    natural_by_name: dict[str, dict],
    split: str,
    seed: int,
    global_literals: set[str],
) -> tuple[list[dict], pd.DataFrame, dict]:
    scenario_df = assign_scenarios(df, split, seed)
    assignment_df = assign_masks(df, split, scenario_df, seed)
    assignment_by_name = {r['image_name']: r for r in assignment_df.to_dict(orient='records')}
    records = []
    extended_rows = []
    ordered = df.sort_values('image_name', kind='mergesort').reset_index(drop=True)
    for index, row in enumerate(ordered.to_dict(orient='records'), start=1):
        name = row['image_name']
        if name not in natural_by_name:
            raise RuntimeError(f'{split}: missing natural provenance for {name}')
        assignment = assignment_by_name[name]
        scenario = assignment['scenario']
        profile = assignment['mask_profile']
        natural = copy.deepcopy(natural_by_name[name])
        synthetic = build_synthetic_provenance(scenario, row, seed)
        candidate = merge_natural_and_synthetic(natural, synthetic, scenario, row)
        effective = apply_mask_profile(candidate, scenario, profile)
        provenance_id = f'prov-{split}-{index:06d}'
        record = {
            'schema_version': PROVENANCE_VERSION,
            'provenance_id': provenance_id,
            'image_name': name,
            'assignment': {
                'split': split,
                'scenario': scenario,
                'mask_profile': profile,
                'seed': seed,
                'scenario_rank_key': assignment['scenario_rank_key'],
                'mask_rank_key': assignment['mask_rank_key'],
            },
            'natural': natural,
            'synthetic': synthetic,
            'effective': effective,
        }
        validate_record(record, row, global_literals)
        records.append(record)
        ext = dict(row)
        ext['provenance_id'] = provenance_id
        ext['provenance_scenario'] = scenario
        ext['provenance_available'] = int(effective['available'])
        ext['provenance_mask_profile'] = profile
        extended_rows.append(ext)
    extended = pd.DataFrame(extended_rows)
    extended = extended[list(df.columns) + PROVENANCE_COLUMNS]
    validate_pass1(extended, split)
    assignment_hash = records_sha256(records)
    summary = build_summary(records, extended, split, assignment_hash)
    if not summary['natural_c2pa_count'] >= 0:
        raise AssertionError("build_split_records: summary['natural_c2pa_count'] >= 0")
    if not summary['natural_c2pa_policy'] == NATURAL_C2PA_POLICY:
        raise AssertionError("build_split_records: summary['natural_c2pa_policy'] == NATURAL_C2PA_POLICY")
    if not summary['effective_available_count'] == 1000:
        raise AssertionError("build_split_records: summary['effective_available_count'] == 1000")
    if not summary['effective_unavailable_count'] == 1500:
        raise AssertionError("build_split_records: summary['effective_unavailable_count'] == 1500")
    if not summary['effective_c2pa_count'] == 500:
        raise AssertionError("build_split_records: summary['effective_c2pa_count'] == 500")
    if not summary['forbidden_field_violations'] == 0:
        raise AssertionError("build_split_records: summary['forbidden_field_violations'] == 0")
    if not summary['semantic_validation_errors'] == 0:
        raise AssertionError("build_split_records: summary['semantic_validation_errors'] == 0")
    return (records, extended, summary)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for record in sorted(records, key=lambda r: r['provenance_id']):
            f.write(canonical_record_line(record) + '\n')


def assert_existing_assignment_immutable(
    split: str, new_records: list[dict], provenance_dir: Path, replace_existing: bool
) -> None:
    existing_path = provenance_dir / f'{split}.jsonl'
    if not existing_path.exists():
        return
    existing_records = []
    with existing_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_records.append(json.loads(line))
    if not existing_records:
        return
    if replace_existing:
        return
    if any(r.get('schema_version') != PROVENANCE_VERSION for r in existing_records):
        raise RuntimeError(
            f'{split}: existing provenance uses another schema version; do not silently overwrite it.'
        )
    existing = {r['image_name']: (r['provenance_id'], r['assignment']) for r in existing_records}
    new = {r['image_name']: (r['provenance_id'], r['assignment']) for r in new_records}
    if existing != new:
        existing_names, new_names = set(existing), set(new)
        detail = {
            'missing_from_new': sorted(existing_names - new_names)[:10],
            'new_images': sorted(new_names - existing_names)[:10],
            'changed_assignments': [
                name for name in sorted(existing_names & new_names) if existing[name] != new[name]
            ][:10],
        }
        raise RuntimeError(f'{split}: immutable existing provenance assignment changed: {detail}')


def preflight_existing_outputs(config: BuildConfig) -> None:
    if config.dry_run or config.replace_existing:
        return
    for split in SPLITS:
        path = config.provenance_dir / f'{split}.jsonl'
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as stream:
            first = next((line for line in stream if line.strip()), None)
        if first is None:
            raise ValueError(f'{split}: existing provenance file is empty: {path}')
        schema = json.loads(first).get('schema_version')
        if schema != PROVENANCE_VERSION:
            raise ValueError(
                f'{split}: existing provenance schema is {schema!r}; '
                'use --replace-existing only if replacement is intended'
            )


def check_split_separation(metadata: dict[str, pd.DataFrame]) -> None:
    val = metadata['validation']
    test = metadata['test']
    shared_hashes = set(val['sha256'].str.lower()) & set(test['sha256'].str.lower())
    if shared_hashes:
        raise ValueError(f"Validation and test share {len(shared_hashes)} image hashes")
    val_ai = set(val.loc[val['label'] == 1, 'generator'])
    test_ai = set(test.loc[test['label'] == 1, 'generator'])
    if val_ai & test_ai:
        raise ValueError(f"AI generators overlap across splits: {sorted(val_ai & test_ai)}")


def check_determinism(
    metadata: dict[str, pd.DataFrame],
    natural: dict[str, dict[str, dict]],
    builds: dict[str, tuple[list[dict], pd.DataFrame, dict]],
    seed: int,
    forbidden_literals: set[str],
) -> None:
    for split in SPLITS:
        records, extended, summary = builds[split]
        repeated, extended_repeated, summary_repeated = build_split_records(
            metadata[split], copy.deepcopy(natural[split]), split, seed, forbidden_literals
        )
        if summary['assignment_sha256'] != summary_repeated['assignment_sha256']:
            raise ValueError(f"{split}: deterministic record hash mismatch")
        if records_sha256(records) != records_sha256(repeated):
            raise ValueError(f"{split}: repeated build differs")
        columns = ['image_name', *PROVENANCE_COLUMNS]
        left = extended[columns].sort_values('image_name').reset_index(drop=True)
        right = extended_repeated[columns].sort_values('image_name').reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"{split}: repeated assignment differs")


def write_stage_files(
    builds: dict[str, tuple[list[dict], pd.DataFrame, dict]],
    stage: Path,
    config: BuildConfig,
) -> list[tuple[Path, Path]]:
    outputs: list[tuple[Path, Path]] = []
    for split in SPLITS:
        records, extended, summary = builds[split]
        staged_provenance = stage / 'provenance'
        staged_metadata = stage / 'metadata'
        staged_provenance.mkdir(parents=True, exist_ok=True)
        staged_metadata.mkdir(parents=True, exist_ok=True)

        jsonl = staged_provenance / f'{split}.jsonl'
        summary_json = staged_provenance / f'{split}_summary.json'
        metadata_csv = staged_metadata / f'{split}.csv'

        write_jsonl(jsonl, records)
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
            encoding='utf-8',
        )
        extended.to_csv(metadata_csv, index=False)

        outputs.extend([
            (jsonl, config.provenance_dir / jsonl.name),
            (summary_json, config.provenance_dir / summary_json.name),
            (metadata_csv, config.output_metadata_dir / metadata_csv.name),
        ])
    return outputs


def publish_outputs(pairs: list[tuple[Path, Path]], stage: Path, verify=None) -> None:
    """Publish verified files and restore existing outputs if a write fails."""
    prepared: list[tuple[Path, Path, Path | None]] = []
    pending_files: list[Path] = []
    published: list[tuple[Path, Path | None]] = []
    try:
        for index, (source, destination) in enumerate(pairs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(
                f'.{destination.name}.{uuid.uuid4().hex}.pending'
            )
            pending_files.append(pending)
            shutil.copyfile(source, pending)
            if sha256_file(source) != sha256_file(pending):
                raise OSError(f'Output verification failed: {destination}')
            backup = None
            if destination.exists():
                backup = stage / f'backup-{index:02d}'
                shutil.copy2(destination, backup)
            prepared.append((pending, destination, backup))

        for pending, destination, backup in prepared:
            os.replace(pending, destination)
            published.append((destination, backup))

        for source, destination in pairs:
            if sha256_file(source) != sha256_file(destination):
                raise OSError(f'Read-back verification failed: {destination}')
        if verify is not None:
            verify()
    except Exception:
        for destination, backup in reversed(published):
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        raise
    finally:
        for pending in pending_files:
            pending.unlink(missing_ok=True)


def check_published_records(config: BuildConfig) -> None:
    for split in SPLITS:
        csv_path = config.output_metadata_dir / f'{split}.csv'
        jsonl_path = config.provenance_dir / f'{split}.jsonl'
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        with jsonl_path.open(encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(df) != 2500 or len(records) != 2500:
            raise ValueError(f'{split}: published record count mismatch')
        if len({r['provenance_id'] for r in records}) != 2500:
            raise ValueError(f'{split}: duplicate published provenance_id')
        if set(df['image_name']) != {r['image_name'] for r in records}:
            raise ValueError(f'{split}: published image_name join mismatch')
        by_name = df.set_index('image_name', verify_integrity=True)
        for record in records:
            row = by_name.loc[record['image_name']]
            if (row['provenance_id'] != record['provenance_id']
                    or row['provenance_scenario'] != record['assignment']['scenario']
                    or row['provenance_mask_profile'] != record['assignment']['mask_profile']
                    or row['provenance_available'] != str(int(record['effective']['available']))):
                raise ValueError(f'{split}: published metadata/provenance mismatch')


def run(config: BuildConfig) -> dict[str, dict]:
    tool = shutil.which(config.exiftool)
    if tool is None:
        raise RuntimeError(f'ExifTool executable not found: {config.exiftool}')
    if config.workers < 1 or config.exiftool_batch_size < 1:
        raise ValueError('workers and exiftool batch size must be positive')
    try:
        c2pa_module = importlib.import_module('c2pa')
    except ImportError as exc:
        raise RuntimeError('c2pa-python is required for natural C2PA auditing') from exc
    if not hasattr(c2pa_module, 'Context') or not hasattr(c2pa_module, 'Reader'):
        raise RuntimeError('c2pa-python must provide Context and Reader')

    preflight_existing_outputs(config)
    metadata = {split: load_split_metadata(split, config) for split in SPLITS}
    check_split_separation(metadata)
    LOGGER.info('Input: %s images', sum(len(part) for part in metadata.values()))
    before = verify_image_hashes(metadata, config)
    catalog = load_unsplash_catalog(config.unsplash_manifest)
    natural = {}
    for split in SPLITS:
        LOGGER.info('Extracting natural provenance: %s', split)
        natural[split] = extract_natural_split(
            metadata[split], split, catalog, config, c2pa_module
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            observed = {
                name for name, record in natural[split].items()
                if record['c2pa']['present']
            }
            by_generator = metadata[split][
                metadata[split]['image_name'].isin(observed)
            ].groupby(['source', 'generator']).size().to_dict()
            LOGGER.debug('%s natural C2PA by source/generator: %s', split, by_generator)

    literals = build_global_forbidden_literals(metadata, config)
    builds = {}
    for split in SPLITS:
        builds[split] = build_split_records(
            metadata[split], natural[split], split, config.seed, literals
        )
    check_determinism(metadata, natural, builds, config.seed, literals)

    after = verify_image_hashes(metadata, config)
    if before != after:
        raise ValueError('Image bytes changed during the build')

    for split in SPLITS:
        if not config.dry_run:
            assert_existing_assignment_immutable(
                split, builds[split][0], config.provenance_dir, config.replace_existing
            )
        summary = builds[split][2]
        LOGGER.info(
            '%s: records=%d, natural_c2pa=%d, effective_c2pa=%d, assignment_sha256=%s',
            split, summary['n_records'], summary['natural_c2pa_count'],
            summary['effective_c2pa_count'], summary['assignment_sha256'],
        )

    if config.dry_run:
        LOGGER.info('Validation complete; no outputs written (--dry-run)')
    else:
        config.output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.provenance-stage-', dir=config.output_root) as temporary:
            staged = write_stage_files(builds, Path(temporary), config)
            publish_outputs(
                staged, Path(temporary), verify=lambda: check_published_records(config)
            )
        LOGGER.info('Published: %s', config.output_root)
    return {split: builds[split][2] for split in SPLITS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build deterministic provenance for existing validation/test images.'
    )
    parser.add_argument(
        '--dataset-root', type=Path, required=True,
        help='Directory containing validation/, test/, metadata/ and manifests/.'
    )
    parser.add_argument('--validation-dir', type=Path, help='Validation image directory.')
    parser.add_argument('--test-dir', type=Path, help='Test image directory.')
    parser.add_argument('--metadata-dir', type=Path, help='Input metadata CSV directory.')
    parser.add_argument('--unsplash-manifest', type=Path, help='Path to real_unsplash.csv.')
    parser.add_argument(
        '--output-root', type=Path,
        help='Output root containing provenance/ and metadata/ (default: dataset root).'
    )
    parser.add_argument('--exiftool', default='exiftool', help='ExifTool executable or path.')
    parser.add_argument('--exiftool-batch-size', type=int, default=80)
    parser.add_argument('--workers', type=int, default=4, help='Image hash worker threads.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Validate without writing outputs.')
    parser.add_argument(
        '--replace-existing', action='store_true',
        help='Allow replacing provenance with a different schema or assignment.'
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', action='store_true', help='Show errors only.')
    verbosity.add_argument('--verbose', action='store_true', help='Enable debug logs.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )
    root = args.dataset_root.expanduser().resolve()
    config = BuildConfig(
        dataset_root=root,
        image_dirs={
            'validation': (args.validation_dir or root / 'validation').expanduser().resolve(),
            'test': (args.test_dir or root / 'test').expanduser().resolve(),
        },
        metadata_dir=(args.metadata_dir or root / 'metadata').expanduser().resolve(),
        unsplash_manifest=(
            args.unsplash_manifest or root / 'manifests' / 'real_unsplash.csv'
        ).expanduser().resolve(),
        output_root=(args.output_root or root).expanduser().resolve(),
        exiftool=args.exiftool,
        exiftool_batch_size=args.exiftool_batch_size,
        workers=args.workers,
        seed=args.seed,
        dry_run=args.dry_run,
        replace_existing=args.replace_existing,
    )
    try:
        run(config)
    except Exception as exc:
        if args.verbose:
            LOGGER.exception('Build failed')
        else:
            LOGGER.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
