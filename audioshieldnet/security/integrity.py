# audioshieldnet/security/integrity.py
import hashlib, pathlib, json

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def manifest_for_dataset(root):
    root = pathlib.Path(root)
    files = sorted(p for p in root.rglob('*.wav'))
    return {str(p.relative_to(root)): file_sha256(p) for p in files}

def save_manifest(out_path, root):
    m = manifest_for_dataset(root)
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    return out_path
