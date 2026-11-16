"""Package only the portable recorded evaluator; exclude secrets and company DBs."""
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_bundle():
    from evaluation.catalog import SAMPLES
    files = [
        ROOT / name for name in ('app.py', 'config.py', 'requirements-demo.txt',
        'Dockerfile.demo', 'Dockerfile.demo.dockerignore', '.dockerignore', 'compose.demo.yaml',
        'deploy/Caddyfile', 'deploy/AWS_DEMO.md', 'templates/showcase.html',
        'templates/evaluation.html', 'static/evaluation.css', 'static/evaluation.js',
        'static/demo_poles.geojson', 'static/model_evaluation.json')
    ]
    files += list((ROOT / 'evaluation').glob('*.py'))
    files += list((ROOT / 'evaluation' / 'recordings').glob('*.json'))
    files += [p for p in (ROOT / 'static' / 'vendor').rglob('*') if p.is_file()]
    files += [ROOT / 'static' / 'demo_samples' / sample['filename'] for sample in SAMPLES]
    files += [ROOT / 'static' / 'demo_samples' / 'PHOTO-2026-05-21-11-32-37.jpg']
    output = ROOT / '.build' / 'polepilot-demo.tar.gz'
    output.parent.mkdir(exist_ok=True)
    with tarfile.open(output, 'w:gz') as bundle:
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(path)
            bundle.add(path, arcname='polepilot-demo/' + path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n')
    print(f'{output}\nFiles: {len(set(files))}\nBytes: {output.stat().st_size}\nSHA256: {digest}')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    build_bundle()
