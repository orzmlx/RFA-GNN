import argparse
import os
import subprocess
import sys
import traceback

import nbformat
from nbclient import NotebookClient


def _normalize_notebook(nb):
    try:
        from nbformat.validator import normalize

        _, nb_norm = normalize(nb)
        return nb_norm
    except Exception:
        return nb


def _run_with_nbconvert(nb_path: str, out_path: str, timeout: int, kernel: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    out_name = os.path.basename(out_path)
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        nb_path,
        "--output",
        out_name,
        "--output-dir",
        out_dir,
        f"--ExecutePreprocessor.timeout={int(timeout)}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    nb_path = os.path.abspath(args.notebook_path)
    out_path = os.path.abspath(args.out or args.notebook_path)
    timeout = int(args.timeout)
    kernel = args.kernel
    retries = max(int(args.retries), 0)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nb = _normalize_notebook(nb)

    last_err = None
    for attempt in range(retries + 1):
        try:
            client = NotebookClient(nb, timeout=timeout, kernel_name=kernel, allow_errors=False)
            client.execute()
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(
                f"[run_notebook] nbclient execute failed (attempt {attempt+1}/{retries+1}): {e}",
                file=sys.stderr,
            )
            traceback.print_exc()

    if last_err is not None:
        try:
            print("[run_notebook] Falling back to nbconvert execution...", file=sys.stderr)
            _run_with_nbconvert(nb_path, out_path, timeout=timeout, kernel=kernel)
            print(out_path)
            return
        except Exception as e:
            print(f"[run_notebook] nbconvert fallback failed: {e}", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1) from last_err

    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(out_path)


if __name__ == "__main__":
    main()
