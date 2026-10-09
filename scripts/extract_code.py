# File: scripts/extract_code.py
# Description: Collects folder structure and project source files into a combined code listing.
# Author Name: Debleena Nandy
# Date: 10-07-2026
# Time: 11:41:01 +05:30

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    #ROOT,
    Path("C:\\AI_Dev\\portfolio\\01-ai-qa-agent-platform\\tests"),
    Path("C:\\AI_Dev\\portfolio\\01-ai-qa-agent-platform\\evaluation"),
]

OUTPUT_FILE = Path("combined_code.txt")

EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".html", ".css", ".sql", ".sh", ".md", ".yml", ".yaml",
    ".json", ".toml", ".ini", ".txt", ".example",
}

EXCLUDED_FOLDERS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".pytest_cache", ".ruff_cache",
    "data", "artifacts",
}


def report_error(error):
    print(f"Warning: {error}")


def walk_folder(root):
    """Walk folders consistently for structure and code extraction."""
    for current, directories, filenames in os.walk(
        root, followlinks=False, onerror=report_error
    ):
        current = Path(current)

        directories[:] = sorted(
            name for name in directories
            if name not in EXCLUDED_FOLDERS
            and not (current / name).is_symlink()
        )

        files = sorted(
            current / name
            for name in filenames
            if not (current / name).is_symlink()
        )

        yield current, files


def write_heading(destination, title):
    destination.write(
        f"\n{'=' * 80}\n"
        f"{title}\n"
        f"{'=' * 80}\n"
    )


def write_folder_structure(destination, roots, output):
    """Write an indented directory listing, including empty folders."""
    write_heading(destination, "FOLDER STRUCTURE")

    for root in roots:
        destination.write(f"\nROOT: {root}\n")

        for current, files in walk_folder(root):
            relative = current.relative_to(root)
            depth = len(relative.parts)
            indent = "    " * depth

            destination.write(f"{indent}{current.name}/\n")

            for source in files:
                # Do not include the generated output in its own listing.
                if source.resolve() == output:
                    continue

                destination.write(f"{indent}    {source.name}\n")


def extract_code():
    roots = [folder.expanduser().resolve() for folder in FOLDERS]
    output = OUTPUT_FILE.expanduser().resolve()

    # Validate all folders before creating the output.
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"Invalid folder: {root}")

    seen = set()
    count = 0

    # Exclusive creation prevents overwriting an existing file.
    with output.open("x", encoding="utf-8") as destination:
        write_folder_structure(destination, roots, output)
        write_heading(destination, "SOURCE CODE")

        for root in roots:
            for current, files in walk_folder(root):
                for source in files:
                    if source.suffix.lower() not in EXTENSIONS:
                        continue

                    resolved = source.resolve()
                    if resolved == output or resolved in seen:
                        continue

                    try:
                        code = source.read_text(encoding="utf-8-sig")
                    except (OSError, UnicodeError) as error:
                        print(f"Skipped {source}: {error}")
                        continue

                    write_heading(destination, f"FILE: {source}")
                    destination.write(code)
                    destination.write("\n")

                    seen.add(resolved)
                    count += 1

    print(f"Extracted folder structure and {count} source files into: {output}")


if __name__ == "__main__":
    try:
        extract_code()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}")