"""Download the example SWITi data and checkpoints from Zenodo."""

import argparse
from pathlib import Path

import pooch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Download data and checkpoints used in the SWITi paper."
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to save the downloaded data and checkpoints.",
    )
    return p.parse_args()


def main():
    """Download and extract data and checkpoints.

    Files are written under the directory selected by `--output-dir`.
    """
    args = parse_args()

    _ = pooch.retrieve(
        url="https://zenodo.org/records/22214478/files/switi_checkpoints.zip?download=1",
        known_hash="ca10202b8d1c4b67e14efa75d5b3c98cc4765930bfe666061fdee5ee05a9d7a0",
        processor=pooch.Unzip(extract_dir=args.output_dir / "checkpoints"),
        progressbar=True,
    )
    _ = pooch.retrieve(
        url="https://zenodo.org/records/22213583/files/switi_data.zip?download=1",
        known_hash="5a06d92155338527556aaadae7ad62ef6ca3b8db4f0a4906f16815c4354339bf",
        processor=pooch.Unzip(extract_dir=args.output_dir / "data"),
        progressbar=True,
    )


if __name__ == "__main__":
    main()
