import gdown
import zipfile
import argparse
from pathlib import Path

from .const import TEST_SAMPLES_DIR_URL, TEST_SAMPLES_DIR


def download_test_samples(
    url        : str  = TEST_SAMPLES_DIR_URL,
    output_dir : Path = TEST_SAMPLES_DIR,
):
    """
    Downloads the InVision test samples folder from Google Drive as a zip
    and extracts it locally for running inference and validation.

    Test samples contain raw .mp4 videos and/or images used to validate
    the InVision detection model pipeline. These are NOT training dataset files.

    Args:
        url       : Google Drive folder URL. Defaults to TEST_SAMPLES_DIR_URL from const.py
        output_dir: Local directory to extract the samples into. Defaults to demo-examples/raw/

    Requirements:
        gdown must be installed:
          pip install gdown
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "test_samples.zip"

    print(f"📥 Downloading test samples from Google Drive...")
    print(f"⚙️  URL        : {url}")
    print(f"⚙️  Output dir : {output_dir}")

    # Download entire Google Drive folder as a single zip file
    gdown.download_folder(
        url         = url,
        output      = str(output_dir),
        quiet       = False,
        use_cookies = False,
    )

    # Unzip if a zip file was produced
    if zip_path.exists():
        print(f"📦 Extracting '{zip_path.name}'...")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)

        zip_path.unlink()  # remove zip after extraction to save space
        print(f"🗑️  Removed zip file after extraction")

    print(f"✅ Test samples ready at: '{output_dir}'")

    # List what was downloaded
    files = [f for f in output_dir.rglob("*") if f.is_file()]  # ← only actual files, no dirs
    print(f"📂 {len(files)} file(s) available:")
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / 1024**2
            print(f"   {f.relative_to(output_dir)} ({size_mb:.1f} MB)")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="📥 Download InVision test samples (videos/images) from Google Drive."
    )
    parser.add_argument(
        "--url", "-u",
        type    = str,
        default = TEST_SAMPLES_DIR_URL,
        help    = "Google Drive folder URL (default: TEST_SAMPLES_DIR_URL from const.py)"
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = str(TEST_SAMPLES_DIR),
        help    = f"Local directory to extract samples into (default: {TEST_SAMPLES_DIR})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download_test_samples(
        url        = args.url,
        output_dir = Path(args.output),
    )
