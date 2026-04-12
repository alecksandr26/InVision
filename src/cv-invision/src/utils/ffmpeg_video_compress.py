import subprocess
import argparse
from pathlib import Path

from .const import DETECTION_INPUT_IMAGE_SIZE, VIDEO_EXTENSIONS

"""
# basic
python ffmpeg_video_compress.py --input demo-examples/my_video.mp4

# with custom output
python ffmpeg_video_compress.py --input demo-examples/my_video.mp4 --output demo-examples/my_video_640.mp4

# aggressive compression for GitHub
python ffmpeg_video_compress.py -i demo-examples/my_video.mp4 -o demo-examples/my_video_640.mp4 --crf 35 --preset fast
"""


def check_ffmpeg():
    """
    Checks if ffmpeg is installed on the system.
    If not, prints installation instructions for common systems and exits.
    """
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)

    if result.returncode != 0:
        print("❌ ffmpeg is not installed or not in PATH.")
        print()
        print("📦 Install it with:")
        print("   Arch Linux  → sudo pacman -S ffmpeg")
        print("   Ubuntu      → sudo apt install ffmpeg")
        print("   macOS       → brew install ffmpeg")
        print("   Windows     → https://ffmpeg.org/download.html")
        raise SystemExit(1)

    print("✅ ffmpeg found!")


def validate_video_extension(input_path: Path) -> None:
    """
    Validates that the input file has a supported video extension.

    Args:
        input_path: Path to the input video file

    Raises:
        SystemExit: If the file extension is not supported
    """
    ext = input_path.suffix.lower()

    if ext not in VIDEO_EXTENSIONS:
        print(f"❌ Unsupported file extension: '{ext}'")
        print(f"📋 Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        raise SystemExit(1)

    print(f"✅ Video format '{ext}' is supported!")


def scale_and_compress_video(
    input_path  : str,
    output_path : str  = None,
    resolution  : tuple = (DETECTION_INPUT_IMAGE_SIZE, DETECTION_INPUT_IMAGE_SIZE),
    crf         : int  = 28,
    preset      : str  = "fast"
):
    """
    Scales and compresses a video using ffmpeg for lightweight GitHub storage.
    Only accepts files with supported video extensions defined in VIDEO_EXTENSIONS.

    Args:
        input_path  : Path to the input video file
        output_path : Path to save the output. Defaults to <input>_compressed.mp4
        resolution  : Target resolution as (width, height). Default (640, 640)
        crf         : Constant Rate Factor — lower = better quality, bigger file
                      18 = high quality | 28 = balanced | 35 = aggressive compression
        preset      : ffmpeg encoding preset — faster = bigger file, slower = smaller
                      options: ultrafast, fast, medium, slow

    Returns:
        output_path: Path to the compressed video

    Requirements:
        ffmpeg must be installed on the system:
          Arch Linux  → sudo pacman -S ffmpeg
          Ubuntu      → sudo apt install ffmpeg
          macOS       → brew install ffmpeg
          Windows     → https://ffmpeg.org/download.html
    """
    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / f"{input_path.stem}_compressed.mp4"

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: '{input_path}'")

    # Validate extension before doing anything else
    validate_video_extension(input_path)

    width, height = resolution

    cmd = [
        "ffmpeg",
        "-i",      str(input_path),
        "-vf",     f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v",    "libx264",        # H.264 codec — widely compatible
        "-crf",    str(crf),         # compression level
        "-preset", preset,           # encoding speed vs size tradeoff
        "-c:a",    "aac",            # audio codec
        "-b:a",    "64k",            # lower audio bitrate for lighter file
        "-movflags", "+faststart",   # web optimized — metadata at start of file
        str(output_path),
        "-y"                         # overwrite output if exists
    ]

    print(f"🎬 Scaling and compressing '{input_path.name}'...")
    print(f"⚙️  Format     : {input_path.suffix.lower()}")
    print(f"⚙️  Resolution : {width}x{height}")
    print(f"⚙️  CRF        : {crf}  (18=high quality | 28=balanced | 35=aggressive)")
    print(f"⚙️  Preset     : {preset}")
    print(f"⚙️  Output     : {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ffmpeg error:\n{result.stderr}")
        raise RuntimeError("ffmpeg processing failed")

    original_mb   = input_path.stat().st_size / 1024**2
    compressed_mb = output_path.stat().st_size / 1024**2
    savings       = (1 - compressed_mb / original_mb) * 100

    print(f"✅ Done!")
    print(f"📦 {original_mb:.1f} MB → {compressed_mb:.1f} MB ({savings:.1f}% smaller)")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 Scale and compress a video to 640x640 for GitHub using ffmpeg.",
        epilog=f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}"  # ← shown in --help
    )
    parser.add_argument(
        "--input",  "-i",
        type     = str,
        required = True,
        help     = "Path to the input video file (e.g. demo-examples/video.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = None,
        help    = "Path to save the compressed video. Defaults to <input>_compressed.mp4"
    )
    parser.add_argument(
        "--crf",
        type    = int,
        default = 28,
        help    = "Compression quality: 18=high, 28=balanced, 35=aggressive (default: 28)"
    )
    parser.add_argument(
        "--preset",
        type    = str,
        default = "fast",
        choices = ["ultrafast", "fast", "medium", "slow"],
        help    = "Encoding speed preset (default: fast)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_ffmpeg()

    scale_and_compress_video(
        input_path  = args.input,
        output_path = args.output,
        crf         = args.crf,
        preset      = args.preset,
    )
