# InVision
Invisivision is a modular Electronics and Communications Engineering project that develops an assistive system for visually impaired individuals, enabling safer and more autonomous outdoor navigation. The prototype consists of a portable device built around three Raspberry Pi boards: a Raspberry Pi


## PiCamera2 Module installation 
### 1. Host OS Prerequisites

Run this globally on your Raspberry Pi:

```bash
sudo apt update && sudo apt install -y \
    libcamera-apps \
    libcamera-dev \
    python3-libcamera \
    python3-picamera2 \
    python3-kms

```

### 2. Environment Patch (Symlinks)

Run this from your project root (`~/Projects/InVision/`) to map the architecture drivers into your `env/` virtual environment:

```bash
cd ~/Projects/InVision/

# Link libcamera binaries
ln -sf /usr/lib/python3/dist-packages/libcamera env/lib/python3.13/site-packages/
ln -sf /usr/lib/python3/dist-packages/_libcamera.cpython*.so env/lib/python3.13/site-packages/

# Link KMS display / pykms preview dependencies
ln -sf /usr/lib/python3/dist-packages/kms env/lib/python3.13/site-packages/
ln -sf /usr/lib/python3/dist-packages/pykms env/lib/python3.13/site-packages/
ln -sf /usr/lib/python3/dist-packages/_pykms.cpython*.so env/lib/python3.13/site-packages/

```

### 3. Verification & Execution

```bash
source env/bin/activate
python -c "import picamera2; import libcamera; print('🚀 Connected to hardware!')"

# Run pipeline
cd src/cv-invision/
python -m src.pipeline.test.pipeline --picamera

```
