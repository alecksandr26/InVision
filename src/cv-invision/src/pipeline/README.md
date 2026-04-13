# The InVision PipeLine Arch


## Simple Arch
```text
┌─────────────────────────────────────────────────────┐
│                  InVision Pipeline                  │
│                                                     │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐ │
│  │  Frame   │───▶│  YOLO    │───▶│   DeepSort   │ │
│  │  Reader  │     │Inference │     │   Tracker    │ │
│  └──────────┘     └──────────┘     └──────────────┘ │
│       │                                  │          │
│       │                                  ▼          │
│       │                         ┌──────────────┐    │
│       │                         │   Results    │    │
│       │                         │   Output     │    │
│       │                         └──────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Full Arch

``` text
Pipeline Architecture:

┌─────────────────────────────────────────────────────────┐
│                    Pipeline Factory                     │
│                                                         │
│  build_detection_pipeline()  ← just detector            │
│  build_tracking_pipeline()   ← detector + tracker ✅    │
│  build_full_pipeline()       ← detector + tracker +     │
│                                 renderer                │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                   Frame Source                          │
│   (abstracted — same interface for all sources)         │
│                                                         │
│  VideoFileSource("video.mp4")                           │
│  CameraSource(0)              ← Pi camera               │
│  StreamSource("rtsp://...")   ← RTSP stream             │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                  Frame Drop Policy                      │
│                                                         │
│  DROP_OLDEST  ← drop oldest frame if queue full         │
│  DROP_NEWEST  ← drop newest frame if queue full         │
│  BLOCK        ← wait until queue has space              │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  Reader  │──▶│ Detector │──▶│ Tracker  │──▶│ Output  │
│ Stage    │    │ Stage    │    │ Stage    │    │ Stage   │
└──────────┘    └──────────┘    └──────────┘    └─────────┘
```


## Threads

``` text
Thread 1: read frames    → Queue 1
Thread 2: YOLO inference → Queue 2
Thread 3: DeepSort       → Queue 3
Thread 4: render/display → output
```



