#!/usr/bin/env python3
import sys
import cv2
import queue

# Modules compiled natively inside your workspace
from lidar_invision import LidarManager, Config as LidarConfig
from cv_invision.pipeline.factory import build_full_pipeline
from cv_invision.pipeline.sources import PiCameraSource # Native Pi Camera
from cv_invision.model.detection_model import load_detection_model_prod
from cv_invision.model.tracker import build_tracker

# 1. Import CLASS_NAMES registry map from your model constants file
from cv_invision.model.const import CLASS_NAMES

from central_control.fusion import SensorFusionTracker

def main():
    print("🚀 Initializing Central Control Orchestrator...")
    
    # Spin up C++ LiDAR background thread
    lidar = LidarManager()
    cfg = LidarConfig()
    cfg.port = "/dev/ttyUSB0"
    cfg.frequency = 10.0
    cfg.min_dist_cm = 10.0
    cfg.max_dist_cm = 400.0
    lidar.configure(cfg)
    lidar.start()
    
    # Build Vision Objects
    source = PiCameraSource() 
    model = load_detection_model_prod()
    tracker = build_tracker()
    
    # Assemble internal components via pipeline factory
    stages, final_out_queue = build_full_pipeline(source, model, tracker)

    print("Stages: ", stages)
    
    # Initialize Sensor Fusion Engine
    fusion_engine = SensorFusionTracker(h_fov=62.2, image_width=640)
    
    # ----------------------------------------------------------------
    # CALLBACK 1: Hook to the Reader Stage (Index 0) to capture data
    # ----------------------------------------------------------------
    reader_stage = stages[0]
    
    def capture_lidar_snapshot_hook(frame_data):
        scan = lidar.fetch_scan(wait_for_fresh=False)
        frame_data.raw_scan = scan.points
        return frame_data
    
    reader_stage.register_callback(capture_lidar_snapshot_hook)
    print(f"⚓ Capture snapshot callback registered to: {reader_stage.name}")
    
    # ----------------------------------------------------------------
    # CALLBACK 2: Hook to the Tracker Stage (Index 2) to compute fusion
    # ----------------------------------------------------------------
    tracker_stage = stages[2]
    
    def process_fusion_hook(frame_data):
        scan_points = getattr(frame_data, 'raw_scan', None)
        print("scanned points: ", scan_points)
        if scan_points is None:
            return frame_data
        return fusion_engine.fuse_spatial_data(frame_data, scan_points)

    tracker_stage.register_callback(process_fusion_hook)
    print(f"⚓ Process fusion callback registered to: {tracker_stage.name}")
    # ----------------------------------------------------------------
    
    # Activate Worker Pipeline Threads
    for stage in stages:
        stage.start()
        
    print("🟢 System running smoothly. Press 'q' on the output window to stop.")
    
    try:
        while True:
            try:
                data = final_out_queue.get(timeout=0.1)
                if data is None:
                    break
                
                canvas = data.annotated if data.annotated is not None else data.frame
                
                # Defensive check using getattr to prevent AttributeError 
                spatial_targets = getattr(data, 'spatial_targets', {})
                
                # Render our top-level telemetry HUD
                for track_id, info in spatial_targets.items():
                    print("info: ", info)
                    x1, y1, x2, y2 = info["bbox"]
                    dist = info["distance_m"]
                    raw_class_id = info["class_id"]
                    
                    # 2. MAP INDEX LOOKUP: Safely map class ID integer to its string name string
                    if raw_class_id is not None and 0 <= int(raw_class_id) < len(CLASS_NAMES):
                        label_name = CLASS_NAMES[int(raw_class_id)]
                    else:
                        label_name = "Unknown"
                    
                    # Include the dynamic class label name directly inside HUD text
                    text = f"{label_name} (ID {track_id}) [{dist:.2f}m]"
                    cv2.putText(canvas, text, (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # --- Control Signal Routine (1 or 0) ---
                collision_imminent = 0
                for track_id, info in spatial_targets.items():
                    if info["distance_m"] < 1.2 and abs(info["angle_deg"]) < 15.0:
                        collision_imminent = 1
                        break
                
                # Output the binary status to console
                print(f"📡 CONTROL SIGNAL -> {collision_imminent}")
                
                cv2.imshow("InVision Sensor Fusion Engine", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Halting via terminal command...")
    finally:
        print("🧹 Cleaning system resources...")
        for stage in stages: stage.stop()
        for stage in stages: stage.join()
        lidar.stop()
        source.release()
        cv2.destroyAllWindows()
        print("🏁 System shutdown complete.")

if __name__ == "__main__":
    main()
