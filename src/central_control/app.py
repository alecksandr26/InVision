#!/usr/bin/env python3
import sys
import cv2
import queue
import time
import argparse
import serial  # <--- Added pyserial import

# Modules compiled natively inside your workspace
from lidar_invision import LidarManager, Config as LidarConfig
from cv_invision.pipeline.factory import build_full_pipeline, build_headless_pipeline
from cv_invision.pipeline.sources import CameraSource, PiCameraSource
from cv_invision.model.detection_model import load_detection_model_prod
from cv_invision.model.tracker import build_tracker
from cv_invision.model.const import CLASS_NAMES
from central_control.fusion import SensorFusionTracker
from cv_invision.utils.log import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Central Control Orchestrator")
    parser.add_argument("--no-ui", action="store_true", help="Use headless pipeline builder & skip windows to maximize performance FPS")
    parser.add_argument("--picamera", action="store_true", help="Use PiCameraSource instead of standard CameraSource")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.debug("🚀 Initializing Central Control Orchestrator...")
    
    # --- UART Setup ---
    # /dev/serial0 automatically targets the primary hardware UART pins 14/15
    try:
        ser = serial.Serial(
            port="/dev/serial0",
            baudrate=115200,
            timeout=1.0,
            write_timeout=1.0
        )
        logger.debug("📟 Hardware UART initialized on /dev/serial0 at 115200 baud.")
    except Exception as e:
        logger.error(f"❌ Failed to open UART interface: {e}")
        sys.exit(1)

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
    source = PiCameraSource() if args.picamera else CameraSource(device_id=0)
    model = load_detection_model_prod()
    tracker = build_tracker()
    
    if args.no_ui:
        logger.debug("⚡ [PERF MODE] Constructing lightweight headless pipeline...")
        stages, final_out_queue = build_headless_pipeline(source, model, tracker)
    else:
        logger.debug("📺 [DISPLAY MODE] Constructing full visual tracking pipeline...")
        stages, final_out_queue = build_full_pipeline(source, model, tracker)
    
    fusion_engine = SensorFusionTracker(h_fov=62.2, image_width=640)
    tracker_stage = stages[2]
    
    def lidar_hook(frame_data):
        scan = lidar.fetch_scan(wait_for_fresh=False)
        return fusion_engine.fuse_spatial_data(frame_data, scan.points)
    
    tracker_stage.register_callback(lidar_hook)
    
    for stage in stages:
        stage.start()
        
    logger.debug("🟢 System running smoothly. Press Ctrl+C in terminal to stop if running with --no-ui.")
    
    prev_time = None
    fps_smooth = 0.0
    alpha = 0.1
    
    last_signal_sent = None
    last_perf_log_time = time.perf_counter()
    PERF_LOG_INTERVAL_SEC = 3.0

    try:
        while True:
            try:
                queue_start_time = time.perf_counter()
                data = final_out_queue.get(timeout=0.1)
                if data is None:
                    break
                
                current_time = time.perf_counter()
                queue_latency_ms = (current_time - queue_start_time) * 1000.0
                
                if prev_time is not None:
                    time_delta = current_time - prev_time
                    if time_delta > 0:
                        instant_fps = 1.0 / time_delta
                        fps_smooth = (alpha * instant_fps) + ((1 - alpha) * fps_smooth)
                else:
                    fps_smooth = 30.0
                
                prev_time = current_time
                spatial_targets = getattr(data, 'spatial_targets', {})
                
                collision_imminent = 0
                COLLISION_THRESHOLD_M = 1.2
                COLLISION_ANGLE_DEG = 15.0

                for track_id, info in spatial_targets.items():
                    dist = info["distance_m"]
                    angle = info["angle_deg"]
                    if (dist < COLLISION_THRESHOLD_M) and (abs(angle) < COLLISION_ANGLE_DEG):
                        collision_imminent = 1
                        break
                
                # --- MODIFIED CONTROL SIGNAL TRANS-MISSION ---
                if collision_imminent == 1:
                    logger.debug(f"🚨 [ALERT] CONTROL SIGNAL -> 1 (Collision Imminent!)")
                    try:
                        logger.debug(f"📟 [UART TX] Attempting to write payload: b'1' to /dev/serial0")
                        
                        ser.write(b'1')  # Transmit state change byte over hardware pins
                        ser.flush()      # Block until all data is physically cleared out of buffers
                        
                        logger.info(f"✅ [UART TX SUCCESS] Sent b'1' byte successfully")
                        last_signal_sent = 1  # Only update state memory if write succeeds
                    except (serial.SerialException, serial.SerialTimeoutException) as uart_err:
                        logger.error(f"⚠️ [UART TX ERROR] Failed to send alert flag '1': {uart_err}")
                        # Optional: Flag system state as degraded, but keep pipeline alive
                elif collision_imminent == 0:
                    logger.debug(f"📡 [CLEAR] CONTROL SIGNAL -> 0 (Path is safe)")
                    try:
                        logger.info(f"📟 [UART TX] Attempting to write payload: b'0' to /dev/serial0")
                        
                        ser.write(b'0')
                        ser.flush()
                        
                        logger.info(f"✅ [UART TX SUCCESS] Sent b'0' byte successfully")
                        last_signal_sent = 0  # Only update state memory if write succeeds
                    except (serial.SerialException, serial.SerialTimeoutException) as uart_err:
                        logger.error(f"⚠️ [UART TX ERROR] Failed to send clear flag '0': {uart_err}")
                
                if current_time - last_perf_log_time >= PERF_LOG_INTERVAL_SEC:
                    mode_str = "HEADLESS MODE" if args.no_ui else "UI MODE"
                    logger.debug(f"📊 [{mode_str} PERF] Speed: {fps_smooth:.1f} FPS | Queue Latency: {queue_latency_ms:.1f}ms")
                    last_perf_log_time = current_time
                
                if not args.no_ui:
                    canvas = data.annotated if data.annotated is not None else data.frame
                    
                    for track_id, info in spatial_targets.items():
                        x1, y1, x2, y2 = info["bbox"]
                        dist = info["distance_m"]
                        raw_class_id = info["class_id"]
                        
                        is_target_dangerous = (dist < COLLISION_THRESHOLD_M) and (abs(info["angle_deg"]) < COLLISION_ANGLE_DEG)
                        alert_color = (0, 0, 255) if is_target_dangerous else (0, 255, 255)
                        
                        if raw_class_id is not None and 0 <= int(raw_class_id) < len(CLASS_NAMES):
                            label_name = CLASS_NAMES[int(raw_class_id)]
                        else:
                            label_name = "Unknown"
                        
                        status_tag = " [⚠️ DANGER]" if is_target_dangerous else ""
                        text = f"{label_name} (ID {track_id}) [{dist:.2f}m]{status_tag}"
                        
                        cv2.putText(canvas, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), alert_color, 2)
                    
                    cv2.putText(canvas, f"Pipeline Speed: {fps_smooth:.1f} FPS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(canvas, f"Queue Fetch: {queue_latency_ms:.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
                    
                    cv2.imshow("InVision Sensor Fusion Engine", canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        logger.debug("\n🛑 Halting via terminal command...")
    finally:
        logger.debug("🧹 Cleaning system resources...")
        for stage in stages: 
            stage.stop()
        for stage in stages: 
            stage.join()
        lidar.stop()
        source.release()
        if 'ser' in locals() and ser.is_open:
            ser.close()  # <--- Safely dismantle serial port access
        cv2.destroyAllWindows()
        logger.debug("🏁 System shutdown complete.")

if __name__ == "__main__":
    main()
