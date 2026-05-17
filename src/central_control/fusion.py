import numpy as np

class SensorFusionTracker:
    def __init__(self, h_fov=62.2, image_width=640):
        self.h_fov = h_fov
        self.width = image_width

    def fuse_spatial_data(self, frame_data, scan_points):
        """
        Calculates geometric intersection of DeepSort tracking tracks 
        and raw laser points array shape (N, 3).
        """
        # Store computed tracking targets dynamically inside FrameData
        frame_data.spatial_targets = {}

        if scan_points.shape[0] == 0 or not frame_data.tracks:
            return frame_data

        for track in frame_data.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_id = track.det_class

            # Get bounding box [x1, y1, x2, y2]
            bbox = track.get_det_supplementary()
            if bbox is None:
                bbox = track.to_ltrb()

            x1, y1, x2, y2 = map(int, bbox)

            # 1. Map pixel center to a real-world angular horizontal offset
            box_center_x = (x1 + x2) / 2.0
            target_angle = ((box_center_x / self.width) - 0.5) * self.h_fov

            # 2. Approximate object's angular width slice
            box_width_pixels = x2 - x1
            angle_span = (box_width_pixels / self.width) * self.h_fov
            
            angle_min = target_angle - (angle_span / 2.0)
            angle_max = target_angle + (angle_span / 2.0)

            # 3. Slice matching LiDAR array: points[:, 0] is angle_deg, points[:, 1] is dist_cm
            matched_indices = (scan_points[:, 0] >= angle_min) & (scan_points[:, 0] <= angle_max)
            target_points = scan_points[matched_indices]

            if target_points.shape[0] > 0:
                # Median filters out edge clipping and stray dust reflections
                median_distance_cm = np.median(target_points[:, 1])
                distance_meters = median_distance_cm / 100.0

                frame_data.spatial_targets[track_id] = {
                    "distance_m": distance_meters,
                    "angle_deg": target_angle,
                    "class_id": class_id,
                    "bbox": (x1, y1, x2, y2)
                }

        return frame_data
