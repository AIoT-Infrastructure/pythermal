#!/usr/bin/env python3
"""
People Counting with Thermal Camera (Adaptive Version) + Straight Right Door Region
- uses adaptive detection (environment temp)
- cluster -> merge cluster members into one bounding box -> track -> count
- visualization blends YUYV frame with thermal colormap
"""

import cv2
import time
import numpy as np
import argparse
from collections import deque

from pythermal import (
    ThermalCapture,
    detect_object_centers,
    detect_humans_adaptive,
    cluster_objects,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from pythermal.utils import estimate_environment_temperature_v1


# -------------------------
# Tracking / Counting Class (straight door)
# -------------------------
class PersonTrack:
    def __init__(self, track_id, center, box_wh=None):
        self.id = track_id
        self.centroids = deque(maxlen=30)
        self.centroids.append(center)
        self.box_wh = box_wh if box_wh is not None else (20, 20)
        self.lost_frames = 0
        self.inside = False
        self.counted_enter = False
        self.counted_exit = False

    def update(self, center, box_wh=None):
        self.centroids.append(center)
        if box_wh is not None:
            self.box_wh = box_wh
        self.lost_frames = 0

    def mark_lost(self):
        self.lost_frames += 1

    def smoothed_centroid(self, last_n=5):
        arr = np.array(self.centroids)
        n = min(len(arr), last_n)
        return tuple(arr[-n:].mean(axis=0))


class PeopleCounterStraightDoor:
    def __init__(self, frame_width=WIDTH, door_width=10, max_track_distance=80, max_lost_frames=15):
        self.frame_width = frame_width
        self.door_width = door_width
        self.max_track_distance = max_track_distance
        self.max_lost_frames = max_lost_frames

        self.next_id = 1
        self.tracks = []
        self.inside_count = 0
        self.total_entered = 0
        self.total_exited = 0

    def _scale_temp_to_frame(self, tx, ty):
        fx = int((tx / float(TEMP_WIDTH)) * WIDTH)
        fy = int((ty / float(TEMP_HEIGHT)) * HEIGHT)
        fx = max(0, min(WIDTH - 1, fx))
        fy = max(0, min(HEIGHT - 1, fy))
        return fx, fy

    def _point_in_door(self, fx, fy):
        """Check if the point is inside the straight door region on the right."""
        door_left = WIDTH - self.door_width
        door_right = WIDTH - 1
        return door_left <= fx <= door_right

    def update(self, merged_clusters):
        """
        merged_clusters: list of dicts:
          {
            'cx_temp': center_x in TEMP coords,
            'cy_temp': center_y in TEMP coords,
            'fx': scaled x in frame coords,
            'fy': scaled y in frame coords,
            'w_temp': width in TEMP coords,
            'h_temp': height in TEMP coords,
            'w_frame': width in frame coords,
            'h_frame': height in frame coords
          }
        """
        detections = []
        for m in merged_clusters:
            detections.append((m['cx_temp'], m['cy_temp'], m['fx'], m['fy'], m['w_frame'], m['h_frame']))

        # Mark all existing tracks as potentially lost
        for t in self.tracks:
            t.mark_lost()

        # Match detections to existing tracks (nearest centroid)
        used_det = set()
        for det_idx, (_, _, fx, fy, w_f, h_f) in enumerate(detections):
            best_track = None
            best_dist = float('inf')
            for t in self.tracks:
                last_x, last_y = t.smoothed_centroid()
                dist = np.hypot(fx - last_x, fy - last_y)
                if dist < best_dist:
                    best_dist = dist
                    best_track = t
            if best_track and best_dist < self.max_track_distance:
                best_track.update((fx, fy), box_wh=(w_f, h_f))
                used_det.add(det_idx)
            else:
                # create new track
                cx_temp, cy_temp, fx_new, fy_new, w_new, h_new = detections[det_idx]
                new_t = PersonTrack(self.next_id, (fx_new, fy_new), box_wh=(w_new, h_new))
                new_t.inside = self._point_in_door(fx_new, fy_new)
                self.next_id += 1
                self.tracks.append(new_t)
                used_det.add(det_idx)

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.lost_frames < self.max_lost_frames]

        # Count enter/exit
        for t in self.tracks:
            if len(t.centroids) == 0:
                continue
            x_now, y_now = t.smoothed_centroid()
            x_now, y_now = int(x_now), int(y_now)
            currently_inside = self._point_in_door(x_now, y_now)

            if (not t.inside) and currently_inside:
                self.inside_count += 1
                self.total_entered += 1
                t.counted_enter = True
            elif t.inside and (not currently_inside):
                self.inside_count = max(0, self.inside_count - 1)
                self.total_exited += 1
                t.counted_exit = True

            t.inside = currently_inside

        return self.inside_count

    def draw(self, frame, alpha_mask=True):
        overlay = frame.copy()
        mask_color = (0, 200, 0)
        door_left = WIDTH - self.door_width
        cv2.rectangle(overlay, (door_left, 0), (WIDTH - 1, HEIGHT), mask_color, -1)

        if alpha_mask:
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        else:
            frame[:] = overlay

        # draw door outline
        outline_color = (0, 200, 0)
        cv2.rectangle(frame, (door_left, 0), (WIDTH - 1, HEIGHT), outline_color, 2)

        # draw tracks
        for t in self.tracks:
            x, y = t.smoothed_centroid()
            xi, yi = int(x), int(y)
            color = (255, 255, 255)
            cv2.circle(frame, (xi, yi), 5, color, -1)
            cv2.putText(frame, f"ID{t.id}", (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            status_color = (0, 255, 0) if t.inside else (0, 0, 255)
            cv2.circle(frame, (xi - 8, yi - 8), 6, status_color, -1)

            # draw estimated box if available
            try:
                bw, bh = int(t.box_wh[0]), int(t.box_wh[1])
                x0 = xi - bw // 2
                y0 = yi - bh // 2
                cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (200, 200, 200), 1)
            except Exception:
                pass

        # draw counts
        cv2.putText(frame, f"Inside: {self.inside_count}", (10, HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Entered: {self.total_entered}  Exited: {self.total_exited}", (10, HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return frame


# -------------------------
# Visualization helper
# -------------------------
def generate_colors(n: int):
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        hue = int(180 * i / max(1, n))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors


def visualize_clustered_objects(
    temp_array: np.ndarray,
    yuyv_frame: np.ndarray,
    objects: list,
    clusters: list,
    min_temp: float,
    max_temp: float
) -> np.ndarray:
    """
    Visualize merged clusters: draw ONE merged bounding box per cluster (merging member boxes).
    """
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    temp_celsius = temp_array
    if temp_array.dtype == np.uint16:
        raw_min = np.min(temp_array)
        raw_max = np.max(temp_array)
        raw_range = raw_max - raw_min
        if raw_range > 0:
            normalized = (temp_array.astype(np.float32) - raw_min) / raw_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
        else:
            temp_celsius = np.full_like(temp_array, (min_temp + max_temp) / 2.0, dtype=np.float32)

    temp_upscaled = cv2.resize(temp_celsius, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    temp_range = max_temp - min_temp
    if temp_range > 0:
        normalized = ((temp_upscaled - min_temp) / temp_range) * 255.0
        normalized = normalized.clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    temp_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_frame, 0.5, temp_colored, 0.5, 0)

    # Colors for clusters
    cluster_colors = generate_colors(len(clusters))

    # Draw merged bounding box per cluster
    for cluster_idx, cluster in enumerate(clusters):
        color = cluster_colors[cluster_idx] if cluster_idx < len(cluster_colors) else (255,255,255)
        if not cluster:
            continue

        # compute merged bounding box in TEMP coords
        lefts   = [o.center_x - (o.width / 2.0) for o in cluster]
        rights  = [o.center_x + (o.width / 2.0) for o in cluster]
        tops    = [o.center_y - (o.height / 2.0) for o in cluster]
        bottoms = [o.center_y + (o.height / 2.0) for o in cluster]

        min_left = min(lefts)
        max_right = max(rights)
        min_top = min(tops)
        max_bottom = max(bottoms)

        # merged box center & size in TEMP coords
        merged_cx_temp = (min_left + max_right) / 2.0
        merged_cy_temp = (min_top + max_bottom) / 2.0
        merged_w_temp = max_right - min_left
        merged_h_temp = max_bottom - min_top

        # scale to frame coords
        center_x = int((merged_cx_temp / TEMP_WIDTH) * WIDTH)
        center_y = int((merged_cy_temp / TEMP_HEIGHT) * HEIGHT)
        width = int((merged_w_temp / TEMP_WIDTH) * WIDTH)
        height = int((merged_h_temp / TEMP_HEIGHT) * HEIGHT)

        # draw merged bounding box
        x = int(center_x - width / 2)
        y = int(center_y - height / 2)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)

        # draw center and label
        cv2.circle(overlay, (center_x, center_y), 6, color, -1)
        label = f"C{cluster_idx}"
        cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show average temperature for cluster
        try:
            temps = [o.avg_temperature for o in cluster]
            temp_label = f"{np.mean(temps):.0f}C"
            cv2.putText(overlay, temp_label, (x, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception:
            pass

    return overlay


def build_merged_cluster_list(clusters):
    """
    Convert cluster list of DetectedObject lists into merged cluster dicts that contain:
      merged temp-center, merged temp-size, and scaled frame coordinates.
    """
    merged = []
    for cluster in clusters:
        if not cluster:
            continue

        lefts   = [o.center_x - (o.width / 2.0) for o in cluster]
        rights  = [o.center_x + (o.width / 2.0) for o in cluster]
        tops    = [o.center_y - (o.height / 2.0) for o in cluster]
        bottoms = [o.center_y + (o.height / 2.0) for o in cluster]

        min_left = min(lefts)
        max_right = max(rights)
        min_top = min(tops)
        max_bottom = max(bottoms)

        merged_cx_temp = (min_left + max_right) / 2.0
        merged_cy_temp = (min_top + max_bottom) / 2.0
        merged_w_temp = max_right - min_left
        merged_h_temp = max_bottom - min_top

        fx = int((merged_cx_temp / TEMP_WIDTH) * WIDTH)
        fy = int((merged_cy_temp / TEMP_HEIGHT) * HEIGHT)
        w_frame = int((merged_w_temp / TEMP_WIDTH) * WIDTH)
        h_frame = int((merged_h_temp / TEMP_HEIGHT) * HEIGHT)

        merged.append({
            'cx_temp': merged_cx_temp,
            'cy_temp': merged_cy_temp,
            'w_temp': merged_w_temp,
            'h_temp': merged_h_temp,
            'fx': fx,
            'fy': fy,
            'w_frame': max(8, w_frame),
            'h_frame': max(8, h_frame),
        })
    return merged


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive people count (straight door)")
    parser.add_argument("source", nargs="?", default=None, help="device index or .tseq file")
    parser.add_argument("--adaptive", action="store_true", help="use adaptive detection")
    parser.add_argument("--face-only", action="store_true", help="face-only adaptive mode")
    parser.add_argument("--env-temp", type=float, default=None, help="optional environment temp (C)")
    parser.add_argument("--min-temp-above-env", type=float, default=2.0)
    parser.add_argument("--max-temp-limit", type=float, default=42.0)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--door-width", type=int, default=90, help="width of straight door (px)")
    parser.add_argument("--cluster-distance", type=float, default=50.0, help="max distance (TEMP coords) to merge cluster members")
    args = parser.parse_args()

    print("Initializing ThermalCapture ...")
    capture = ThermalCapture(args.source)
    is_recorded = capture.is_recorded

    window_name = "Thermal People Count (Straight Door)" + (" (Replay)" if is_recorded else "")
    people_counter = PeopleCounterStraightDoor(frame_width=WIDTH, door_width=args.door_width)

    frame_count = 0
    paused = False
    last_display_time = time.time()

    try:
        while True:
            if not paused:
                if not capture.has_new_frame():
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {frame_count} frames")
                        break
                    time.sleep(0.01)
                    continue

                meta = capture.get_metadata()
                temp_array = capture.get_temperature_array()
                yuyv_frame = capture.get_yuyv_frame()

                if meta is None or temp_array is None or yuyv_frame is None:
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {frame_count} frames")
                        break
                    continue

                # estimate environment temp if needed
                if args.adaptive:
                    if args.env_temp is not None:
                        env_temp = args.env_temp
                    else:
                        env_temp = estimate_environment_temperature_v1(temp_array, meta.min_temp, meta.max_temp)

                    alpha_min = 0.5 if args.face_only else 0.4
                    alpha_max = 0.7

                    objects = detect_humans_adaptive(
                        temp_array=temp_array,
                        min_temp=meta.min_temp,
                        max_temp=meta.max_temp,
                        environment_temp=env_temp,
                        min_area=args.min_area,
                        min_temp_above_env=args.min_temp_above_env,
                        max_temp_limit=args.max_temp_limit,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                    )
                else:
                    objects = detect_object_centers(
                        temp_array=temp_array,
                        min_temp=meta.min_temp,
                        max_temp=meta.max_temp,
                        temp_min=30.0,
                        temp_max=40.0,
                        min_area=args.min_area,
                    )

                # Cluster objects with a larger merge radius so bodily parts join
                clusters = cluster_objects(objects, max_distance=args.cluster_distance)

                # Build merged-cluster list (one merged box per cluster) used for tracking
                merged_clusters = build_merged_cluster_list(clusters)

                # Visualize using merged boxes
                vis = visualize_clustered_objects(
                    temp_array=temp_array,
                    yuyv_frame=yuyv_frame,
                    objects=objects,
                    clusters=clusters,
                    min_temp=meta.min_temp,
                    max_temp=meta.max_temp
                )

                # Update person counter with merged clusters and draw counts/tracks
                inside = people_counter.update(merged_clusters)
                vis = people_counter.draw(vis)

                info_lines = [
                    f"Frame: {meta.seq}",
                    f"Objects: {len(objects)}, Clusters: {len(clusters)}, Merged: {len(merged_clusters)}",
                ]
                if args.adaptive:
                    info_lines.append(f"Adaptive (Room: {env_temp:.1f}C)")
                else:
                    info_lines.append("Range: fixed")

                if objects:
                    temps = [o.avg_temperature for o in objects]
                    info_lines.append(f"Detected: {min(temps):.0f}C - {max(temps):.0f}C")

                y_offset = 15
                for line in info_lines:
                    cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    y_offset += 18

                cv2.imshow(window_name, vis)
                capture.mark_frame_read()
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' ') and is_recorded:
                paused = not paused
                print("Paused" if paused else "Resumed")

        print(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
