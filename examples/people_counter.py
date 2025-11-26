# people_counter.py
import numpy as np
import cv2
from collections import deque
from pythermal import WIDTH, HEIGHT, TEMP_WIDTH, TEMP_HEIGHT

class PersonTrack:
    def __init__(self, track_id, center):
        self.track_id = track_id
        self.centers = deque(maxlen=10)
        self.centers.append(center)
        self.last_seen = 0

    def update(self, center):
        self.centers.append(center)
        self.last_seen = 0

    def get_last_center(self):
        return self.centers[-1]


class PeopleCounter:
    def __init__(self, frame_width=240, enter_line_ratio=0.33, exit_line_ratio=0.66, max_distance=35):
        self.frame_width = frame_width
        self.enter_line = int(frame_width * enter_line_ratio)
        self.exit_line = int(frame_width * exit_line_ratio)
        self.max_distance = max_distance

        self.next_id = 1
        self.tracks = {}      # track_id → PersonTrack
        self.inside_count = 0
        self.total_entered = 0
        self.total_exited = 0

    def _scale_coord(self, temp_x, temp_y):
        """Scale from 96x96 to 240x240 coordinates."""
        x = int((temp_x / TEMP_WIDTH) * WIDTH)
        y = int((temp_y / TEMP_HEIGHT) * HEIGHT)
        return x, y

    def update_tracks(self, clusters):
        """Match clusters to tracks."""
        detected_centers = []

        # Extract cluster center points
        for cluster in clusters:
            # Use cluster mean point
            xs = [obj.center_x for obj in cluster]
            ys = [obj.center_y for obj in cluster]
            cx = np.mean(xs)
            cy = np.mean(ys)
            detected_centers.append((cx, cy))

        # First increment "not seen" counters
        for tid in list(self.tracks.keys()):
            self.tracks[tid].last_seen += 1
            if self.tracks[tid].last_seen > 10:
                del self.tracks[tid]

        # Match by nearest center
        for cx, cy in detected_centers:
            scaled_x, scaled_y = self._scale_coord(cx, cy)

            matched_id = None
            min_dist = 9999

            for tid, track in self.tracks.items():
                tx, ty = track.get_last_center()
                dist = np.hypot(tx - scaled_x, ty - scaled_y)
                if dist < self.max_distance and dist < min_dist:
                    matched_id = tid
                    min_dist = dist

            # If no match -> new track
            if matched_id is None:
                self.tracks[self.next_id] = PersonTrack(self.next_id, (scaled_x, scaled_y))
                matched_id = self.next_id
                self.next_id += 1
            else:
                self.tracks[matched_id].update((scaled_x, scaled_y))

        return detected_centers

    def count_logic(self):
        """Check track movement to determine enter/exit actions."""
        for tid, track in self.tracks.items():
            if len(track.centers) < 2:
                continue

            x_prev = track.centers[-2][0]
            x_now = track.centers[-1][0]

            # Enter (left → right)
            if x_prev < self.enter_line and x_now >= self.enter_line:
                self.inside_count += 1
                self.total_entered += 1

            # Exit (right → left)
            if x_prev > self.exit_line and x_now <= self.exit_line:
                self.inside_count = max(0, self.inside_count - 1)
                self.total_exited += 1

    def visualize(self, frame):
        """Draw tracking visualization on the 240x240 image."""
        # Draw lines
        cv2.line(frame, (self.enter_line, 0), (self.enter_line, HEIGHT), (0, 255, 0), 2)
        cv2.line(frame, (self.exit_line, 0), (self.exit_line, HEIGHT), (0, 0, 255), 2)

        # Draw tracking points + IDs
        for tid, track in self.tracks.items():
            x, y = track.get_last_center()
            cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
            cv2.putText(
                frame, f"ID {tid}", (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

        # Draw count
        cv2.putText(
            frame, f"Inside: {self.inside_count}",
            (10, HEIGHT - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Entered: {self.total_entered}, Exited: {self.total_exited}",
            (10, HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        return frame

    def process(self, clusters, frame):
        """Main entrypoint: update, count, and draw."""
        self.update_tracks(clusters)
        self.count_logic()
        return self.visualize(frame), self.inside_count
