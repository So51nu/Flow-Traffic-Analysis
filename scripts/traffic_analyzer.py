import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
import os
from pathlib import Path
import yt_dlp
from ultralytics import YOLO
import argparse

class TrafficFlowAnalyzer:
    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize YOLO model for vehicle detection
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Tracking variables
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30  # frames before removing a track
        
        # Lane counting
        self.lane_counts = {1: 0, 2: 0, 3: 0}
        self.counted_vehicles = {1: set(), 2: set(), 3: set()}
        
        # CSV data storage
        self.csv_data = []
        
        # Lane definitions (will be set based on frame size)
        self.lanes = {}
        
    def download_youtube_video(self, url, output_path="temp_video.mp4"):
        """Download YouTube video for processing"""
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for processing speed
            'outtmpl': output_path,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return output_path
    
    def define_lanes(self, frame_width, frame_height):
        """Define three lane regions as polygons"""
        # For a typical traffic camera view, divide into three vertical lanes
        lane_width = frame_width // 3
        
        # Lane 1 (left)
        self.lanes[1] = np.array([
            [0, 0],
            [lane_width, 0],
            [lane_width, frame_height],
            [0, frame_height]
        ], np.int32)
        
        # Lane 2 (center)
        self.lanes[2] = np.array([
            [lane_width, 0],
            [2 * lane_width, 0],
            [2 * lane_width, frame_height],
            [lane_width, frame_height]
        ], np.int32)
        
        # Lane 3 (right)
        self.lanes[3] = np.array([
            [2 * lane_width, 0],
            [frame_width, 0],
            [frame_width, frame_height],
            [2 * lane_width, frame_height]
        ], np.int32)
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using cv2.pointPolygonTest"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def get_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detections, frame_num, timestamp):
        """Simple tracking algorithm based on IoU matching"""
        if not detections:
            # Age all existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for detection in detections:
                bbox, conf, cls = detection
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'disappeared': 0,
                    'class': cls,
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'first_time': timestamp,
                    'last_time': timestamp
                }
                self.next_id += 1
            return [(track_id, data['bbox'], data['class']) for track_id, data in self.tracks.items()]
        
        # Match detections to existing tracks using IoU
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        matched_tracks = []
        used_detections = set()
        
        # Find best matches
        for i, track_id in enumerate(track_ids):
            best_iou = 0
            best_detection_idx = -1
            
            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue
                
                bbox, conf, cls = detection
                iou = self.calculate_iou(track_bboxes[i], bbox)
                
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_detection_idx = j
            
            if best_detection_idx != -1:
                # Update existing track
                bbox, conf, cls = detections[best_detection_idx]
                self.tracks[track_id]['bbox'] = bbox
                self.tracks[track_id]['disappeared'] = 0
                self.tracks[track_id]['last_frame'] = frame_num
                self.tracks[track_id]['last_time'] = timestamp
                matched_tracks.append((track_id, bbox, cls))
                used_detections.add(best_detection_idx)
            else:
                # Track not matched, increment disappeared counter
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] <= self.max_disappeared:
                    matched_tracks.append((track_id, self.tracks[track_id]['bbox'], self.tracks[track_id]['class']))
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detections:
                bbox, conf, cls = detection
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'disappeared': 0,
                    'class': cls,
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'first_time': timestamp,
                    'last_time': timestamp
                }
                matched_tracks.append((self.next_id, bbox, cls))
                self.next_id += 1
        
        # Remove tracks that have disappeared for too long
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
        
        return matched_tracks
    
    def assign_to_lane(self, centroid):
        """Assign vehicle to lane based on centroid position"""
        for lane_id, polygon in self.lanes.items():
            if self.point_in_polygon(centroid, polygon):
                return lane_id
        return None
    
    def process_video(self, youtube_url=None):
        """Main processing function"""
        # Download video if YouTube URL provided
        if youtube_url:
            print("Downloading video from YouTube...")
            self.video_path = self.download_youtube_video(youtube_url)
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Define lanes
        self.define_lanes(frame_width, frame_height)
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.output_dir / 'processed_video.mp4'),
            fourcc, fps, (frame_width, frame_height)
        )
        
        frame_num = 0
        start_time = time.time()
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_num / fps
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Extract vehicle detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in self.vehicle_classes and conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append(([int(x1), int(y1), int(x2), int(y2)], conf, cls))
            
            # Update tracks
            tracked_vehicles = self.update_tracks(detections, frame_num, timestamp)
            
            # Process each tracked vehicle
            for track_id, bbox, cls in tracked_vehicles:
                centroid = self.get_centroid(bbox)
                lane = self.assign_to_lane(centroid)
                
                if lane and track_id not in self.counted_vehicles[lane]:
                    # New vehicle in this lane
                    self.lane_counts[lane] += 1
                    self.counted_vehicles[lane].add(track_id)
                    
                    # Add to CSV data
                    self.csv_data.append({
                        'Vehicle_ID': track_id,
                        'Lane_No': lane,
                        'First_Frame': self.tracks[track_id]['first_frame'],
                        'Last_Frame': self.tracks[track_id]['last_frame'],
                        'First_Time_s': round(self.tracks[track_id]['first_time'], 2),
                        'Last_Time_s': round(self.tracks[track_id]['last_time'], 2),
                        'Vehicle_Type': self.class_names.get(cls, 'unknown')
                    })
            
            # Draw overlay on frame
            overlay_frame = self.draw_overlay(frame, tracked_vehicles)
            
            # Write frame to output video
            out.write(overlay_frame)
            
            frame_num += 1
            
            # Progress update
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_num) * (total_frames - frame_num)
                print(f"Progress: {progress:.1f}% - ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save CSV data
        self.save_csv()
        
        # Print summary
        self.print_summary()
        
        print(f"\nProcessing complete!")
        print(f"Output video: {self.output_dir / 'processed_video.mp4'}")
        print(f"CSV data: {self.output_dir / 'vehicle_counts.csv'}")
    
    def draw_overlay(self, frame, tracked_vehicles):
        """Draw lanes, vehicles, and counts on frame"""
        overlay = frame.copy()
        
        # Draw lane boundaries
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors
        for i, (lane_id, polygon) in enumerate(self.lanes.items()):
            cv2.polylines(overlay, [polygon], True, colors[i], 2)
            
            # Add lane labels
            label_pos = (polygon[0][0] + 10, polygon[0][1] + 30)
            cv2.putText(overlay, f"Lane {lane_id}: {self.lane_counts[lane_id]}", 
                       label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        
        # Draw tracked vehicles
        for track_id, bbox, cls in tracked_vehicles:
            x1, y1, x2, y2 = bbox
            centroid = self.get_centroid(bbox)
            lane = self.assign_to_lane(centroid)
            
            # Choose color based on lane
            if lane:
                color = colors[lane - 1]
            else:
                color = (128, 128, 128)  # Gray for unassigned
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw vehicle ID and type
            label = f"ID:{track_id} {self.class_names.get(cls, 'vehicle')}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw centroid
            cv2.circle(overlay, centroid, 3, color, -1)
        
        # Add total count
        total_count = sum(self.lane_counts.values())
        cv2.putText(overlay, f"Total Vehicles: {total_count}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return overlay
    
    def save_csv(self):
        """Save vehicle data to CSV file"""
        if self.csv_data:
            df = pd.DataFrame(self.csv_data)
            df = df.drop_duplicates(subset=['Vehicle_ID', 'Lane_No'])  # Remove duplicates
            df.to_csv(self.output_dir / 'vehicle_counts.csv', index=False)
            print(f"Saved {len(df)} vehicle records to CSV")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*50)
        print("TRAFFIC FLOW ANALYSIS SUMMARY")
        print("="*50)
        for lane_id, count in self.lane_counts.items():
            print(f"Lane {lane_id}: {count} vehicles")
        print(f"Total: {sum(self.lane_counts.values())} vehicles")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Analysis')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--youtube', type=str, help='YouTube URL')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.video and not args.youtube:
        # Default to the YouTube URL from the theory guide
        args.youtube = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    
    analyzer = TrafficFlowAnalyzer(args.video, args.output)
    
    try:
        analyzer.process_video(args.youtube)
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
