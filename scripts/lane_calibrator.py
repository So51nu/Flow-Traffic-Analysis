"""
Interactive tool to manually define lane boundaries for better accuracy
"""
import cv2
import numpy as np
import json
from pathlib import Path

class LaneCalibrator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.lanes = {1: [], 2: [], 3: []}
        self.current_lane = 1
        self.drawing = False
        self.frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing lane boundaries"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lanes[self.current_lane].append([x, y])
            cv2.circle(self.frame, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Lane Calibrator', self.frame)
            
    def calibrate(self):
        """Interactive lane boundary definition"""
        cap = cv2.VideoCapture(self.video_path)
        ret, self.frame = cap.read()
        
        if not ret:
            print("Could not read video")
            return None
            
        cv2.namedWindow('Lane Calibrator')
        cv2.setMouseCallback('Lane Calibrator', self.mouse_callback)
        
        print("Lane Calibration Instructions:")
        print("- Click to define lane boundaries")
        print("- Press 1, 2, 3 to switch between lanes")
        print("- Press 'c' to clear current lane")
        print("- Press 's' to save and exit")
        print("- Press 'q' to quit without saving")
        
        while True:
            display_frame = self.frame.copy()
            
            # Draw existing points
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for lane_id, points in self.lanes.items():
                if points:
                    color = colors[lane_id - 1]
                    for i, point in enumerate(points):
                        cv2.circle(display_frame, tuple(point), 3, color, -1)
                        if i > 0:
                            cv2.line(display_frame, tuple(points[i-1]), tuple(point), color, 2)
                    
                    # Close polygon if enough points
                    if len(points) > 2:
                        cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), color, 2)
            
            # Show current lane
            cv2.putText(display_frame, f"Current Lane: {self.current_lane}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Lane Calibrator', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                self.current_lane = 1
            elif key == ord('2'):
                self.current_lane = 2
            elif key == ord('3'):
                self.current_lane = 3
            elif key == ord('c'):
                self.lanes[self.current_lane] = []
                self.frame = cap.read()[1]
            elif key == ord('s'):
                # Save lane definitions
                lane_data = {}
                for lane_id, points in self.lanes.items():
                    if len(points) >= 3:
                        lane_data[lane_id] = np.array(points, np.int32).tolist()
                
                with open('lane_definitions.json', 'w') as f:
                    json.dump(lane_data, f, indent=2)
                
                print("Lane definitions saved to lane_definitions.json")
                break
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.lanes

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python lane_calibrator.py <video_path>")
        sys.exit(1)
    
    calibrator = LaneCalibrator(sys.argv[1])
    calibrator.calibrate()
