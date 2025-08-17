"""
Demo script to run traffic analysis with different configurations
"""
import os
import sys
from pathlib import Path
from traffic_analyzer import TrafficFlowAnalyzer

def run_demo():
    """Run a demo of the traffic flow analysis system"""
    
    print("Traffic Flow Analysis Demo")
    print("=" * 40)
    
    # Create analyzer instance
    analyzer = TrafficFlowAnalyzer(video_path=None, output_dir="demo_output")
    
    # Option 1: Use the YouTube video from the theory guide
    youtube_url = "https://youtu.be/MNn9qKG2UFI?si=1GL-gnlbj3eBpF1I"
    
    print(f"Processing YouTube video: {youtube_url}")
    print("This will:")
    print("1. Download the video")
    print("2. Detect vehicles using YOLOv8")
    print("3. Track vehicles across frames")
    print("4. Count vehicles per lane")
    print("5. Generate overlay video")
    print("6. Export CSV with detailed data")
    print()
    
    try:
        analyzer.process_video(youtube_url)
        
        print("\nDemo completed successfully!")
        print("Check the 'demo_output' folder for:")
        print("- processed_video.mp4 (with overlays)")
        print("- vehicle_counts.csv (detailed data)")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure you have all required packages installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    run_demo()
