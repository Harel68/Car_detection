"""
Raspberry Pi Client - Smart Parking Gate Camera
"""

import os
import time
import requests
from datetime import datetime
from pathlib import Path


try:
    from plate_processor import check
    PLATE_PROCESSOR_AVAILABLE = True
except ImportError:
    print(" Warning: plate_processor not found")
    PLATE_PROCESSOR_AVAILABLE = False
    
    def check(image_num):
        return f"DEMO-{image_num:03d}"


try:
    from picamera2 import Picamera2
    CAMERA_VERSION = "picamera2"
    print(" Using picamera2 (recommended for Pi 4/5)")
except ImportError:
    try:
        import picamera
        CAMERA_VERSION = "picamera"
        print(" Using picamera (legacy)")
    except ImportError:
        print(" Warning: No camera library found. Using file copy mode for testing.")
        CAMERA_VERSION = "none"


SERVER_URL = "http://10.0.0.12:5000/report"
CAPTURE_FOLDER = "uploads"  # plate_processor expects images in uploads/
CAPTURE_INTERVAL = 5  
MAX_RETRIES = 3  

os.makedirs(CAPTURE_FOLDER, exist_ok=True)


class ParkingCamera:
    
    
    def __init__(self):
        self.image_counter = 1
        self.camera = None
        self._init_camera()
    
    def _init_camera(self):
        
        if CAMERA_VERSION == "picamera2":
            self.camera = Picamera2()
            config = self.camera.create_still_configuration()
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  
            print(" Camera initialized (picamera2)")
        
        elif CAMERA_VERSION == "picamera":
            self.camera = picamera.PiCamera()
            self.camera.resolution = (1024, 768)
            time.sleep(2)
            print(" Camera initialized (picamera)")
        
        else:
            print(" Running in TEST MODE ")
    
    def capture_image(self):
        
        filename = f"{self.image_counter}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        
        try:
            if CAMERA_VERSION == "picamera2":
                self.camera.capture_file(filepath)
            
            elif CAMERA_VERSION == "picamera":
                self.camera.capture(filepath)
            
            else:
                
                Path(filepath).touch()
            
            print(f" Image captured: {filename}")
            return self.image_counter, filepath
        
        except Exception as e:
            print(f" Capture failed: {e}")
            return None, None
    
    def cleanup(self):
        
        if CAMERA_VERSION == "picamera2" and self.camera:
            self.camera.stop()
        elif CAMERA_VERSION == "picamera" and self.camera:
            self.camera.close()
        print(" Camera closed")


def process_plate(image_number):
    try:
        
        plate_number = check(image_number)
        
        if plate_number and isinstance(plate_number, str):
            
            return plate_number, 95.0
        else:
            print(f" No plate detected")
            return None, 0
    
    except Exception as e:
        print(f" Plate processing error: {e}")
        return None, 0


def send_to_server(plate_number, image_path, confidence):
   
    if not plate_number:
        print(" No plate detected - skipping report")
        return False
    
    try:
        
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'plate_number': plate_number,
                'confidence': confidence
            }
            
            print(f"\n Sending report to server...")
            print(f"    Plate: {plate_number}")
            print(f"   Confidence: {confidence:.2f}%")
            
            
            response = requests.post(
                SERVER_URL,
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result.get('action', 'unknown')
                
                print(f"  Server response: {action}")
                
                if action == "open_gate":
                    print("  GATE OPENED - Authorized vehicle")
                elif action == "deny_access":
                    print("    ACCESS DENIED - Unauthorized vehicle")
                
                return True
            else:
                print(f"    Server error: {response.status_code}")
                return False
    
    except requests.exceptions.Timeout:
        print(" Server timeout - check connection")
        return False
    
    except requests.exceptions.ConnectionError:
        print("  Connection failed - is server running?")
        return False
    
    except Exception as e:
        print(f"   Send error: {e}")
        return False


def main():
    
    print("\n" + "="*50)
    print("SMART PARKING GATE - CLIENT STARTED")
    print("="*50)
    print(f"Server: {SERVER_URL}")
    print(f"Storage: {CAPTURE_FOLDER}/ (shared with plate_processor)")
    print(f"Interval: {CAPTURE_INTERVAL} seconds")
    print("="*50 + "\n")
    
    camera = ParkingCamera()
    
    try:
        while True:
            print(f"\n{'='*50}")
            print(f"Cycle #{camera.image_counter}")
            print(f"{'='*50}")
            
            
            image_num, image_path = camera.capture_image()
            
            if image_num is None:
                print("Skipping this cycle due to capture failure")
                time.sleep(CAPTURE_INTERVAL)
                continue
        
    
            print("Processing plate...")
            plate_number, confidence = process_plate(image_num)
            
            if plate_number:
                print(f"Detected: {plate_number} ({confidence:.1f}%)")
                
              
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    if attempt > 1:
                        print(f"Retry attempt {attempt}/{MAX_RETRIES}")
                    
                    success = send_to_server(plate_number, image_path, confidence)
                    
                    if success:
                        break
                    elif attempt < MAX_RETRIES:
                        print(f"   Waiting 2 seconds")
                        time.sleep(2)
                
                if not success:
                    print("Failed to send after all retries")
            else:
                print("No plate detected in image")
            
            
            camera.image_counter += 1
            
           
            print(f"\n Waiting {CAPTURE_INTERVAL} seconds until next capture...")
            time.sleep(CAPTURE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n Stopping client ")
    
    except Exception as e:
        print(f"\n\n Unexpected error: {e}")
    
    finally:
        camera.cleanup()
        print(" Client shut down \n")


if __name__ == "__main__":
    main()