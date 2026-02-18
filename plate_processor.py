import cv2
import numpy as np
import os
from segment_plate import running

class LicensePlateProcessor:
    def __init__(self):
        """Initialize the license plate processor"""
        print(" License Plate Processor initialized")
        
    def detect_yellow_plate(self, image_path):
        """
        Detect yellow license plate using color detection
        Returns: cropped plate image and debug image
        """
        print(f"\n1. Loading image: {image_path}")
        img = cv2.imread(image_path)
        
        if img is None:
            print(" Could not load image")
            return None, None
            
        print(f"   Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Convert to HSV color space
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite("hsv.jpg", hsv)

        # Define range for yellow color in HSV
       
        lower_yellow = np.array([10, 80, 80])   
        upper_yellow = np.array([40, 255, 255])  
        
        print("\n2. Detecting yellow regions...")
        # Create mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        cv2.imwrite("mask.jpg", mask)

        # Apply morphological operations to clean up the mask
        #kernel = np.ones((6,6), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        #cv2.imwrite("mask_after_morphology.jpg", mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        '''
        #showing the islands that we observe :

        viz_on_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
      
        cv2.drawContours(viz_on_mask, contours, -1, (0, 255, 0), 2)
        
        
        cv2.imwrite("all_contours.jpg", viz_on_mask)
        
        '''



        if not contours:
            print(" No yellow regions detected")
            return None, None
        
        print(f"Found {len(contours)} yellow regions")
        
        # Filter contours by size and aspect ratio
        plate_candidates = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h 
            
           
            if 2.5 < aspect_ratio < 5.0 and area > 1000:
                plate_candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
                print(f"   Candidate: size={w}x{h}, area={area}, ratio={aspect_ratio:.2f}")
        
        if not plate_candidates:
            print(" No valid license plate candidates found")
            print("   (Looking for wide yellow rectangles)")
            return None, None
        
        # Get the largest candidate 
        plate = max(plate_candidates, key=lambda x: x['area'])
        x, y, w, h = plate['bbox']
        
        print(f"\n License plate detected!")
        print(f"   Position: ({x}, {y})")
        print(f"   Size: {w}x{h}")
        print(f"   Aspect ratio: {plate['aspect_ratio']:.2f}")
        
        # Add some padding around the plate
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Crop the plate
        plate_img = img[y_start:y_end, x_start:x_end]

        target_size = (640, 160)  # width x height

        
        plate_img = cv2.resize(plate_img, target_size, interpolation=cv2.INTER_AREA)
        
        # Create debug image with all candidates and the selected one
        debug_img = img.copy()
        
        # Draw all candidates in blue
        for candidate in plate_candidates:
            x, y, w, h = candidate['bbox']
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw selected plate in green
        x, y, w, h = plate['bbox']
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(debug_img, "LICENSE PLATE", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        
        return plate_img, debug_img
    
    def process(self, image_path, save_debug=True):
        """
        Complete processing pipeline
        """
        print("=" * 60)
        print(f"Processing: {image_path}")
        print("=" * 60)
        
        # Detect yellow plate
        result = self.detect_yellow_plate(image_path)
        
        if result[0] is None:
            print("\n Failed: Could not detect license plate")
            
            return None
        
        plate_img, debug_img = result
        
        # Save debug images
        if save_debug:
            '''
            #Save detected plate with rectangle
            debug_path = image_path.replace('.jpg', '_detected.jpg')
            cv2.imwrite(debug_path, debug_img)
            print(f"\n Saved detection visualization: {debug_path}")
            '''
            # Save the cropped plate
            plate_path = image_path.replace('.jpg', '_plate.jpg')
            cv2.imwrite(plate_path, plate_img)
            print(f" Saved cropped plate: {plate_path}")
            
            
            
        
        print("\n" + "=" * 60)
        print(f" SUCCESS! Plate extracted")
        print("=" * 60)
        
        return {
            'plate_image': plate_img,
            'detected': True
        }


def oneImageplate(image_path):
    
    processor = LicensePlateProcessor()
   
    
    image_name = os.path.basename(image_path)
    print(f"[Processing: {image_name}")
    try:
        result = processor.process(image_path, save_debug=True)
    except Exception as e:
        print(f"    Error processing {image_name}: {e}")
        
    if result:
        print(f"    Plate successfully extracted for {image_name}")
    else:
        print(f"    Could not detect plate for {image_name}")
    print("-" * 60)
    
        
def check(number):
    oneImageplate(f'uploads/{number}.jpg')
    license_number1 = running(number)
    return license_number1
    


def main():
    license_number = check(20)
    print(f"The Number is: {license_number}")












































































































































if __name__ == "__main__":
    main()