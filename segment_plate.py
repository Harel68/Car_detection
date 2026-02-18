import cv2
import numpy as np
import sys
import os
import torch
from dot_detector import DotDetectorCNN 
import matplotlib.pyplot as plt
from predict_digit import predict_license_plate

class DotDetector:
    """
    Class for detecting separator dots and plate limits
    Model Output: Channel 0 (Dots), Channel 1 (Limits)
    """
    def __init__(self, model_path='best_dot_detector.pth'):
        self.img_width = 640
        self.img_height = 160
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_path}...")
        try:
            # Initialize the updated 2-channel model
            self.model = DotDetectorCNN().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f" Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        if (h, w) != (self.img_height, self.img_width):
            resized = cv2.resize(image, (self.img_width, self.img_height))
        else:
            resized = image
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor, (h, w)
    
    def postprocess_heatmap(self, heatmap, orig_size, threshold=0.4):
        """
        Convert heatmap to coordinates with specific filtering:
        1. Dots: Highest 2 confidence only.
        2. Limits: Leftmost (min x) and Rightmost (max x) only.
        """
        h, w = orig_size
        heatmap = heatmap.squeeze(0).cpu().numpy()
        
        # --- PROCESS DOTS (Channel 0) ---
        dots_heatmap = heatmap[0]
        dots = self._extract_keypoints(dots_heatmap, w, h, threshold)
        
        # FILTER 1: Sort by confidence (highest first)
        dots.sort(key=lambda d: d['confidence'], reverse=True)
        
        # FILTER 2: Keep only top 2
        dots = dots[:2]
        
        # FILTER 3: Sort by X for consistent left-to-right ordering
        dots.sort(key=lambda d: d['x'])
        
        # --- PROCESS LIMITS (Channel 1) ---
        limits_heatmap = heatmap[1]
        limits = self._extract_keypoints(limits_heatmap, w, h, threshold)
        
        # FILTER: Keep only Leftmost and Rightmost
        final_limits = []
        if len(limits) >= 2:
            # Sort by X coordinate
            limits.sort(key=lambda d: d['x'])
            
            # Take the lowest X (Left limit) and highest X (Right limit)
            left_limit = limits[0]
            right_limit = limits[-1]
            
            final_limits = [left_limit, right_limit]
        elif len(limits) == 1:
            final_limits = limits
            
        return dots, final_limits
    
    def _extract_keypoints(self, heatmap, orig_w, orig_h, threshold):
        binary = (heatmap > threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        keypoints = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                cx_orig = int(cx * orig_w / self.img_width)
                cy_orig = int(cy * orig_h / self.img_height)
                
                confidence = heatmap[cy, cx]
                
                keypoints.append({
                    'x': cx_orig,
                    'y': cy_orig,
                    'confidence': float(confidence)
                })
        return keypoints
    
    def detect(self, plate_image, threshold=0.5, visualize=False):
        input_tensor, orig_size = self.preprocess_image(plate_image)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            heatmap = self.model(input_tensor)
        
        dots, limits = self.postprocess_heatmap(heatmap, orig_size, threshold)
        
        if visualize:
            viz_image = self.visualize_results(plate_image, dots, limits, heatmap)
            return dots, limits, viz_image
        
        return dots, limits

    def visualize_results(self, plate_image, dots, limits, heatmap):
        viz = plate_image.copy()
        
        # Draw Dots (Green)
        for i, dot in enumerate(dots):
            cv2.circle(viz, (dot['x'], dot['y']), 6, (0, 255, 0), -1)
            cv2.putText(viz, f"D{i}", (dot['x'], dot['y']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw Limits (Blue Lines)
        for i, limit in enumerate(limits):
            x = limit['x']
            cv2.line(viz, (x, 0), (x, viz.shape[0]), (255, 0, 0), 3)
            label = "L_Limit" if i == 0 else "R_Limit"
            cv2.putText(viz, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return viz


class SmartSegmenter:
    def __init__(self):
        pass

    def preprocess_segment(self, img):
        """
        1. Remove top/bottom background (Yellow filtering)
        2. Convert to Binary (Inverted: White digits, Black background)
        """
        h, w = img.shape[:2]
        
        # --- 1. Y-Axis Crop (Yellow Filter) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 60, 60])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find yellow rows to crop top/bottom noise
        row_sums = np.sum(mask, axis=1)
        yellow_rows = np.where(row_sums > (w * 0.2))[0]
        
        if len(yellow_rows) > 0:
            y_start = max(0, yellow_rows[0] - 2)
            y_end = min(h, yellow_rows[-1] + 2)
            img = img[y_start:y_end, :]
        
        # --- 2. Binary Conversion ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu's thresholding automatically finds the best split
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean small noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)
        h, w = binary.shape
        
        # חישוב אחוז הלבן בכל שורה:
        row_sums = np.sum(binary, axis=1)
        white_percentage = row_sums / (w * 255.0)
        
        
        # שורה תקינה = שורה שיש בה *פחות* מ-80% לבן (<= 0.8)
        valid_rows = np.where(white_percentage <= 0.8)[0]
        
        if len(valid_rows) > 0:
            y_start = valid_rows[0]
            y_end = valid_rows[-1] + 1
            binary = binary[y_start:y_end, :]

        return binary
    def keep_largest_island(self, binary_digit):
        """
        Keeps only the largest white object in the image.
        Removes all other small noise/dots.
        """
        # 1. מציאת כל האיים הלבנים (Connected Components)
        # num_labels: כמה איים נמצאו סך הכל
        # labels: תמונה שבה כל פיקסל מקבל את המספר של האי שלו
        # stats: טבלה עם נתונים על כל אי (שטח, מיקום, גובה, רוחב)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_digit, connectivity=8)
        
        # אם לא נמצאו איים (רק רקע שחור), מחזירים את התמונה כמו שהיא
        if num_labels <= 1:
            return binary_digit
            
        # 2. חיפוש האי עם השטח הכי גדול (מדלגים על אינדקס 0 כי הוא הרקע השחור!)
        largest_label = 1
        max_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                largest_label = i
                
        # 3. יצירת תמונה נקייה
        clean_digit = np.zeros_like(binary_digit)
        
        # מעתיקים לתמונה החדשה *רק* את הפיקסלים ששייכים לאי הכי גדול
        clean_digit[labels == largest_label] = 255
        
        return clean_digit
    def find_cuts_by_mass(self, binary_img, n_digits,segment_idx=None):
        """
        Finds cut columns based on equal distribution of white pixels.
        Includes 'Gap Correction' to snap to the nearest black column.
        """
        h, w = binary_img.shape
        col_sums = np.sum(binary_img == 255, axis=0) # Count white pixels per column
        total_white = np.sum(col_sums)
        
        cuts = []
        ideals = []

        if n_digits == 2:
            # Target: 50% of white pixels (Halfway point)
            target_mass = total_white / 2
            
            # Find index where cumulative sum crosses target
            cum_sum = np.cumsum(col_sums)
            ideal_cut = np.searchsorted(cum_sum, target_mass)
            ideals.append(ideal_cut)

            # Correction: Find the "blackest" column near the ideal cut
            corrected_cut = self._snap_to_gap(col_sums, ideal_cut, search_window=20)
            cuts.append(corrected_cut)
            
        elif n_digits == 3:
            # Targets: 33% and 66% (Thirds)
            target_1 = total_white / 3
            target_2 = (total_white / 3) * 2
            
            cum_sum = np.cumsum(col_sums)
            ideal_1 = np.searchsorted(cum_sum, target_1)
            ideal_2 = np.searchsorted(cum_sum, target_2)
            ideals.append(ideal_1)
            ideals.append(ideal_2)

            cut1 = self._snap_to_gap(col_sums, ideal_1, search_window=15)
            cut2 = self._snap_to_gap(col_sums, ideal_2, search_window=15)
            cuts.append(cut1)
            cuts.append(cut2)

        if segment_idx is not None:
            self._plot_debug_graph(col_sums, cuts, ideals, n_digits, segment_idx)
        return cuts
    
    def _plot_debug_graph(self, col_sums, final_cuts, ideal_cuts, n_digits, segment_idx):
        """
        Helper function to plot the graph using PRE-CALCULATED data.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot Density
        plt.plot(col_sums, color='black', linewidth=2, label='Pixel Density')
        plt.fill_between(range(len(col_sums)), col_sums, color='gray', alpha=0.3)
        
        # Plot Ideals (Green)
        for ideal in ideal_cuts:
            plt.axvline(x=ideal, color='green', linestyle='--', label='Ideal Math Cut')
            
        # Plot Finals (Red)
        for cut in final_cuts:
            plt.axvline(x=cut, color='red', linewidth=2.5, label='Final Cut')
            
        plt.title(f"Segment {segment_idx} Analysis (Digits: {n_digits})")
        plt.legend()
        plt.savefig(f"debug_graph_segment_{segment_idx}.png")
        plt.close()
        print(f"   [Graph] Saved efficient graph to debug_graph_segment_{segment_idx}.png")
    
    def _snap_to_gap(self, col_sums, center_idx, search_window):
        """Finds the column with minimum white pixels within a window"""
        start = max(0, center_idx - search_window)
        end = min(len(col_sums), center_idx + search_window)
        
        # Get the slice of column sums
        window_sums = col_sums[start:end]
        
        if len(window_sums) == 0:
            return center_idx
            
        # Find index of minimum value in this window
        min_local_idx = np.argmin(window_sums)
        return start + min_local_idx

    def process_plate(self, image_path, detector):
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading {image_path}")
            return

        print(f"Processing: {image_path}")
        
        # 2. Detect Dots & Limits
        dots, limits, viz = detector.detect(img, visualize=True)
        cv2.imwrite("step1_detection.jpg", viz)
        print(" Dots and limits detected!!")

        if len(dots) != 2 or len(limits) != 2:
            print(" Error: Need exactly 2 dots and 2 limits to segment.")
            print(f"   Found {len(dots)} dots and {len(limits)} limits.")
            return

        # 3. Define Segments Coordinates
        # Structure: Left Limit -> Dot 0 -> Dot 1 -> Right Limit
        limits.sort(key=lambda x: x['x']) # [Left, Right]
        dots.sort(key=lambda x: x['x'])   # [D0, D1]
        
        boundaries = [limits[0]['x'], dots[0]['x'], dots[1]['x'], limits[1]['x']]
        
        # 4. Extract and Process Each Segment
        final_digits = []
        output_dir = "final_digits"
        os.makedirs(output_dir, exist_ok=True)
       
        # --- CONFIGURATION: HARDCODED STRUCTURE ---
        dot_distance = dots[1]['x'] - dots[0]['x']
        if dot_distance < 200:
            digit_counts = [3, 2, 3]
        else:
            digit_counts = [2, 3, 2]
        total_digits = sum(digit_counts)
        image_paths = [f"{output_dir}/digit_{i}.jpg" for i in range(1, total_digits+1)]
        print(f"Using fixed plate structure: {digit_counts}")
        
        # --- PROCESS EACH SEGMENT ---
        for i in range(3):
            print(f"\n--- Segment {i+1} ---")
            x1, x2 = boundaries[i], boundaries[i+1]
            
            # Check for sanity
            if x2 <= x1:
                print(f" Warning: Segment {i} has invalid width (x1={x1}, x2={x2}). Skipping.")
                continue

            # Crop Segment (add small margin)
            margin = -3
            seg_img = img[:, max(0, x1-margin):min(img.shape[1], x2+margin)]
            
            # Preprocess 
            binary = self.preprocess_segment(seg_img)

            # Find Cuts based on hardcoded digit count
            n_digits = digit_counts[i]
            cuts = self.find_cuts_by_mass(binary, n_digits, i)
            print(f"  Digits: {n_digits} | Cuts at columns: {cuts}")
            
            # Split Digit Images
            seg_boundaries = [0] + cuts + [binary.shape[1]]
            
            for j in range(len(seg_boundaries)-1):
                d_x1 = seg_boundaries[j]
                d_x2 = seg_boundaries[j+1]
                
                # Crop digit from BINARY image 
                digit = binary[:, d_x1:d_x2]
                digit = self.keep_largest_island(digit)
                # Resize to standard size for recognition (e.g., 75x100)
                digit_resized = cv2.resize(digit, (28, 28))
                
                final_digits.append(digit_resized)
                
                # Save
                # Format: digit_INDEX_group_GROUP.jpg
                # Global index helps keeps them in order
                global_idx = len(final_digits) 
                fname = f"{output_dir}/digit_{global_idx}.jpg"
                cv2.imwrite(fname, digit_resized)
                print(f"  Saved digit to {fname}")
                
            # Visualization of cuts
            viz_seg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for c in cuts:
                cv2.line(viz_seg, (c, 0), (c, binary.shape[0]), (0, 0, 255), 2)
            cv2.imwrite(f"step2_segment_{i}_cuts.jpg", viz_seg)

        print(f"\n Done! Saved {len(final_digits)} digits to '{output_dir}/'")
        license_number = predict_license_plate(image_paths, show_plot=True)
        return license_number

def running(photo_num):
    IMAGE_PATH = f'uploads/{photo_num}_plate.jpg'
    try:
        detector = DotDetector('best_dot_detector.pth')
    except:
        print("Could not load DotDetector")
        sys.exit(1)

    # Run Segmentation
    segmenter = SmartSegmenter()
    license_number = segmenter.process_plate(IMAGE_PATH, detector)

    return license_number 


if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = 'uploads/41_plate.jpg' 
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    # Initialize Detector
    try:
        detector = DotDetector('best_dot_detector.pth')
    except:
        print("Could not load DotDetector")
        sys.exit(1)

    # Run Segmentation
    segmenter = SmartSegmenter()
    segmenter.process_plate(IMAGE_PATH, detector)