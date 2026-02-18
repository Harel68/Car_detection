from flask import Flask, request, jsonify
import os
from datetime import datetime
from database import log_entry 

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUTHORIZED_PLATES = [
    "302-34-402",
    "98-765-43",
    
]

@app.route('/report', methods=['POST'])
def report_from_camera():
  
    try:
        
        plate_number = request.form.get('plate_number')
        confidence = float(request.form.get('confidence', 0))
        
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image evidence provided'}), 400
            
        file = request.files['image']
        
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{plate_number}_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"\n Received Report from Pi:")
        print(f"    Plate: {plate_number}")
        print(f"    Confidence: {confidence:.2f}%")

        
        is_authorized = plate_number in AUTHORIZED_PLATES
        
        if is_authorized:
            print("    STATUS: AUTHORIZED")
        else:
            print("    STATUS: GUEST / UNAUTHORIZED")

        
        log_entry(plate_number, filepath, is_authorized, confidence)

        
        return jsonify({
            "status": "success",
            "action": "open_gate" if is_authorized else "deny_access",
            "message": "Report logged successfully"
        }), 200

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)