import sqlite3
import random
from datetime import datetime, timedelta
import os

DB_NAME = "parking_logs.db"

def generate_fake_data():
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    
    c.execute("DELETE FROM entries")
    
    
    authorized_plates = ["12-345-67", "98-765-43", "44-555-66", "11-222-33"]
    guest_plates = ["55-123-45", "77-888-99", "33-000-11", "66-777-88", "12-345-67"] 
    
    print("Generating fake parking data...")
    
    
    for i in range(50):
        
        if random.random() > 0.3: 
            plate_number = random.choice(authorized_plates)
            is_auth = True
            confidence = random.uniform(95.0, 99.9)
        else:
            
            plate_number = random.choice(guest_plates)
            
            is_auth = plate_number in authorized_plates
            confidence = random.uniform(85.0, 98.0)
            
       
        hour_bias = random.choices([8, 9, 10, 17, 18, 19, 12, 14, 2, 3], 
                                 weights=[10, 10, 8, 10, 10, 8, 5, 5, 1, 1], k=1)[0]
        minute = random.randint(0, 59)
        
    
        days_back = 0 if random.random() > 0.2 else 1 
        fake_time = datetime.now() - timedelta(days=days_back)
        fake_time = fake_time.replace(hour=hour_bias, minute=minute, second=random.randint(0, 59))
        timestamp_str = fake_time.strftime("%Y-%m-%d %H:%M:%S")
        
        
        image_path = "test_car.jpg" 
        
        
        c.execute("INSERT INTO entries (plate_number, timestamp, image_path, is_authorized, confidence) VALUES (?, ?, ?, ?, ?)",
                  (plate_number, timestamp_str, image_path, is_auth, confidence))

    conn.commit()
    conn.close()
    print(" Successfully added 50 fake entries to the database!")

if __name__ == "__main__":
    from database import init_db
    init_db()
    generate_fake_data()