from flask import Flask, render_template, send_file
import os
import re
import logging
import random

app = Flask(__name__)

# Configuration
DETECTION_FOLDER = r'C:\Users\Joel Fredrick\Desktop\call-out-signal\detected_frames_20250326_014256'

@app.route('/')
def results():
    detection_folder = DETECTION_FOLDER
    
    # If the folder doesn't exist, return an error
    if not os.path.exists(detection_folder):
        return render_template('results.html', error=f'Detection folder not found: {detection_folder}')

    try:
        # Get all files in the folder
        all_files = os.listdir(detection_folder)
        
        # Log all files in the folder for debugging
        print("All files in detection folder:")
        for file in all_files:
            print(file)

        # Get all image files 
        image_files = [f for f in all_files 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        print(f"Number of image files found: {len(image_files)}")

        # Create results list for images
        detected_images = []
        for image_filename in image_files:
            detected_images.append({
                'filename': image_filename
            })
        
        # Generate random coordinates with signal status
        coordinates = []
        latitude_base = 15.26
        longitude_base = 73.98
        
        for i in range(50):  # Generate 50 coordinate entries
            # Slightly randomize latitude and longitude
            latitude = round(latitude_base + random.uniform(-0.5, 0.5), 6)
            longitude = round(longitude_base + random.uniform(-0.5, 0.5), 6)
            
            # Randomly decide signal status
            status = random.choice(['Done', 'Missed'])
            
            coordinates.append(f"Signal {status}: Latitude {latitude}, Longitude {longitude}")
        
        # Prepare context for the template
        context = {
            'total_signals': len(detected_images),
            'processing_time': round(random.uniform(30, 60), 2),
            'video_duration': round(random.uniform(5, 15), 2),
            'log_file': 'detection_log.txt',
            'updated_report': 'updated_report.xlsx',
            'signals_file': 'signals_status.csv',
            'detected_images': detected_images,
            'signal_coordinates': '\n'.join(coordinates)
        }
        
        return render_template('results.html', **context)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return render_template('results.html', error=f"Error reading results: {str(e)}")

@app.route('/images/<filename>')
def serve_detection_image(filename):
    detection_folder = DETECTION_FOLDER
    return send_file(os.path.join(detection_folder, filename))

if __name__ == '__main__':
    app.run(debug=True)