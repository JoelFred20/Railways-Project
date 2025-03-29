from flask import Flask, render_template, send_file
import os
import random

app = Flask(__name__)

# Configuration
DETECTION_FOLDER = r'C:\Users\Joel Fredrick\Desktop\call-out-signal\phone_detection_output\frames'

@app.route('/')
def results():
    detection_folder = DETECTION_FOLDER
    
    if not os.path.exists(detection_folder):
        return render_template('mobile_results.html', error=f'Detection folder not found: {detection_folder}')

    try:
        all_files = os.listdir(detection_folder)
        
        # Get all image files 
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        detected_images = [{'filename': image_filename} for image_filename in image_files]
        
        # Generate random coordinates with signal status
        coordinates = []
        latitude_base = 15.26
        longitude_base = 73.98
        
        for _ in range(50):  # Generate 50 coordinate entries
            latitude = round(latitude_base + random.uniform(-0.5, 0.5), 6)
            longitude = round(longitude_base + random.uniform(-0.5, 0.5), 6)
            status = random.choice(['Done', 'Missed'])
            coordinates.append(f"Signal {status}: Latitude {latitude}, Longitude {longitude}")
        
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
        
        return render_template('mobile_results.html', **context)
    
    except Exception as e:
        return render_template('mobile_results.html', error=f"Error reading results: {str(e)}")

@app.route('/images/<filename>')
def serve_detection_image(filename):
    return send_file(os.path.join(DETECTION_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)