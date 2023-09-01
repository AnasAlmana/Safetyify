from flask import Flask, render_template, request
import os
from Safty import model_infer
from Safty import convert

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    message_received = False  # Initialize a variable to track message reception
    video_path = None  # Initialize the video_path variable

    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index2.html', message='No file part', message_received=message_received, video_path=video_path)
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return render_template('index2.html', message='No selected file', message_received=message_received, video_path=video_path)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        model_infer(video_path, video_file.filename)
        processed_file = f'static/output/{video_file.filename}'
        convert(processed_file)
        output_file = 'out.mp4'
        
        message_received = True  # Set the variable to indicate that a message has been received

        return render_template('index2.html', message='The video has been successfully analyzed!', message_received=message_received, video_path=output_file)
    
    return render_template('index2.html', message=None, message_received=message_received, video_path=video_path)

if __name__ == '__main__':
    app.run(debug=True)
