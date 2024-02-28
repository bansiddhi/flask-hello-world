from flask import Flask, request, jsonify
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import dlib
import numpy as np
import cv2

app = Flask(__name__)

# Load the pre-trained face landmark model from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the base64-encoded image from the request
        data = request.get_json()
        encoded_image = data.get('encodedImage')

        if not encoded_image:
            raise ValueError("No 'encodedImage' field in the request JSON.")

        # Decode base64 and process the image
        image_blob = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_blob))

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

        # Check if a face is detected
        if len(dets) > 0:
            # Get the facial landmarks
            shape = predictor(gray, dets[0])

            # Sample operation: Draw a rectangle around the detected face
            draw = ImageDraw.Draw(image)
            for i in range(68):  # Assuming you have 68 landmarks
                x, y = shape.part(i).x, shape.part(i).y
                draw.rectangle([(x-2, y-2), (x+2, y+2)], outline="red")

            # Convert the modified image back to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            modified_image_blob = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({'modifiedImage': modified_image_blob})

        else:
            raise ValueError("No faces detected in the image.")

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)
