from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('best_model.h5')

# Emotion labels (adjust to your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)

        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected'})

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({
            'emotion': emotion,
            'box': [int(x), int(y), int(w), int(h)]  # return face bounding box
        })

    except Exception as e:
        return jsonify({'error': str(e)})

    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)

        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # OPTIONAL: Face detection (still uses original color image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected'})

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]  # crop color face
        face = cv2.resize(face, (224, 224))  # resize to model input shape
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))  # match expected model input

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]  # Remove base64 prefix
        img_bytes = base64.b64decode(img_data)

        # Convert to numpy array
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess image (resize and grayscale as needed by your model)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':

    app.run(debug=True)