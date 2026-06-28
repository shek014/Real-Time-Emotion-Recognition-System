from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import cv2, threading, time

app = Flask(__name__)
model = YOLO("best.pt")  # your model

# Shared state
capture_thread = None
capture_event = threading.Event()   # set() = camera running
frame_lock = threading.Lock()
latest_frame = None                 # bytes of most recent JPEG

def camera_worker():
    """Background thread that reads frames, runs YOLO and stores latest JPEG bytes."""
    global latest_frame
    cap = cv2.VideoCapture(0)
    # small delay for camera warm-up
    time.sleep(0.5)

    try:
        while capture_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # camera read failed; wait a bit and retry
                time.sleep(0.1)
                continue

            # Run YOLO inference (non-blocking options may exist depending on model)
            results = model.predict(frame, verbose=False, conf=0.25)
            annotated = results[0].plot()

            # Encode to JPEG
            ret2, buf = cv2.imencode('.jpg', annotated)
            if not ret2:
                continue

            with frame_lock:
                latest_frame = buf.tobytes()

            # throttle to avoid hogging CPU/GPU (adjust as needed)
            time.sleep(0.02)
    finally:
        cap.release()
        with frame_lock:
            latest_frame = None

def start_camera():
    global capture_thread
    if capture_thread is None or not capture_thread.is_alive():
        capture_event.set()
        capture_thread = threading.Thread(target=camera_worker, daemon=True)
        capture_thread.start()

def stop_camera():
    capture_event.clear()
    # give worker a moment to exit and release camera
    if capture_thread is not None:
        capture_thread.join(timeout=2)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Generator to stream MJPEG from the latest_frame bytes."""
    global latest_frame
    try:
        # run until client disconnects
        while True:
            # if camera not running, pause and wait
            if not capture_event.is_set():
                time.sleep(0.1)
                continue

            with frame_lock:
                data = latest_frame

            if data:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
                except GeneratorExit:
                    break
                except Exception:
                    break
            else:
                time.sleep(0.01)
    finally:
        # When client disconnects nothing more to do here
        pass

@app.route('/video_feed')
def video_feed():
    # streaming endpoint (browser <img> will request this)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle-camera', methods=['POST'])
def toggle_camera():
    # toggle camera on/off and return status
    if not capture_event.is_set():
        start_camera()
        return jsonify({'camera_on': True})
    else:
        stop_camera()
        return jsonify({'camera_on': False})

if __name__ == '__main__':
    # disable the reloader to avoid double background threads during development
    app.run(debug=True, use_reloader=False)
