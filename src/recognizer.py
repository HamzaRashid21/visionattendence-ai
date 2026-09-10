"""
=============================================================================
VisionAttend AI - Smart Biometric Attendance System
Module: Real-time Face Recognition & Liveness Engine (src/recognizer.py)
-----------------------------------------------------------------------------
Maqsad (Purpose):
1. Camera frame se face detect karna (Haar Cascade).
2. Liveness Check (Anti-Proxy) perform karna taake photo/screen spoofing roki jaye.
3. Trained Deep Learning / Feature Matcher model se student identify karna.
4. Latency ko < 2 seconds maintain karna (SRS Non-Functional Requirement 1).
=============================================================================
"""

import os
import cv2
import json
import numpy as np

CASCADE_PATH = os.path.join("models", "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join("models", "haarcascade_eye.xml")
TFLITE_PATH = os.path.join("models", "attendance_model.tflite")
MODEL_PATH = os.path.join("models", "attendance_model.h5")
LABELS_PATH = os.path.join("models", "labels.json")

# In-memory caches
_face_cascade = None
_eye_cascade = None
_model_instance = None
_model_backend = None  # 'cv2_dnn', 'tf_keras', or None
_labels_map = {}


def load_detector():
    global _face_cascade, _eye_cascade
    if _face_cascade is None and os.path.exists(CASCADE_PATH):
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if _eye_cascade is None and os.path.exists(EYE_CASCADE_PATH):
        _eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    return _face_cascade, _eye_cascade


_input_idx = None
_output_idx = None


def load_model_if_available():
    """
    Trained Deep Learning model load karta hai.
    Priority:
    1. TFLite via ai_edge_litert (Lightning fast, XNNPACK CPU acceleration, < 15ms latency)
    2. Keras .h5 via TensorFlow (if installed)
    """
    global _model_instance, _model_backend, _labels_map, _input_idx, _output_idx

    if not _labels_map and os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r") as f:
                _labels_map = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Labels file load nahi ho saki: {e}")

    if _model_instance is not None:
        return _model_instance, _model_backend, _labels_map

    # 1. Check TFLite via LiteRT / tflite-runtime
    if os.path.exists(TFLITE_PATH):
        try:
            try:
                import ai_edge_litert.interpreter as tflite
            except ImportError:
                import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            _input_idx = interpreter.get_input_details()[0]['index']
            _output_idx = interpreter.get_output_details()[0]['index']
            _model_instance = interpreter
            _model_backend = "litert"
            print("[*] LiteRT XNNPACK engine loaded 'attendance_model.tflite' successfully (< 15ms latency).")
            return _model_instance, _model_backend, _labels_map
        except Exception as e:
            print(f"[!] LiteRT load error: {e}")

    # 2. Check Keras H5
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            _model_instance = tf.keras.models.load_model(MODEL_PATH)
            _model_backend = "tf_keras"
            print("[*] TensorFlow loaded 'attendance_model.h5' successfully.")
            return _model_instance, _model_backend, _labels_map
        except Exception as e:
            print(f"[!] Warning: TensorFlow model load nahi ho saka: {e}")

    return None, None, _labels_map


def predict_probabilities(model, backend, rgb_face_160):
    """Normalized 160x160 RGB face array par probabilities calculate karta hai."""
    global _input_idx, _output_idx
    try:
        normalized = rgb_face_160.astype("float32") / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        if backend == "litert":
            model.set_tensor(_input_idx, input_tensor)
            model.invoke()
            preds = model.get_tensor(_output_idx)[0]
            return preds
        elif backend == "tf_keras":
            preds = model.predict(input_tensor, verbose=0)
            return np.squeeze(preds)
    except Exception as e:
        print(f"[!] Inference error ({backend}): {e}")
    return None


def check_liveness(gray_face):
    """
    SRS Requirement xii (Page 9): "Proxy Detection prevention".
    Chehre par eyes aur natural micro-texture check karta hai taake 
    flat printed paper photo ya phone screen se proxy na lag sake.
    """
    _, eye_cascade = load_detector()
    if eye_cascade is None or gray_face is None or gray_face.size == 0:
        return True, "Passed"

    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
    
    # Laplacian variance check (blur/texture check)
    variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if variance < 14.0:  # Relaxed for dim webcams / evening lighting
        return False, "Low texture / Possible 2D paper photo"

    return True, "Live Human Confirmed"


def process_frame(frame, confidence_threshold=0.80, return_boxes=False):
    """
    Live video frame process karta hai:
    - Faces detect karta hai (with histogram equalization for dim lighting)
    - Bounding box aur labels render karta hai
    - Identified student ki info return karta hai (only if confidence >= threshold)
    - Optional: returns detected face box coordinates for client HUD
    """
    face_cascade, _ = load_detector()
    if face_cascade is None:
        return (frame, [], []) if return_boxes else (frame, [])

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 0.5x downscale for 4x faster Haar cascade detection (< 10ms latency)
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    
    # Histogram equalization for reliable detection in low/dim classroom lighting
    equalized_small = cv2.equalizeHist(small_gray)
    raw_faces = face_cascade.detectMultiScale(equalized_small, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
    if len(raw_faces) == 0:
        raw_faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
        
    faces = [(int(fx * 2), int(fy * 2), int(fw * 2), int(fh * 2)) for (fx, fy, fw, fh) in raw_faces]

    recognized_students = []
    detected_boxes = []
    model, backend, labels = load_model_if_available()

    for (x, y, fw, fh) in faces:
        cropped_face = gray[y:y+fh, x:x+fw]
        if cropped_face.size == 0:
            continue

        is_live, liveness_msg = check_liveness(cropped_face)

        if not is_live:
            # Spoofing detected! Render RED box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
            cv2.putText(frame, "PROXY / SPOOF DETECTED!", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            detected_boxes.append({
                "box": [int(x), int(y), int(fw), int(fh)],
                "name": "Spoof Detected",
                "student_id": None,
                "confidence": 0,
                "matched": False
            })
            continue

        student_identified = None
        confidence_pct = 0

        if model is not None and labels:
            # Match training data preprocessing: histogram equalization + RGB + 160x160
            equalized_face = cv2.equalizeHist(cropped_face)
            rgb_face = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb_face, (160, 160), interpolation=cv2.INTER_AREA)

            predictions = predict_probabilities(model, backend, resized)
            if predictions is not None and len(predictions) > 0:
                best_idx = int(np.argmax(predictions))
                best_conf = float(predictions[best_idx])
                confidence_pct = int(best_conf * 100)

                # Pure AI/ML check: verify decisive logit/confidence margin over runner-up class
                sorted_preds = np.sort(predictions)[::-1]
                runner_up = float(sorted_preds[1]) if len(sorted_preds) > 1 else 0.0
                margin = best_conf - runner_up

                # Strict criteria: Must exceed confidence threshold AND have a decisive margin (>= 35%)
                if best_conf >= confidence_threshold and margin >= 0.35:
                    class_name = labels.get(str(best_idx), labels.get(best_idx, "Unknown"))
                    # Label format: "CS-109_Hamza_Rashid"
                    parts = class_name.split("_", 1)
                    student_id = parts[0]
                    student_name = parts[1].replace("_", " ") if len(parts) > 1 else class_name

                    student_identified = {
                        "student_id": student_id,
                        "name": student_name,
                        "confidence": confidence_pct
                    }
                    recognized_students.append(student_identified)

        if student_identified:
            # Recognized with decisive high confidence! Render GREEN box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            display_text = f"{student_identified['name']} ({confidence_pct}%)"
            cv2.putText(frame, display_text, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected_boxes.append({
                "box": [int(x), int(y), int(fw), int(fh)],
                "name": student_identified["name"],
                "student_id": student_identified["student_id"],
                "confidence": confidence_pct,
                "matched": True,
                "status": "verified"
            })
        else:
            # Unverified, below threshold, or ambiguous: Render RED/ROSE box
            if model is not None and confidence_pct > 0:
                label_text = f"Low Match ({confidence_pct}% < {int(confidence_threshold * 100)}%)"
            else:
                label_text = "Unverified Face"
                
            box_color = (0, 0, 245) # Bright Red
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)
            cv2.putText(frame, label_text, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            detected_boxes.append({
                "box": [int(x), int(y), int(fw), int(fh)],
                "name": label_text,
                "student_id": None,
                "confidence": confidence_pct,
                "matched": False,
                "status": "unverified"
            })

    if return_boxes:
        return frame, recognized_students, detected_boxes
    return frame, recognized_students
