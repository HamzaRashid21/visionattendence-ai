"""
=============================================================================
VisionAttend AI - Smart Biometric Attendance System
Module: Dataset Collector (collect_dataset.py)
-----------------------------------------------------------------------------
Maqsad (Purpose):
SRS ke Page 6 ki requirement ke mutabiq har student ke chehre (face) ki 100+ 
photos webcam se capture karke pre-process (crop, grayscale, 160x160 resize) 
karke dataset folder mein save karna.

Kyun Zaroori Hai? (Judges Defense):
Model tabhi kisi student ko pehchan sakta hai jab uske paas har student ke 
alif-se-ye tak 100 photos mukhtalif angles (sidha, thora sa right/left, smile) 
ke sath hon. Background ko crop karke hataya jata hai taake model deewar ya 
pankhay ke bajaye sirf insani chehre ke features seekhe.
=============================================================================
"""

import os
import cv2
import time
import zipfile
import argparse

# Path constants
CASCADE_PATH = os.path.join("models", "haarcascade_frontalface_default.xml")
DATASET_DIR = "dataset"
TARGET_IMAGE_SIZE = (160, 160)  # Standard face size for Deep Learning models
DEFAULT_TARGET_COUNT = 100      # SRS Page 6 requirement: "over 100 images per student"


def init_face_cascade():
    """Haar Cascade Face Detection classifier load karta hai."""
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(
            f"Face cascade model file '{CASCADE_PATH}' nahi mili! "
            f"Baraye meharbani ensure karein ke models/ folder mein yeh file mojood ho."
        )
    return cv2.CascadeClassifier(CASCADE_PATH)


def zip_dataset(zip_filename="dataset.zip"):
    """
    Pore dataset folder ko ek single zip file banata hai.
    Faida: Yeh zip file aap 1 click se Google Colab par upload kar sakte hain.
    """
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("[!] Dataset folder khali hai, zip karne ke liye kuch nahi mila.")
        return False

    print(f"\n[*] Dataset ko compress kiya ja raha hai: '{zip_filename}'...")
    student_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    total_images = 0

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in student_folders:
            folder_path = os.path.join(DATASET_DIR, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, os.path.dirname(DATASET_DIR))
                        zipf.write(full_path, rel_path)
                        total_images += 1

    print(f"[OK] Mubarak ho! '{zip_filename}' successfully tayyar hai!")
    print(f"     Total Enrolled Students: {len(student_folders)}")
    print(f"     Total Images Packed: {total_images}")
    print(f"     Ab aap is '{zip_filename}' ko directly Google Colab pe upload kar sakte hain.\n")
    return True


def capture_student_data(student_id: str, student_name: str, target_count: int = DEFAULT_TARGET_COUNT):
    """
    Webcam on karke specific student ki images capture aur process karta hai.
    """
    # 1. Clean folder name generate karna: e.g. "101_Ali"
    clean_name = "".join(c for c in student_name if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    folder_name = f"{student_id}_{clean_name}"
    student_dir = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(student_dir, exist_ok=True)

    face_cascade = init_face_cascade()

    # 2. Camera start karna (0 default webcam hota hai)
    print(f"\n[*] Camera start ho raha hai student: '{student_name}' (ID: {student_id})...")
    cap = cv2.VideoCapture(0)

    # Resolution set karna (640x480 for smooth 30 FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Camera open nahi ho saka! Check karein koi doosri app webcam use to nahi kar rahi.")
        return

    # Check already existing images
    existing_images = [f for f in os.listdir(student_dir) if f.lower().endswith('.jpg')]
    count = len(existing_images)
    print(f"[*] Folder mein pehle se {count} images mojood hain.")
    print(f"[*] Target: {target_count} images capture karni hain.")
    print("[*] Screen par 'q' dabane se capture stop ho jayega.\n")

    capture_delay = 0.08  # Har frame ke darmiyan thora gap taake different angles capture hon
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera se frame read nahi ho saka.")
            break

        # Horizontal flip (mirror effect for user convenience)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Grayscale for Haar Cascade detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (scaleFactor 1.2, minNeighbors 5)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        current_time = time.time()

        for (x, y, fw, fh) in faces:
            # Face box draw karna (Green rectangle)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            # Agar count abhi pura nahi hua aur time gap ho chuka hai to photo save karein
            if count < target_count and (current_time - last_capture_time >= capture_delay):
                # Padding add karein taake forehead ya chin cut na ho (10% padding)
                pad_w = int(fw * 0.1)
                pad_h = int(fh * 0.1)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + fw + pad_w)
                y2 = min(h, y + fh + pad_h)

                # Crop only the face
                cropped_face = gray_frame[y1:y2, x1:x2]

                if cropped_face.size > 0:
                    # Pre-processing: Standard 160x160 size & Histogram Equalization (Lighting fix)
                    resized_face = cv2.resize(cropped_face, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    equalized_face = cv2.equalizeHist(resized_face)

                    # Save image (e.g. 101_Ali_045.jpg)
                    count += 1
                    img_filename = f"{folder_name}_{count:03d}.jpg"
                    img_filepath = os.path.join(student_dir, img_filename)
                    cv2.imwrite(img_filepath, equalized_face)
                    last_capture_time = current_time

            # Label on bounding box
            label = f"{student_name} ({count}/{target_count})"
            cv2.putText(frame, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # On-screen HUD (Heads-Up Display) Header
        cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)
        progress_pct = int((count / target_count) * 100) if target_count > 0 else 100
        progress_pct = min(100, progress_pct)

        header_text = f"Student: {student_name} [ID: {student_id}] | Captured: {count}/{target_count} ({progress_pct}%)"
        cv2.putText(frame, header_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        hint_text = "Instructions: Turn head slightly left, right, up, smile. Press 'q' to stop."
        cv2.putText(frame, hint_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        # Progress bar draw karna
        bar_x, bar_y, bar_w, bar_h = 15, h - 25, w - 30, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int((progress_pct / 100.0) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)

        cv2.imshow("VisionAttend AI - Biometric Dataset Collector", frame)

        # Check for exit or completion
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[!] User ne capture process cancel kar diya.")
            break

        if count >= target_count:
            print(f"\n[SUCCESS] Mubarak! {student_name} ki tamam {target_count} images successfully capture ho gayin!")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def list_existing_students():
    """Current dataset mein registered students ki summary dikhata hai."""
    if not os.path.exists(DATASET_DIR):
        print("\n[i] Abhi tak koi student dataset nahi bana.")
        return []

    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    if not folders:
        print("\n[i] Abhi tak koi student dataset nahi bana.")
        return []

    print("\n" + "="*50)
    print("   CURRENT ENROLLED STUDENTS IN DATASET")
    print("="*50)
    total_imgs = 0
    for idx, folder in enumerate(folders, 1):
        folder_path = os.path.join(DATASET_DIR, folder)
        imgs = len([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
        total_imgs += imgs
        status = "[READY]" if imgs >= 100 else f"[INCOMPLETE: {imgs}/100]"
        print(f" {idx}. {folder:<25} -> {imgs} images {status}")
    print("-" * 50)
    print(f" Total Students: {len(folders)} | Total Dataset Images: {total_imgs}")
    print("="*50 + "\n")
    return folders


def main():
    parser = argparse.ArgumentParser(description="VisionAttend AI - Biometric Dataset Collector")
    parser.add_argument("--id", type=str, help="Student Roll Number / Unique ID (e.g. 101)")
    parser.add_argument("--name", type=str, help="Student Name (e.g. Ali)")
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET_COUNT, help="Number of images to capture (Default: 100)")
    parser.add_argument("--zip", action="store_true", help="Zip the entire dataset folder for Google Colab upload")
    parser.add_argument("--list", action="store_true", help="List enrolled students and image counts")

    args = parser.parse_args()

    if args.zip:
        zip_dataset()
        return

    if args.list:
        list_existing_students()
        return

    print("=" * 60)
    print("    VISIONATTEND AI: BIOMETRIC DATASET COLLECTOR    ")
    print("=" * 60)
    list_existing_students()

    student_id = args.id
    if not student_id:
        student_id = input("Enter Student ID (e.g., 101): ").strip()
        while not student_id:
            student_id = input("ID khali nahi ho sakti! Enter Student ID: ").strip()

    student_name = args.name
    if not student_name:
        student_name = input("Enter Student Name (e.g., Ali Khan): ").strip()
        while not student_name:
            student_name = input("Name khali nahi ho sakta! Enter Student Name: ").strip()

    target_count = args.count

    capture_student_data(student_id, student_name, target_count)
    list_existing_students()

    ask_zip = input("Kya aap abhi Google Colab ke liye 'dataset.zip' banana chahte hain? (y/n): ").strip().lower()
    if ask_zip == 'y':
        zip_dataset()


if __name__ == "__main__":
    main()
