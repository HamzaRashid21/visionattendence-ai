"""
=============================================================================
VisionAttend AI - Smart Biometric Attendance System
Main Application Server (app.py)
-----------------------------------------------------------------------------
Features:
- Multi-Role Portals (Admin vs Student)
- Secure Authentication with Passwords Hashed (werkzeug.security)
- Whitelisted Entrance Kiosk Mode (No Login Required)
- Dynamic SQLite Tables (Users, AcademicClasses, Settings, Students, Attendance)
=============================================================================
"""

import os
import io
import cv2
import time
import base64
import zipfile
import json
import numpy as np
from functools import wraps
from datetime import datetime, date, timedelta
from flask import (
    Flask, render_template, Response, request, redirect,
    url_for, flash, jsonify, send_file, session, send_from_directory
)

from src.database import db, init_db, User, AcademicClass, Student, Attendance, Setting, ProxyAlert
from src.attendance_manager import (
    record_attendance,
    mark_absent_for_unmarked_students,
    generate_attendance_dataframe
)
from src.recognizer import process_frame

app = Flask(__name__)
app.config['SECRET_KEY'] = 'visionattend-enterprise-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

init_db(app)

# Global tracker for live recognition HUD (starts inactive until actual face detected)
_live_status = {
    "active": False,
    "student_id": None,
    "name": None,
    "department": None,
    "confidence": 0,
    "in_time": None,
    "duration_minutes": 0,
    "unknown": False,
    "last_update": 0
}


# ========================================================
# Database Seeding (Users, Classes, Settings & Students)
# ========================================================
def populate_sample_data_if_empty():
    """Initial sample users, classes, settings, and students add karta hai."""
    with app.app_context():
        # 1. Seed Academic Classes
        if AcademicClass.query.count() == 0:
            default_classes = [
                AcademicClass(class_code="BCS-1A", class_name="Bachelor of Computer Science", department="Computer Science", semester="Semester 1"),
                AcademicClass(class_code="BAI-3B", class_name="Bachelor of Artificial Intelligence", department="AI & Data Science", semester="Semester 3"),
                AcademicClass(class_code="BSE-5A", class_name="Software Engineering Lab", department="Software Engineering", semester="Semester 5")
            ]
            db.session.bulk_save_objects(default_classes)
            db.session.commit()
            print("[*] Default Academic Classes seeded.")

        # 2. Seed System Settings
        if Setting.query.count() == 0:
            default_settings = [
                Setting(key="institute_name", value="Apex Institute of Technology", description="Campus or Organization Name"),
                Setting(key="recognition_threshold", value="80", description="Minimum match confidence percentage"),
                Setting(key="cooldown_seconds", value="60", description="Duplicate scan lockout delay in seconds"),
                Setting(key="liveness_check_enabled", value="true", description="Enable anti-spoofing liveness verification"),
                Setting(key="auto_absent_time", value="17:00", description="Daily cutoff time for automated absent marking")
            ]
            db.session.bulk_save_objects(default_settings)
            db.session.commit()
            print("[*] Default System Settings seeded.")

        # 3. Seed Students
        if Student.query.count() == 0:
            samples = [
                Student(student_id="CS-001", name="Ali Raza", department="BCS - 1A", email="ali.raza@university.edu"),
                Student(student_id="CS-002", name="Ayesha Khan", department="BCS - 1A", email="ayesha.khan@university.edu"),
                Student(student_id="CS-003", name="Hamza Ali", department="BCS - 1A", email="hamza.ali@university.edu"),
                Student(student_id="CS-004", name="Saman Fatima", department="BCS - 1A", email="saman.f@university.edu"),
                Student(student_id="CS-005", name="Usman Tariq", department="BCS - 1A", email="usman.t@university.edu"),
                Student(student_id="CS-006", name="Zainab Noor", department="BCS - 1A", email="zainab.n@university.edu")
            ]
            db.session.bulk_save_objects(samples)
            db.session.commit()
            print("[*] Default Students seeded.")

        # 4. Seed Users (Admin and Students)
        if User.query.count() == 0:
            admin_user = User(username="admin", email="admin@university.edu", role="admin")
            admin_user.set_password("admin123")
            db.session.add(admin_user)

            student_user = User(username="CS-001", email="ali.raza@university.edu", role="student", student_id="CS-001")
            student_user.set_password("student123")
            db.session.add(student_user)

            # Ensure student profile also exists for CS-001
            if not Student.query.filter_by(student_id="CS-001").first():
                s = Student(student_id="CS-001", name="Ali Raza", department="BCS - 1A", email="ali.raza@university.edu")
                db.session.add(s)

            db.session.commit()
            print("[*] Default Admin (admin/admin123) and Student (CS-001/student123) users created.")


populate_sample_data_if_empty()


# ========================================================
# Authentication Middleware & Decorators
# ========================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this portal.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash("Unauthorized: Administrator permissions required.", "error")
            return redirect(url_for('student_dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'student':
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# ========================================================
# Authentication Routes (Login / Logout)
# ========================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form.get('login_id', '').strip() or request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Find user by username or email or student_id
        user = User.query.filter(
            (User.username == login_id) | (User.email == login_id) | (User.student_id == login_id)
        ).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['student_id'] = user.student_id

            flash(f"Welcome back, {user.username}!", "success")

            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        else:
            flash("Invalid credentials! Please verify your username and password.", "error")

    return render_template('login.html')


@app.route('/api/notifications')
def get_notifications():
    """Returns latest live attendance alerts and check-ins for the notification bell."""
    today = date.today()
    recent = Attendance.query.filter_by(date=today, status='Present').order_by(Attendance.id.desc()).limit(6).all()
    notifications = []
    for r in recent:
        notifications.append({
            "id": r.id,
            "title": f"{r.student.name if r.student else r.student_id} marked Present",
            "meta": f"{r.student.department if r.student else 'BCS - 1A'} · {r.in_time or 'Just now'}",
            "type": "success"
        })
    alerts = ProxyAlert.query.order_by(ProxyAlert.timestamp.desc()).limit(2).all()
    for a in alerts:
        notifications.append({
            "id": a.id,
            "title": f"Security Alert: {a.alert_type}",
            "meta": f"Target: {a.student_id_attempted or 'Unknown'} · {a.details}",
            "type": "warning"
        })
    return jsonify({"count": len(notifications), "notifications": notifications})


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for('login'))


# ========================================================
# Whitelisted Public Kiosk Mode (NO LOGIN REQUIRED)
# ========================================================
@app.route('/kiosk')
def kiosk():
    """
    Public Classroom Entrance Kiosk.
    Whitelisted route: Anyone can walk up to camera and get attendance marked.
    """
    inst_setting = Setting.query.filter_by(key='institute_name').first()
    institute_name = inst_setting.value if inst_setting else 'Apex Institute of Technology'
    return render_template('kiosk.html', institute_name=institute_name)


# ========================================================
# Student Portal Routes (Student Role Protected)
# ========================================================
@app.route('/student/dashboard')
@student_required
def student_dashboard():
    """Student personal portal view."""
    student_id = session.get('student_id', 'CS-001')
    student = Student.query.filter_by(student_id=student_id).first()

    if not student:
        flash("Student profile not found.", "error")
        return redirect(url_for('login'))

    my_records = Attendance.query.filter_by(student_id=student_id).order_by(Attendance.date.desc()).all()
    total_sessions = len(my_records) or 1
    present_count = sum(1 for r in my_records if r.status == 'Present')
    late_count = sum(1 for r in my_records if r.status == 'Late')
    absent_count = sum(1 for r in my_records if r.status == 'Absent')

    effective_present = present_count + (0.5 * late_count)
    attendance_pct = int((effective_present / total_sessions) * 100)

    return render_template(
        'student_dashboard.html',
        student=student,
        my_records=my_records,
        total_sessions=total_sessions,
        present_count=present_count,
        absent_count=absent_count,
        late_count=late_count,
        attendance_pct=attendance_pct
    )


# ========================================================
# High-Speed Thread-Safe Camera Manager (DirectShow on Windows)
# ========================================================
import threading

class GlobalCameraManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.current_jpeg = None
        self.clients = 0

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _open_camera(self):
        """Opens camera using DirectShow with sensor priming."""
        if self.cap is not None and self.cap.isOpened():
            return True
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Throw away first 3 frames to prime auto-exposure
            for _ in range(3):
                self.cap.read()
            print("[*] GlobalCameraManager: Hardware sensor initialized successfully.")
            return True
        except Exception as e:
            print(f"[!] Camera initialization error: {e}")
            return False

    def _get_threshold(self):
        """Fetches active confidence threshold percentage from SQLite system_settings."""
        try:
            with app.app_context():
                rec = Setting.query.filter_by(key="recognition_threshold").first()
                if rec and rec.value:
                    val = float(rec.value.strip())
                    return max(0.50, min(0.99, val / 100.0 if val > 1 else val))
        except Exception:
            pass
        return 0.80

    def start(self):
        with self._lock:
            self.clients += 1
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()

    def stop_client(self):
        with self._lock:
            self.clients = max(0, self.clients - 1)

    def _worker(self):
        global _live_status
        idle_ticks = 0
        threshold_check_time = 0
        current_threshold = 0.80

        while self.running:
            if self.clients > 0:
                idle_ticks = 0
                if self.cap is None or not self.cap.isOpened():
                    if not self._open_camera():
                        time.sleep(0.3)
                        continue

                # Refresh threshold from DB every 3 seconds
                if time.time() - threshold_check_time > 3.0:
                    current_threshold = self._get_threshold()
                    threshold_check_time = time.time()

                success, frame = self.cap.read()
                if not success or frame is None:
                    time.sleep(0.04)
                    continue

                frame = cv2.flip(frame, 1)
                annotated_frame, recognized_list = process_frame(frame, confidence_threshold=current_threshold)

                if recognized_list:
                    top_match = recognized_list[0]
                    with app.app_context():
                        status_ok, msg, att_data = record_attendance(top_match["student_id"])
                        stu = Student.query.filter_by(student_id=top_match["student_id"]).first()
                        dept = stu.department if stu else "Computer Science"
                        now_str = datetime.now().strftime("%I:%M %p")
                        
                        if att_data and att_data.get("in_time"):
                            logged_time = att_data.get("in_time")
                        else:
                            today_rec = Attendance.query.filter_by(student_id=top_match["student_id"], date=date.today()).first()
                            logged_time = today_rec.in_time if today_rec and today_rec.in_time else now_str

                        _live_status = {
                            "active": True,
                            "student_id": top_match["student_id"],
                            "name": top_match["name"],
                            "department": dept,
                            "confidence": top_match["confidence"],
                            "in_time": logged_time,
                            "duration_minutes": att_data.get("duration_minutes", 0) if att_data else 0,
                            "unknown": False,
                            "last_update": time.time()
                        }

                ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    self.current_jpeg = buffer.tobytes()

                time.sleep(0.025)
            else:
                # No client watching
                idle_ticks += 1
                if idle_ticks > 25 and self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    print("[*] Camera hardware released cleanly (idle mode).")
                time.sleep(0.2)


def generate_camera_frames():
    """Streams JPEG frames to client from the unified singleton camera manager."""
    manager = GlobalCameraManager.get_instance()
    manager.start()
    try:
        timeout = time.time() + 2.5
        while manager.current_jpeg is None and time.time() < timeout:
            time.sleep(0.05)

        while True:
            jpeg = manager.current_jpeg
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.033)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError):
        pass
    except Exception:
        pass
    finally:
        manager.stop_client()


@app.route('/video_feed')
def video_feed():
    """Video streaming route for camera feed (Whitelisted for kiosk and scanner)."""
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/live_recognition_status')
def live_recognition_status():
    """Returns current live recognized student for UI HUD with auto-expiry."""
    global _live_status
    # Auto-expire after 3.2 seconds if no face recently seen
    if _live_status.get("active") and (time.time() - _live_status.get("last_update", 0)) > 3.2:
        _live_status["active"] = False
    return jsonify(_live_status)


# ========================================================
# Admin Portal Routes (Admin Role Protected)
# ========================================================
@app.route('/')
@app.route('/dashboard')
@admin_required
def dashboard():
    """Main Admin Dashboard dynamically populated from SQLite database."""
    today = date.today()
    all_students = Student.query.order_by(Student.student_id.asc()).all()
    total_students = len(all_students)

    today_records = Attendance.query.filter_by(date=today).all()
    student_map = {s.student_id: s for s in all_students}

    present_today = sum(1 for r in today_records if r.status in ('Present', 'Late'))
    late_today = sum(1 for r in today_records if r.status == 'Late')
    absent_today = max(0, total_students - present_today)
    attendance_rate = f"{(present_today / total_students * 100):.1f}" if total_students > 0 else "0.0"

    # Recent attendance items from today (most recent first)
    recent_records = Attendance.query.filter_by(date=today).order_by(Attendance.id.desc()).limit(8).all()
    recent_items = []
    for r in recent_records:
        stu = student_map.get(r.student_id)
        recent_items.append({
            "name": stu.name if stu else r.student_id,
            "roll_no": r.student_id,
            "class_name": stu.department if stu else "Computer Science",
            "status": r.status,
            "time": r.in_time or "-"
        })

    # Past 7 Days Dynamic Weekly Analytics
    weekly_labels = []
    weekly_present = []
    weekly_absent = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        weekly_labels.append(day.strftime("%a"))
        day_records = Attendance.query.filter_by(date=day).all()
        p = sum(1 for r in day_records if r.status in ('Present', 'Late'))
        weekly_present.append(p)
        weekly_absent.append(max(0, total_students - p))

    # Model metrics status
    has_model = os.path.exists(os.path.join("models", "attendance_model.tflite")) or os.path.exists(os.path.join("models", "attendance_model.h5"))
    has_curves = os.path.exists(os.path.join("models", "training_curves.png"))
    has_confusion = os.path.exists(os.path.join("models", "confusion_matrix.png"))

    enrolled_classes_count = AcademicClass.query.count()

    # Dynamic trained model classes
    model_classes_list = []
    labels_file = os.path.join("models", "labels.json")
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r") as f:
                raw_labels = json.load(f)
                for k in sorted(raw_labels.keys(), key=lambda x: int(x)):
                    val = raw_labels[k]
                    parts = val.split("_", 1)
                    clean_name = parts[1].replace("_", " ") if len(parts) > 1 else val
                    model_classes_list.append(clean_name)
        except Exception:
            pass
    if model_classes_list:
        model_classes_str = ", ".join(model_classes_list) if len(model_classes_list) > 2 else " & ".join(model_classes_list)
    else:
        sample_students = [s.name for s in Student.query.limit(3).all()]
        model_classes_str = ", ".join(sample_students) if sample_students else "All Enrolled Students"

    now = datetime.now()
    today_str = now.strftime("%a, %d %b %Y")
    current_time_str = now.strftime("%I:%M %p")

    return render_template(
        'dashboard.html',
        active_page='dashboard',
        total_students=total_students,
        present_today=present_today,
        late_today=late_today,
        absent_today=absent_today,
        attendance_rate=attendance_rate,
        enrolled_classes_count=enrolled_classes_count,
        model_classes_str=model_classes_str,
        recent_items=recent_items,
        weekly_labels=weekly_labels,
        weekly_present=weekly_present,
        weekly_absent=weekly_absent,
        has_model=has_model,
        has_curves=has_curves,
        has_confusion=has_confusion,
        today_str=today_str,
        current_time_str=current_time_str
    )


@app.route('/model/metrics/<path:filename>')
@admin_required
def serve_model_metric(filename):
    """Serves model performance graphs and metric images."""
    models_dir = os.path.abspath("models")
    return send_from_directory(models_dir, filename)


@app.route('/model-metrics')
@admin_required
def model_metrics():
    """Redirect removed analytics page to dashboard."""
    return redirect(url_for('dashboard'))


@app.route('/attendance')
@admin_required
def attendance():
    """Attendance Management (Table Sheet by default, Camera on demand) with Date Filtering."""
    date_str = request.args.get('date', '').strip()
    selected_date = date.today()
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            pass

    all_students = Student.query.order_by(Student.student_id.asc()).all()
    all_classes = AcademicClass.query.order_by(AcademicClass.class_code.asc()).all()

    records = Attendance.query.filter_by(date=selected_date).all()
    rec_map = {r.student_id: r for r in records}

    student_attendance_list = []
    for s in all_students:
        att = rec_map.get(s.student_id)
        student_attendance_list.append({
            "student_id": s.student_id,
            "name": s.name,
            "department": s.department,
            "in_time": att.in_time if att else None,
            "out_time": att.out_time if att else None,
            "status": att.status if att else "Absent"
        })

    return render_template(
        'attendance.html',
        active_page='attendance',
        student_attendance_list=student_attendance_list,
        classes=all_classes,
        selected_date_str=selected_date.strftime("%Y-%m-%d"),
        today_str=selected_date.strftime("%d %b %Y"),
        is_today=(selected_date == date.today())
    )


@app.route('/attendance/manual/<student_id>/<status>', methods=['POST'])
@admin_required
def manual_mark_student(student_id, status):
    """Manually update student attendance status for current or selected date."""
    target_date = date.today()
    date_param = request.args.get('date', '').strip()
    if date_param:
        try:
            target_date = datetime.strptime(date_param, "%Y-%m-%d").date()
        except ValueError:
            pass

    now_str = datetime.now().strftime("%I:%M %p")
    att = Attendance.query.filter_by(student_id=student_id, date=target_date).first()
    student = Student.query.filter_by(student_id=student_id).first()
    student_name = student.name if student else student_id

    if not att:
        att = Attendance(
            student_id=student_id,
            date=target_date,
            in_time=now_str if status in ('Present', 'Late') else None,
            status=status,
            verification_method="Manual Admin Override"
        )
        db.session.add(att)
    else:
        att.status = status
        att.verification_method = "Manual Admin Override"
        if status in ('Present', 'Late') and not att.in_time:
            att.in_time = now_str

    db.session.commit()
    flash(f"Updated {student_name} to '{status}' for {target_date.strftime('%d %b %Y')}.", "success")
    return redirect(url_for('attendance', date=target_date.strftime("%Y-%m-%d")))


@app.route('/attendance/mark-all/<status>', methods=['POST'])
@admin_required
def mark_all_manual(status):
    """Bulk manual attendance marking for current or selected date."""
    target_date = date.today()
    date_param = request.args.get('date', '').strip()
    if date_param:
        try:
            target_date = datetime.strptime(date_param, "%Y-%m-%d").date()
        except ValueError:
            pass

    now_str = datetime.now().strftime("%I:%M %p")
    students = Student.query.all()

    for s in students:
        att = Attendance.query.filter_by(student_id=s.student_id, date=target_date).first()
        if not att:
            att = Attendance(
                student_id=s.student_id,
                date=target_date,
                in_time=now_str if status in ('Present', 'Late') else None,
                status=status,
                verification_method="Bulk Manual Action"
            )
            db.session.add(att)
        else:
            att.status = status

    db.session.commit()
    flash(f"All students marked '{status}' for {target_date.strftime('%d %b %Y')}.", "success")
    return redirect(url_for('attendance', date=target_date.strftime("%Y-%m-%d")))


@app.route('/students')
@admin_required
def students():
    """Student Profiles Management with dynamic search filter."""
    q = request.args.get('q', '').strip()
    if q:
        all_students = Student.query.filter(
            (Student.name.ilike(f"%{q}%")) |
            (Student.student_id.ilike(f"%{q}%")) |
            (Student.department.ilike(f"%{q}%"))
        ).order_by(Student.student_id.asc()).all()
    else:
        all_students = Student.query.order_by(Student.student_id.asc()).all()

    all_classes = AcademicClass.query.order_by(AcademicClass.class_code.asc()).all()
    # Check dataset counts for each student
    dataset_counts = {}
    if os.path.exists("dataset"):
        for folder in os.listdir("dataset"):
            folder_path = os.path.join("dataset", folder)
            if os.path.isdir(folder_path):
                st_id = folder.split("_")[0]
                imgs = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                dataset_counts[st_id] = imgs

    return render_template('students.html', active_page='students', students=all_students, classes=all_classes, dataset_counts=dataset_counts, search_query=q)


@app.route('/api/dataset/save_sample', methods=['POST'])
@admin_required
def save_dataset_sample():
    """
    Saves a verified, high-quality face sample frame from the browser webcam.
    Strictly verifies face presence using Haar Cascade face detection before saving.
    Crops with 10% padding, resizes to (160, 160), and applies histogram equalization (SRS Page 6).
    """
    data = request.get_json() or {}
    student_id = data.get('student_id', '').strip()
    student_name = data.get('student_name', '').strip()
    image_b64 = data.get('image_data', '')

    if not student_id or not image_b64:
        return jsonify({"success": False, "error": "Missing student ID or image data"}), 400

    # Ensure folder name format: "CS-001_Ali_Raza"
    clean_name = "".join(c for c in student_name if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    folder_name = f"{student_id}_{clean_name}"
    student_dir = os.path.join("dataset", folder_name)
    os.makedirs(student_dir, exist_ok=True)

    # Decode base64 image
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Unable to decode image"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    # Convert to grayscale for Haar face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join("models", "haarcascade_frontalface_default.xml")
    
    face_crop = None
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(70, 70))
        if len(faces) > 0:
            # Pick largest face in frame
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, fw, fh = faces[0]
            h, w = gray.shape
            pad_w = int(fw * 0.1)
            pad_h = int(fh * 0.1)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)
            face_crop = gray[y1:y2, x1:x2]

    # STRICT CHECK: If no face detected, do NOT count or save frame
    if face_crop is None or face_crop.size == 0:
        return jsonify({
            "success": False,
            "face_detected": False,
            "error": "No face detected in focus. Please align face inside the guide."
        }), 200

    # Resize to standard 160x160 and equalize histogram
    resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)

    # Count existing images
    existing = [f for f in os.listdir(student_dir) if f.lower().endswith('.jpg')]
    count = len(existing) + 1
    filename = f"{folder_name}_{count:03d}.jpg"
    filepath = os.path.join(student_dir, filename)
    cv2.imwrite(filepath, equalized)

    return jsonify({
        "success": True,
        "face_detected": True,
        "count": count,
        "target": 60,
        "student_id": student_id,
        "filename": filename
    })


@app.route('/api/dataset/clear/<student_id>', methods=['POST'])
@admin_required
def clear_dataset(student_id):
    """Clears existing dataset photos for a student to re-capture fresh samples."""
    cleared = 0
    if os.path.exists("dataset"):
        for folder in os.listdir("dataset"):
            if folder.startswith(f"{student_id}_") or folder == student_id:
                folder_path = os.path.join("dataset", folder)
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        os.remove(os.path.join(folder_path, f))
                        cleared += 1
    return jsonify({"success": True, "cleared": cleared})


@app.route('/api/dataset/export_zip')
@admin_required
def export_dataset_zip():
    """Zips the entire dataset/ directory into dataset.zip for Colab download."""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists("dataset"):
            for root, _, files in os.walk("dataset"):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        zipf.write(full_path, rel_path)

    memory_file.seek(0)
    return Response(
        memory_file.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=dataset.zip"}
    )


@app.route('/students/add', methods=['POST'])
@admin_required
def add_student():
    """Add a new student profile."""
    student_id = request.form.get('student_id', '').strip()
    name = request.form.get('name', '').strip()
    department = request.form.get('department', '').strip() or 'BCS - 1A'
    email = request.form.get('email', '').strip()

    if not student_id or not name:
        flash("Roll Number aur Name lazmi hain!", "error")
        return redirect(url_for('students'))

    existing = Student.query.filter_by(student_id=student_id).first()
    if existing:
        flash(f"Student ID '{student_id}' pehle se registered hai!", "error")
        return redirect(url_for('students'))

    new_student = Student(student_id=student_id, name=name, department=department, email=email)
    db.session.add(new_student)

    # Automatically create student login account
    if not User.query.filter_by(username=student_id).first():
        new_user = User(username=student_id, email=email or f"{student_id.lower()}@student.edu", role="student", student_id=student_id)
        new_user.set_password("student123")
        db.session.add(new_user)

    db.session.commit()
    flash(f"Student '{name}' (#{student_id}) enrolled with login credentials (Password: student123).", "success")
    return redirect(url_for('students'))


@app.route('/students/delete/<student_id>', methods=['POST'])
@admin_required
def delete_student(student_id):
    """Delete a student profile."""
    student = Student.query.filter_by(student_id=student_id).first()
    if student:
        user = User.query.filter_by(student_id=student_id).first()
        if user:
            db.session.delete(user)
        db.session.delete(student)
        db.session.commit()
        flash(f"Student #{student_id} delete ho gaya.", "success")
    return redirect(url_for('students'))


@app.route('/classes')
@admin_required
def classes():
    """Academic classes management from SQLite."""
    all_classes = AcademicClass.query.order_by(AcademicClass.class_code.asc()).all()
    return render_template('classes.html', active_page='classes', classes=all_classes)


@app.route('/classes/add', methods=['POST'])
@admin_required
def add_class():
    """Add a new academic class to SQLite."""
    class_code = request.form.get('class_code', '').strip()
    class_name = request.form.get('class_name', '').strip()
    department = request.form.get('department', '').strip() or 'Computer Science'
    semester = request.form.get('semester', '').strip() or 'Semester 1'

    if not class_code or not class_name:
        flash("Class Code aur Class Name lazmi hain!", "error")
        return redirect(url_for('classes'))

    existing = AcademicClass.query.filter_by(class_code=class_code).first()
    if existing:
        flash(f"Class '{class_code}' pehle se mojood hai!", "error")
        return redirect(url_for('classes'))

    new_class = AcademicClass(class_code=class_code, class_name=class_name, department=department, semester=semester)
    db.session.add(new_class)
    db.session.commit()
    flash(f"Class '{class_code}' successfully added!", "success")
    return redirect(url_for('classes'))


@app.route('/classes/delete/<int:class_id>', methods=['POST'])
@admin_required
def delete_class(class_id):
    """Delete an academic class from SQLite."""
    c = AcademicClass.query.get(class_id)
    if c:
        db.session.delete(c)
        db.session.commit()
        flash(f"Class '{c.class_code}' deleted.", "success")
    return redirect(url_for('classes'))


@app.route('/settings')
@admin_required
def settings():
    """System settings view backed by SQLite."""
    settings_records = Setting.query.all()
    settings_dict = {s.key: s.value for s in settings_records}
    return render_template('settings.html', active_page='settings', settings=settings_dict)


@app.route('/settings/update', methods=['POST'])
@admin_required
def update_settings():
    """Persist system configuration changes to SQLite."""
    for key, value in request.form.items():
        setting = Setting.query.filter_by(key=key).first()
        if setting:
            setting.value = value.strip()
        else:
            db.session.add(Setting(key=key, value=value.strip()))

    db.session.commit()
    flash("System settings saved successfully to SQLite database!", "success")
    return redirect(url_for('settings'))


@app.route('/reports')
@admin_required
def reports():
    """Attendance reports with date filtering."""
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')

    query = Attendance.query
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            query = query.filter(Attendance.date >= start_date)
        except ValueError:
            pass

    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            query = query.filter(Attendance.date <= end_date)
        except ValueError:
            pass

    records = query.order_by(Attendance.date.desc(), Attendance.id.desc()).all()
    return render_template('reports.html', active_page='reports', records=records,
                           start_date=start_date_str, end_date=end_date_str)


@app.route('/face-database')
@admin_required
def face_database():
    """Face biometric database viewer."""
    dataset_dir = "dataset"
    folders_data = []
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, f)
            if os.path.isdir(path):
                imgs = len([i for i in os.listdir(path) if i.lower().endswith('.jpg')])
                folders_data.append({"name": f, "count": imgs})

    return render_template('face_database.html', active_page='face_database', dataset_folders=folders_data)


@app.route('/export/csv')
@admin_required
def export_csv():
    """Exports attendance records to CSV."""
    df = generate_attendance_dataframe()
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=attendance_report_{date.today()}.csv"}
    )


@app.route('/export/excel')
@admin_required
def export_excel():
    """Exports attendance records to Excel (.xlsx)."""
    df = generate_attendance_dataframe()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Attendance', index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"attendance_report_{date.today()}.xlsx"
    )


@app.route('/action/auto_absent', methods=['POST'])
@admin_required
def run_auto_absent():
    """SRS Rule: Auto mark unverified students as Absent."""
    count = mark_absent_for_unmarked_students()
    flash(f"Auto-Absent engine executed! {count} remaining students marked Absent.", "success")
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print("   VISIONATTEND AI - SMART BIOMETRIC ATTENDANCE SYSTEM")
    print(f"   Server running at: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
