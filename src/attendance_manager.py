"""
=============================================================================
VisionAttend AI - Smart Biometric Attendance System
Module: Attendance Business Logic Engine (src/attendance_manager.py)
-----------------------------------------------------------------------------
Maqsad (Purpose):
SRS ke Rules ko enforce karna:
1. In-Time & Out-Time record karna aur presence duration calculate karna.
2. Anti-spam / Cooldown check taake 1 second mein 50 dafa attendance mark na ho.
3. Auto-Absent Marker: Jo students session/din ke aakhir mein scan nahi hue, 
   unko automatically 'Absent' mark karna.
4. Excel aur CSV report export generate karna.
=============================================================================
"""

from datetime import datetime, date
import pandas as pd
from src.database import db, Student, Attendance, ProxyAlert

# Anti-spam cooldown: Ek student agar abhi scan hua hai to agle 20 seconds tak 
# uski baar baar duplicate entry na ho (Smooth live experience).
SCAN_COOLDOWN_SECONDS = 20
_recent_scans = {}  # {student_id: timestamp}


def record_attendance(student_id: str, verification_method="Facial Recognition"):
    """
    Student ke face match hone par attendance mark ya out-time update karta hai.
    Returns: (status_code, message, attendance_dict)
    """
    now = datetime.now()
    today = date.today()
    time_str = now.strftime("%I:%M:%S %p")

    # 1. Verify student exists in enrolled database
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return False, f"Student ID '{student_id}' enrolled nahi hai!", None

    # 2. Cooldown check
    last_scanned = _recent_scans.get(student_id)
    if last_scanned:
        elapsed = (now - last_scanned).total_seconds()
        if elapsed < SCAN_COOLDOWN_SECONDS:
            remaining = int(SCAN_COOLDOWN_SECONDS - elapsed)
            return False, f"Already recorded! Please wait {remaining}s for next scan.", None

    _recent_scans[student_id] = now

    # 3. Check if attendance row exists for today
    att_record = Attendance.query.filter_by(student_id=student_id, date=today).first()

    if not att_record:
        # First scan of the day -> Mark PRESENT with IN-TIME
        att_record = Attendance(
            student_id=student_id,
            date=today,
            in_time=time_str,
            out_time=None,
            duration_minutes=0,
            status="Present",
            verification_method=verification_method
        )
        db.session.add(att_record)
        db.session.commit()
        return True, f"Welcome {student.name}! In-time marked at {time_str}", att_record.to_dict()
    else:
        # Repeat scan on the same day -> Update OUT-TIME and calculate duration
        att_record.out_time = time_str

        # Calculate presence duration in minutes
        if att_record.in_time:
            try:
                t_in = datetime.strptime(att_record.in_time, "%I:%M:%S %p")
                t_out = datetime.strptime(time_str, "%I:%M:%S %p")
                duration = int((t_out - t_in).total_seconds() / 60)
                att_record.duration_minutes = max(0, duration)
            except Exception:
                pass

        db.session.commit()
        return True, f"Goodbye {student.name}! Out-time updated at {time_str} (Duration: {att_record.duration_minutes} mins)", att_record.to_dict()


def mark_absent_for_unmarked_students(target_date=None):
    """
    SRS Requirement (Page 6 Step 6 & Page 8 vii):
    "Once all present students have been marked, the remaining enrolled students
     who were not detected will be automatically recorded as absent."
    """
    if target_date is None:
        target_date = date.today()

    all_students = Student.query.all()
    absent_count = 0

    for student in all_students:
        existing = Attendance.query.filter_by(student_id=student.student_id, date=target_date).first()
        if not existing:
            # Student was not detected today -> Create Absent record
            absent_record = Attendance(
                student_id=student.student_id,
                date=target_date,
                in_time=None,
                out_time=None,
                duration_minutes=0,
                status="Absent",
                verification_method="System Auto-Marker"
            )
            db.session.add(absent_record)
            absent_count += 1

    db.session.commit()
    return absent_count


def generate_attendance_dataframe(start_date=None, end_date=None):
    """
    Generates a clean Pandas DataFrame of attendance logs for reporting & analytics.
    """
    query = Attendance.query
    if start_date:
        query = query.filter(Attendance.date >= start_date)
    if end_date:
        query = query.filter(Attendance.date <= end_date)

    records = query.order_by(Attendance.date.desc(), Attendance.id.desc()).all()
    data = [r.to_dict() for r in records]

    if not data:
        return pd.DataFrame(columns=[
            "ID", "Roll No", "Student Name", "Department", "Date",
            "In-Time", "Out-Time", "Duration (Mins)", "Status", "Method"
        ])

    df = pd.DataFrame(data)
    df.rename(columns={
        "id": "Record ID",
        "student_id": "Roll No",
        "student_name": "Student Name",
        "department": "Department",
        "date": "Date",
        "in_time": "In-Time",
        "out_time": "Out-Time",
        "duration_minutes": "Duration (Mins)",
        "status": "Status",
        "verification_method": "Method"
    }, inplace=True)
    return df
