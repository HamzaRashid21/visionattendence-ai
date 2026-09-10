"""
=============================================================================
VisionAttend AI - Smart Biometric Attendance System
Module: Database Models & Engine (src/database.py)
-----------------------------------------------------------------------------
Maqsad (Purpose):
SQLite database ke andar Users (Admin / Student), Academic Classes, Students, 
Attendance Logs, System Settings, aur Proxy Alerts ko store karna.
=============================================================================
"""

import os
from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """
    Authentication & Role Management.
    Roles: 'admin' (Faculty/Administrator), 'student' (Student Portal user).
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="student", nullable=False)  # 'admin' or 'student'
    student_id = db.Column(db.String(50), nullable=True)                # Linked to Student.student_id if role=='student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "student_id": self.student_id
        }


class AcademicClass(db.Model):
    """
    Academic Classes & Batches (e.g., BCS - 1A, BAI - 3B).
    """
    __tablename__ = 'academic_classes'

    id = db.Column(db.Integer, primary_key=True)
    class_code = db.Column(db.String(50), unique=True, nullable=False, index=True)  # e.g., "BCS-1A"
    class_name = db.Column(db.String(100), nullable=False)                         # e.g., "Bachelor of Computer Science"
    department = db.Column(db.String(100), default="Computer Science")
    semester = db.Column(db.String(50), default="Semester 1")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "department": self.department,
            "semester": self.semester
        }


class Student(db.Model):
    """
    Registered Student profile table.
    """
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)  # e.g., "CS-001"
    name = db.Column(db.String(100), nullable=False)                                # e.g., "Ali Raza"
    department = db.Column(db.String(100), default="BCS - 1A")                      # Class / Department code
    email = db.Column(db.String(120), nullable=True)
    sample_image = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    attendances = db.relationship('Attendance', backref='student', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "name": self.name,
            "department": self.department,
            "email": self.email,
            "sample_image": self.sample_image,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else ""
        }


class Attendance(db.Model):
    """
    Daily attendance logs tracking In-time, Out-time, presence duration, and status.
    """
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), db.ForeignKey('students.student_id'), nullable=False, index=True)
    date = db.Column(db.Date, default=date.today, index=True)                       # e.g., 2026-09-09
    in_time = db.Column(db.String(20), nullable=True)                               # e.g., "10:22:15 AM"
    out_time = db.Column(db.String(20), nullable=True)                              # e.g., "01:30:10 PM"
    duration_minutes = db.Column(db.Integer, default=0)                             # Total duration in minutes
    status = db.Column(db.String(20), default="Present")                            # "Present", "Absent", "Late"
    verification_method = db.Column(db.String(50), default="Facial Recognition")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else "Unknown",
            "department": self.student.department if self.student else "N/A",
            "date": self.date.strftime("%Y-%m-%d") if self.date else "",
            "in_time": self.in_time or "--",
            "out_time": self.out_time or "--",
            "duration_minutes": self.duration_minutes,
            "status": self.status,
            "verification_method": self.verification_method
        }


class Setting(db.Model):
    """
    Dynamic System Configuration stored in SQLite.
    """
    __tablename__ = 'system_settings'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False, index=True)
    value = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "description": self.description
        }


class ProxyAlert(db.Model):
    """
    Security logs for suspected proxy / spoofing attempts.
    """
    __tablename__ = 'proxy_alerts'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(100), default="Spoofing Suspected")
    student_id_attempted = db.Column(db.String(50), nullable=True)
    details = db.Column(db.String(255), nullable=True)
    severity = db.Column(db.String(20), default="High")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else "",
            "alert_type": self.alert_type,
            "student_id_attempted": self.student_id_attempted or "Unknown",
            "details": self.details,
            "severity": self.severity
        }


def init_db(app):
    """Initializes the database and creates tables if they don't exist."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("[*] Database initialized successfully (all tables ready).")
