#!/usr/bin/env python3

import sys
import os
import json
import sqlite3
import datetime
import time
import select
import multiprocessing as mp
import threading
import numpy as np
import configparser
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QStackedWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QComboBox, QLineEdit,
    QSpinBox, QTextEdit, QTabWidget, QHeaderView, QScrollArea, QDoubleSpinBox,
    QPlainTextEdit, QCheckBox, QSplitter, QProgressDialog, QShortcut, QSlider
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from scipy.signal import welch, butter, filtfilt, spectrogram, hilbert, find_peaks
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.interpolate import interp1d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.load()
    
    def load(self):
        self.config.read('vibration_lab.ini')
        if 'Settings' not in self.config:
            self.config['Settings'] = {
                'db_path': 'vibration_lab.db',
                'default_fs': '1000',
                'alarm_rms': '0.5',
                'alarm_peak': '1.0',
                'mqtt_broker': 'localhost',
                'mqtt_port': '1883',
                'mqtt_topic': 'vibration/data'
            }
            self.save()
    
    def save(self):
        with open('vibration_lab.ini', 'w') as f:
            self.config.write(f)
    
    def get(self, key, fallback=''):
        return self.config.get('Settings', key, fallback=fallback)
    
    def set(self, key, value):
        self.config.set('Settings', key, value)
        self.save()

class DatabaseManager:
    def __init__(self, db_path="vibration_lab.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                test_type TEXT,
                machine_id INTEGER,
                results_json TEXT,
                notes TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                config_json TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS bearing_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER,
                bpfi REAL,
                bpfo REAL,
                bsf REAL,
                ftf REAL,
                envelope_peaks TEXT
            )''')
            try:
                c.execute("ALTER TABLE tests ADD COLUMN machine_id INTEGER")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()

    def add_machine(self, name, description=""):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO machines (name, description) VALUES (?, ?)", (name, description))
                conn.commit()
                return c.lastrowid
            except:
                return None
            finally:
                conn.close()

    def get_machines(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT id, name, description FROM machines")
            rows = c.fetchall()
            conn.close()
            return rows

    def save_test(self, test_type, results_dict, machine_id=None, notes=""):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO tests (timestamp, test_type, machine_id, results_json, notes) VALUES (?, ?, ?, ?, ?)",
                      (datetime.datetime.now().isoformat(), test_type, machine_id, json.dumps(results_dict), notes))
            test_id = c.lastrowid
            conn.commit()
            conn.close()
            return test_id

    def get_tests(self, test_type=None, machine_id=None):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            query = "SELECT id, timestamp, test_type, machine_id, results_json, notes FROM tests"
            params = []
            conditions = []
            if test_type:
                conditions.append("test_type=?")
                params.append(test_type)
            if machine_id:
                conditions.append("machine_id=?")
                params.append(machine_id)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            c.execute(query, params)
            rows = c.fetchall()
            conn.close()
            return rows

class AsyncSerialReader(QThread):
    data_received = pyqtSignal(float)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.active = False
        self.ser = None

    def run(self):
        if not SERIAL_AVAILABLE:
            return
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
            self.active = True
            while self.active:
                r, w, e = select.select([self.ser], [], [], 0.01)
                if r:
                    line = self.ser.readline().decode().strip()
                    if line:
                        try:
                            val = float(line)
                            self.data_received.emit(val)
                        except:
                            pass
                else:
                    time.sleep(0.001)
        except Exception as e:
            print(f"Serial error: {e}")
        finally:
            if self.ser:
                self.ser.close()

    def stop(self):
        self.active = False
        self.wait()

def cwt_process(data, scales):
    if not WAVELET_AVAILABLE:
        return None
    coeffs, freqs = pywt.cwt(data, scales, 'cmor', sampling_period=1.0)
    return coeffs

class CalibrationModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.offset = 0.0
        self.gain = 1.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sensor Calibration"))
        form = QFormLayout()
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e6, 1e6)
        self.offset_spin.setValue(0.0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.001, 1e6)
        self.gain_spin.setValue(1.0)
        form.addRow("Offset:", self.offset_spin)
        form.addRow("Gain:", self.gain_spin)
        self.apply_btn = QPushButton("Apply Calibration")
        self.apply_btn.clicked.connect(self.apply_calibration)
        layout.addLayout(form)
        layout.addWidget(self.apply_btn)

    def apply_calibration(self):
        self.offset = self.offset_spin.value()
        self.gain = self.gain_spin.value()
        QMessageBox.information(self, "Calibration", f"Offset={self.offset}, Gain={self.gain}")

    def calibrate(self, raw_value):
        return (raw_value - self.offset) * self.gain

class AIModel:
    def __init__(self, model_path=None):
        self.model = None
        self.model_type = None
        if model_path and os.path.exists(model_path):
            if model_path.endswith('.pth') and TORCH_AVAILABLE:
                self.model = self._load_torch(model_path)
                self.model_type = 'torch'
            elif model_path.endswith('.h5') and TF_AVAILABLE:
                self.model = self._load_tf(model_path)
                self.model_type = 'tf'
        else:
            self.model = None

    def _load_torch(self, path):
        model = torch.load(path, map_location='cpu')
        model.eval()
        return model

    def _load_tf(self, path):
        return tf.keras.models.load_model(path)

    def predict(self, signal):
        if self.model is None:
            if len(signal) == 0:
                return "Unknown", 0.0
            rms = np.sqrt(np.mean(signal**2))
            kurtosis_val = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-12)
            if kurtosis_val > 5:
                return "Bearing Fault", 0.85
            elif rms < 0.2:
                return "Normal", 0.85
            elif rms < 0.5:
                return "Bearing Fault", 0.70
            else:
                return "Misalignment", 0.75

        if len(signal) > 1024:
            signal = signal[:1024]
        else:
            signal = np.pad(signal, (0, 1024 - len(signal)))

        if self.model_type == 'torch':
            with torch.no_grad():
                x = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
                out = self.model(x)
                pred = torch.argmax(out, dim=1).item()
            classes = ["Normal", "Bearing Fault", "Misalignment"]
            return classes[pred], 0.9
        elif self.model_type == 'tf':
            x = signal.reshape(1, 1024, 1)
            pred = np.argmax(self.model.predict(x, verbose=0)[0])
            classes = ["Normal", "Bearing Fault", "Misalignment"]
            return classes[pred], 0.85
        else:
            return "Unknown", 0.0

class BearingFaultSimulator:
    @staticmethod
    def generate_bearing_signal(fs, duration, shaft_speed, bpfi, bpfo, bsf, ftf, fault_type='outer'):
        t = np.linspace(0, duration, int(fs*duration))
        signal = 0.5 * np.sin(2 * np.pi * shaft_speed/60 * t)
        if fault_type == 'outer':
            freq = bpfo
        elif fault_type == 'inner':
            freq = bpfi
        elif fault_type == 'ball':
            freq = bsf
        else:
            freq = ftf
        impulses = np.zeros_like(t)
        impulse_period = 1.0 / freq
        for i in range(int(duration * freq)):
            idx = int(i * impulse_period * fs)
            if idx < len(t):
                impulses[idx] = 1.0
        impulse_response = np.exp(-50 * t) * np.sin(2 * np.pi * 2000 * t)
        fault_signal = np.convolve(impulses, impulse_response, mode='same')[:len(t)]
        noise = 0.1 * np.random.randn(len(t))
        signal = signal + 0.3 * fault_signal + noise
        return t, signal

class OrderTracker:
    @staticmethod
    def resample_to_order(time, signal, tacho_pulses, order_resolution=0.1, max_order=50):
        if len(tacho_pulses) < 2:
            return None, None
        pulse_times = np.array(tacho_pulses)
        rpm = 60.0 / np.diff(pulse_times)
        angle = np.cumsum(np.diff(pulse_times) * np.interp(pulse_times[1:], pulse_times[:-1], rpm) / 60.0 * 360)
        angle = np.insert(angle, 0, 0)
        angle_deg = angle * 360 / (2 * np.pi)
        orders = np.arange(0, max_order, order_resolution)
        resampler = interp1d(angle_deg, signal, kind='linear', fill_value='extrapolate')
        signal_order = resampler(orders)
        return orders, signal_order

class MQTTClient:
    def __init__(self, broker="localhost", port=1883, topic="vibration/data"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = None
        self.connected = False
        if MQTT_AVAILABLE:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect

    def on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)

    def connect(self):
        if self.client and MQTT_AVAILABLE:
            try:
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
                return True
            except:
                return False
        return False

    def publish(self, data):
        if self.client and self.connected and MQTT_AVAILABLE:
            try:
                self.client.publish(self.topic, json.dumps(data))
                return True
            except:
                return False
        return False

    def disconnect(self):
        if self.client and MQTT_AVAILABLE:
            self.client.loop_stop()
            self.client.disconnect()

class LogPanel(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)
        self.setStyleSheet("font-family: monospace;")

    def log(self, msg):
        self.appendPlainText(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

class WaterfallPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectrograms = []
        self.freqs = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Time Index")
        self.ax.set_zlabel("PSD (dB)")
        layout.addWidget(self.canvas)
    
    def add_spectrum(self, freqs, psd):
        if self.freqs is None:
            self.freqs = freqs
        self.spectrograms.append(psd)
        if len(self.spectrograms) > 30:
            self.spectrograms.pop(0)
        self.update_plot()
    
    def update_plot(self):
        if not self.spectrograms:
            return
        self.ax.clear()
        X, Y = np.meshgrid(self.freqs, np.arange(len(self.spectrograms)))
        Z = np.array(self.spectrograms)
        Z_db = 10 * np.log10(Z + 1e-12)
        self.ax.plot_surface(X, Y, Z_db, cmap='viridis')
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Time Index")
        self.ax.set_zlabel("PSD (dB)")
        self.canvas.draw()

class SignalAnalytics:
    @staticmethod
    def compute_iri(distance, elevation, speed_kmh=80.0):
        distance = np.asarray(distance)
        elevation = np.asarray(elevation)
        if len(distance) < 2:
            return 0.0
        dx = distance[1] - distance[0]
        if not np.allclose(np.diff(distance), dx, rtol=1e-5):
            x_uniform = np.arange(distance[0], distance[-1], dx)
            elev_uniform = np.interp(x_uniform, distance, elevation)
            distance = x_uniform
            elevation = elev_uniform
            dx = distance[1] - distance[0]
        v = speed_kmh / 3.6
        dt = dx / v
        ms, mu = 250.0, 35.0
        ks, kt = 16000.0, 160000.0
        cs = 1000.0
        state = np.zeros(4)
        iri_acc = 0.0
        zr = elevation / 1000.0
        for i in range(1, len(distance)):
            ddzu = (kt * (zr[i-1] - state[2]) + ks * (state[0] - state[2]) + cs * (state[1] - state[3])) / mu - 9.81
            ddzs = (ks * (state[2] - state[0]) + cs * (state[3] - state[1])) / ms
            state[0] += state[1] * dt
            state[1] += ddzs * dt
            state[2] += state[3] * dt
            state[3] += ddzu * dt
            iri_acc += abs(state[1] - state[3]) * dt
        total_length = distance[-1] - distance[0]
        if total_length <= 0:
            return 0.0
        return (iri_acc / total_length) * 1000.0

    @staticmethod
    def compute_statistics(signal):
        signal = np.asarray(signal)
        if len(signal) == 0:
            return {}
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / (rms + 1e-12)
        skewness = np.mean((signal - mean_val)**3) / (std_val**3 + 1e-12)
        kurtosis = np.mean((signal - mean_val)**4) / (std_val**4 + 1e-12)
        clearance = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-12)
        shape_factor = rms / (np.mean(np.abs(signal)) + 1e-12)
        impulse_factor = peak / (np.mean(np.abs(signal)) + 1e-12)
        return {
            'RMS': rms,
            'Peak': peak,
            'Crest Factor': crest_factor,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Clearance Factor': clearance,
            'Shape Factor': shape_factor,
            'Impulse Factor': impulse_factor
        }

    @staticmethod
    def envelope_spectrum(signal, fs, lowcut=100, highcut=10000):
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        envelope = envelope - np.mean(envelope)
        f, Pxx = welch(envelope, fs, nperseg=min(1024, len(envelope)), scaling='density')
        mask = (f >= lowcut) & (f <= highcut)
        return f[mask], Pxx[mask]

    @staticmethod
    def bearing_characteristic_frequencies(shaft_rpm, n_balls, ball_diameter, pitch_diameter, contact_angle):
        fr = shaft_rpm / 60.0
        bd = ball_diameter
        pd = pitch_diameter
        phi = np.radians(contact_angle)
        bpfi = (n_balls / 2.0) * fr * (1 + (bd / pd) * np.cos(phi))
        bpfo = (n_balls / 2.0) * fr * (1 - (bd / pd) * np.cos(phi))
        bsf = (pd / (2.0 * bd)) * fr * (1 - (bd / pd)**2 * np.cos(phi)**2)
        ftf = (fr / 2.0) * (1 - (bd / pd) * np.cos(phi))
        return bpfi, bpfo, bsf, ftf

class RoadRoughnessModule(QWidget):
    def __init__(self, parent=None, db=None, log=None):
        super().__init__(parent)
        self.db = db
        self.log = log
        self.dist = None
        self.elev = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("ROAD ROUGHNESS (IRI)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FFFF;")
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.load_btn)
        layout.addLayout(top)

        self.fig = Figure(figsize=(8, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Elevation (mm)")
        layout.addWidget(self.canvas)

        stats_group = QGroupBox("Results")
        form = QFormLayout()
        self.iri_label = QLabel("-- m/km")
        self.length_label = QLabel("-- m")
        self.points_label = QLabel("--")
        form.addRow("IRI:", self.iri_label)
        form.addRow("Length:", self.length_label)
        form.addRow("Points:", self.points_label)
        stats_group.setLayout(form)
        layout.addWidget(stats_group)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Dist (m)", "Elev (mm)", "Slope"])
        layout.addWidget(self.table)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
            if data.shape[1] < 2:
                raise ValueError("Need 2 columns")
            dist = data[:, 0]
            elev = data[:, 1]
            valid = ~(np.isnan(dist) | np.isnan(elev))
            self.set_data(dist[valid], elev[valid])
            if self.log:
                self.log.log(f"Loaded road profile: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def set_data(self, distance, elevation):
        if len(distance) < 2:
            QMessageBox.warning(self, "Error", "Need >=2 points")
            return
        self.dist = np.array(distance)
        self.elev = np.array(elevation)
        iri = SignalAnalytics.compute_iri(self.dist, self.elev)
        self.iri_label.setText(f"{iri:.2f} m/km")
        self.length_label.setText(f"{self.dist[-1]-self.dist[0]:.2f} m")
        self.points_label.setText(str(len(self.dist)))
        self.ax.clear()
        self.ax.plot(self.dist, self.elev, 'b-')
        self.ax.set_title(f"IRI = {iri:.2f} m/km")
        self.canvas.draw()
        self.update_table()
        if self.db:
            self.db.save_test("Road", {"iri": iri, "length": self.length_label.text()})
        if self.log:
            self.log.log(f"IRI computed: {iri:.2f} m/km")

    def update_table(self):
        n = min(20, len(self.dist))
        self.table.setRowCount(n)
        slopes = np.gradient(self.elev, self.dist)
        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(f"{self.dist[i]:.3f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{self.elev[i]:.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{slopes[i]:.2f}"))

class SeismicModule(QWidget):
    def __init__(self, parent=None, db=None, log=None):
        super().__init__(parent)
        self.db = db
        self.log = log
        self.time = None
        self.acc = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("SEISMIC (PGA/PSD)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FFFF;")
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.load_btn)
        top.addWidget(self.sim_btn)
        layout.addLayout(top)

        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax_time = self.fig.add_subplot(211)
        self.ax_psd = self.fig.add_subplot(212)
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Accel (g)")
        self.ax_psd.set_xlabel("Freq (Hz)")
        self.ax_psd.set_ylabel("PSD")
        layout.addWidget(self.canvas)

        stats = QGroupBox("Parameters")
        form = QFormLayout()
        self.pga_label = QLabel("-- g")
        self.pgv_label = QLabel("-- m/s")
        self.dur_label = QLabel("-- s")
        form.addRow("PGA:", self.pga_label)
        form.addRow("PGV:", self.pgv_label)
        form.addRow("Duration:", self.dur_label)
        stats.setLayout(form)
        layout.addWidget(stats)

        self.simulate()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
            if data.shape[1] < 2:
                raise ValueError("Need time and accel columns")
            self.set_data(data[:, 0], data[:, 1])
            if self.log:
                self.log.log(f"Loaded seismic data: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def simulate(self):
        fs = 200.0
        t = np.linspace(0, 20, int(fs*20))
        envelope = np.exp(-0.2 * t)
        signal = envelope * np.sin(2 * np.pi * (1 + 14 * t/20) * t)
        self.set_data(t, signal * 0.5)
        if self.log:
            self.log.log("Generated synthetic earthquake signal")

    def set_data(self, time, accel):
        if len(time) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 points")
            return
        self.time = np.array(time)
        self.acc = np.array(accel)
        pga = np.max(np.abs(self.acc))
        self.pga_label.setText(f"{pga:.4f} g")
        dt = self.time[1] - self.time[0]
        if len(self.acc) > 1:
            vel = np.cumsum(self.acc[1:] * np.diff(self.time)) * 9.81
            pgv = np.max(np.abs(vel)) if len(vel) > 0 else 0.0
        else:
            pgv = 0.0
        self.pgv_label.setText(f"{pgv:.3f} m/s")
        self.dur_label.setText(f"{self.time[-1]-self.time[0]:.1f} s")
        self.ax_time.clear()
        self.ax_time.plot(self.time, self.acc, 'g-')
        self.ax_time.set_title(f"PGA = {pga:.4f} g")
        fs = 1.0 / dt
        f, Pxx = welch(self.acc, fs, nperseg=min(256, len(self.acc)), scaling='density')
        self.ax_psd.clear()
        self.ax_psd.semilogy(f, Pxx, 'purple')
        self.canvas.draw()
        if self.db:
            self.db.save_test("Seismic", {"pga": pga, "pgv": pgv})

class CantileverBeamModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("CANTILEVER BEAM")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FFFF;")
        layout.addWidget(title)

        input_group = QGroupBox("Parameters")
        form = QFormLayout()

        self.length_val = QDoubleSpinBox()
        self.length_val.setRange(0, 1e9)
        self.length_val.setValue(1.0)
        self.length_unit = QComboBox()
        self.length_unit.addItems(["m", "mm"])

        self.width_val = QDoubleSpinBox()
        self.width_val.setRange(0, 1e9)
        self.width_val.setValue(0.05)
        self.width_unit = QComboBox()
        self.width_unit.addItems(["m", "mm"])

        self.height_val = QDoubleSpinBox()
        self.height_val.setRange(0, 1e9)
        self.height_val.setValue(0.01)
        self.height_unit = QComboBox()
        self.height_unit.addItems(["m", "mm"])

        self.young_val = QDoubleSpinBox()
        self.young_val.setRange(0, 1e12)
        self.young_val.setValue(200e9)
        self.young_unit = QComboBox()
        self.young_unit.addItems(["Pa", "GPa"])

        self.density_val = QDoubleSpinBox()
        self.density_val.setRange(0, 1e9)
        self.density_val.setValue(7850)
        self.density_val.setSuffix(" kg/m³")

        self.force_val = QDoubleSpinBox()
        self.force_val.setRange(0, 1e12)
        self.force_val.setValue(100)
        self.force_unit = QComboBox()
        self.force_unit.addItems(["N", "kN"])

        def hbox(spin, combo):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0,0,0,0)
            h.addWidget(spin)
            h.addWidget(combo)
            return w

        form.addRow("Length:", hbox(self.length_val, self.length_unit))
        form.addRow("Width:", hbox(self.width_val, self.width_unit))
        form.addRow("Height:", hbox(self.height_val, self.height_unit))
        form.addRow("Young's Modulus:", hbox(self.young_val, self.young_unit))
        form.addRow("Density:", self.density_val)
        form.addRow("End Force:", hbox(self.force_val, self.force_unit))
        input_group.setLayout(form)
        layout.addWidget(input_group)

        self.calc_btn = QPushButton("Compute")
        self.calc_btn.clicked.connect(self.compute)
        layout.addWidget(self.calc_btn)

        results_group = QGroupBox("Results")
        results_form = QFormLayout()
        self.deflection_label = QLabel("--")
        self.stress_label = QLabel("--")
        self.freq_label = QLabel("--")
        results_form.addRow("Max Deflection:", self.deflection_label)
        results_form.addRow("Max Stress:", self.stress_label)
        results_form.addRow("Natural Frequency:", self.freq_label)
        results_group.setLayout(results_form)
        layout.addWidget(results_group)

        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Position (m)")
        self.ax.set_ylabel("Deflection (mm)")
        layout.addWidget(self.canvas)

        self.compute()

    def compute(self):
        L = self.length_val.value()
        if self.length_unit.currentText() == "mm":
            L /= 1000.0
        b = self.width_val.value()
        if self.width_unit.currentText() == "mm":
            b /= 1000.0
        h = self.height_val.value()
        if self.height_unit.currentText() == "mm":
            h /= 1000.0
        E = self.young_val.value()
        if self.young_unit.currentText() == "GPa":
            E *= 1e9
        rho = self.density_val.value()
        F = self.force_val.value()
        if self.force_unit.currentText() == "kN":
            F *= 1000.0

        I = (b * h**3) / 12.0
        if I <= 0:
            self.deflection_label.setText("Invalid I (zero)")
            self.stress_label.setText("Invalid I")
            self.freq_label.setText("Invalid I")
            return

        delta_m = (F * L**3) / (3 * E * I)
        delta_mm = delta_m * 1000.0
        self.deflection_label.setText(f"{delta_mm:.3f} mm")

        M = F * L
        y_max = h / 2.0
        sigma_pa = M * y_max / I
        sigma_mpa = sigma_pa / 1e6
        self.stress_label.setText(f"{sigma_mpa:.2f} MPa")

        A = b * h
        if A <= 0:
            self.freq_label.setText("Invalid area")
            return
        omega = (1.875104**2) / (L**2) * np.sqrt((E * I) / (rho * A))
        f_hz = omega / (2 * np.pi)
        self.freq_label.setText(f"{f_hz:.2f} Hz")

        x = np.linspace(0, L, 100)
        defl = (F * x**2) / (6 * E * I) * (3*L - x)
        defl_mm = defl * 1000.0
        self.ax.clear()
        self.ax.plot(x, defl_mm, 'b-', linewidth=2)
        self.ax.set_xlabel("Position (m)")
        self.ax.set_ylabel("Deflection (mm)")
        self.ax.grid(True)
        self.canvas.draw()

class VibrationModule(QWidget):
    def __init__(self, parent=None, db=None, log=None):
        super().__init__(parent)
        self.db = db
        self.log = log
        self.serial_reader = None
        self.time_buffer = []
        self.data_buffer = []
        self.fs = 1000.0
        self.current_filtered = None
        self.ai_model = AIModel()
        self.ollama = OllamaEngineer()
        self.calib = CalibrationModule()
        self.plot_update_counter = 0
        self.waterfall = WaterfallPlot()
        self.alarm_rms = 0.5
        self.alarm_peak = 1.0
        self.config = Config()
        self.mqtt_client = None
        self.mqtt_enabled = False
        self.fft_data_buffer = deque(maxlen=100)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        rt_tab = QWidget()
        rt_layout = QVBoxLayout(rt_tab)
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        if SERIAL_AVAILABLE:
            for port in serial.tools.list_ports.comports():
                self.port_combo.addItem(port.device)
        self.port_combo.addItem("Simulated")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        self.mqtt_btn = QPushButton("MQTT Off")
        self.mqtt_btn.clicked.connect(self.toggle_mqtt)
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.port_combo)
        port_layout.addStretch()
        port_layout.addWidget(self.start_btn)
        port_layout.addWidget(self.stop_btn)
        port_layout.addWidget(self.export_btn)
        port_layout.addWidget(self.mqtt_btn)
        rt_layout.addLayout(port_layout)

        plot_splitter = QSplitter(Qt.Horizontal)
        self.live_plot = pg.PlotWidget()
        self.live_plot.setLabel('left', 'Amplitude')
        self.live_plot.setLabel('bottom', 'Time (s)')
        self.live_curve = self.live_plot.plot(pen='b')
        plot_splitter.addWidget(self.live_plot)

        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fft_curve = self.fft_plot.plot(pen='r')
        plot_splitter.addWidget(self.fft_plot)
        rt_layout.addWidget(plot_splitter)

        metrics_group = QGroupBox("Metrics")
        metrics_form = QFormLayout()
        self.rms_label = QLabel("--")
        self.peak_label = QLabel("--")
        self.dom_freq_label = QLabel("--")
        self.kurtosis_label = QLabel("--")
        self.crest_label = QLabel("--")
        metrics_form.addRow("RMS:", self.rms_label)
        metrics_form.addRow("Peak:", self.peak_label)
        metrics_form.addRow("Dominant Freq (Hz):", self.dom_freq_label)
        metrics_form.addRow("Kurtosis:", self.kurtosis_label)
        metrics_form.addRow("Crest Factor:", self.crest_label)
        metrics_group.setLayout(metrics_form)
        rt_layout.addWidget(metrics_group)
        tabs.addTab(rt_tab, "Real‑time")

        adv_tab = QWidget()
        adv_layout = QVBoxLayout(adv_tab)
        filter_group = QGroupBox("Butterworth Filter")
        filter_hlayout = QHBoxLayout()
        self.filter_type = QComboBox()
        self.filter_type.addItems(["lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QLineEdit("100")
        self.filter_cutoff2 = QLineEdit("1000")
        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 8)
        self.filter_order.setValue(4)
        self.filter_apply_btn = QPushButton("Apply")
        self.filter_apply_btn.clicked.connect(self.apply_filter)
        filter_hlayout.addWidget(QLabel("Type:"))
        filter_hlayout.addWidget(self.filter_type)
        filter_hlayout.addWidget(QLabel("Cutoff (Hz):"))
        filter_hlayout.addWidget(self.filter_cutoff)
        filter_hlayout.addWidget(QLabel("Cutoff2 (Hz):"))
        filter_hlayout.addWidget(self.filter_cutoff2)
        filter_hlayout.addWidget(QLabel("Order:"))
        filter_hlayout.addWidget(self.filter_order)
        filter_hlayout.addWidget(self.filter_apply_btn)
        filter_group.setLayout(filter_hlayout)
        adv_layout.addWidget(filter_group)

        btn_layout = QHBoxLayout()
        self.cwt_btn = QPushButton("CWT (parallel)")
        self.psd_btn = QPushButton("PSD")
        self.spec_btn = QPushButton("Spectrogram")
        self.envelope_btn = QPushButton("Envelope Spectrum")
        btn_layout.addWidget(self.cwt_btn)
        btn_layout.addWidget(self.psd_btn)
        btn_layout.addWidget(self.spec_btn)
        btn_layout.addWidget(self.envelope_btn)
        adv_layout.addLayout(btn_layout)

        self.adv_fig = Figure(figsize=(8, 4))
        self.adv_canvas = FigureCanvas(self.adv_fig)
        self.adv_ax = self.adv_fig.add_subplot(111)
        adv_layout.addWidget(self.adv_canvas)

        self.cwt_btn.clicked.connect(self.compute_cwt_parallel)
        self.psd_btn.clicked.connect(self.compute_psd)
        self.spec_btn.clicked.connect(self.compute_spectrogram)
        self.envelope_btn.clicked.connect(self.compute_envelope_spectrum)
        tabs.addTab(adv_tab, "Advanced")

        bearing_tab = QWidget()
        bearing_layout = QVBoxLayout(bearing_tab)
        bearing_params = QGroupBox("Bearing Parameters")
        form = QFormLayout()
        self.bearing_rpm = QDoubleSpinBox()
        self.bearing_rpm.setRange(0, 50000)
        self.bearing_rpm.setValue(1800)
        self.bearing_balls = QSpinBox()
        self.bearing_balls.setRange(3, 50)
        self.bearing_balls.setValue(8)
        self.bearing_ball_dia = QDoubleSpinBox()
        self.bearing_ball_dia.setRange(0.1, 100)
        self.bearing_ball_dia.setValue(12.5)
        self.bearing_pitch_dia = QDoubleSpinBox()
        self.bearing_pitch_dia.setRange(1, 500)
        self.bearing_pitch_dia.setValue(100)
        self.bearing_angle = QDoubleSpinBox()
        self.bearing_angle.setRange(0, 45)
        self.bearing_angle.setValue(0)
        form.addRow("Shaft RPM:", self.bearing_rpm)
        form.addRow("Number of Balls:", self.bearing_balls)
        form.addRow("Ball Diameter (mm):", self.bearing_ball_dia)
        form.addRow("Pitch Diameter (mm):", self.bearing_pitch_dia)
        form.addRow("Contact Angle (deg):", self.bearing_angle)
        bearing_params.setLayout(form)
        bearing_layout.addWidget(bearing_params)

        self.calc_freq_btn = QPushButton("Calculate Fault Frequencies")
        self.calc_freq_btn.clicked.connect(self.calculate_bearing_frequencies)
        bearing_layout.addWidget(self.calc_freq_btn)

        freq_display = QGroupBox("Characteristic Frequencies (Hz)")
        freq_form = QFormLayout()
        self.bpfi_label = QLabel("--")
        self.bpfo_label = QLabel("--")
        self.bsf_label = QLabel("--")
        self.ftf_label = QLabel("--")
        freq_form.addRow("BPFI:", self.bpfi_label)
        freq_form.addRow("BPFO:", self.bpfo_label)
        freq_form.addRow("BSF:", self.bsf_label)
        freq_form.addRow("FTF:", self.ftf_label)
        freq_display.setLayout(freq_form)
        bearing_layout.addWidget(freq_display)

        self.simulate_bearing_btn = QPushButton("Simulate Bearing Fault")
        self.simulate_bearing_btn.clicked.connect(self.simulate_bearing_fault)
        bearing_layout.addWidget(self.simulate_bearing_btn)
        tabs.addTab(bearing_tab, "Bearing Analysis")

        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.stats_table)
        self.compute_stats_btn = QPushButton("Compute Statistics")
        self.compute_stats_btn.clicked.connect(self.compute_statistics)
        stats_layout.addWidget(self.compute_stats_btn)
        tabs.addTab(stats_tab, "Statistics")

        wf_tab = QWidget()
        wf_layout = QVBoxLayout(wf_tab)
        wf_layout.addWidget(self.waterfall)
        self.wf_capture_btn = QPushButton("Capture Current PSD")
        self.wf_capture_btn.clicked.connect(self.capture_waterfall)
        wf_layout.addWidget(self.wf_capture_btn)
        tabs.addTab(wf_tab, "Waterfall")

        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        self.classify_btn = QPushButton("Run Diagnosis")
        self.ai_result = QLabel("Not classified")
        ai_layout.addWidget(self.classify_btn)
        ai_layout.addWidget(self.ai_result)
        self.llm_btn = QPushButton("Generate LLM Report")
        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)
        ai_layout.addWidget(self.llm_btn)
        ai_layout.addWidget(self.llm_output)
        self.pdf_btn = QPushButton("PDF Report")
        ai_layout.addWidget(self.pdf_btn)
        self.classify_btn.clicked.connect(self.run_ai)
        self.llm_btn.clicked.connect(self.run_llm)
        self.pdf_btn.clicked.connect(self.generate_pdf)
        tabs.addTab(ai_tab, "AI & Reports")

        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        self.machine_combo = QComboBox()
        self.machine_combo.addItem("None", None)
        self.refresh_machines()
        hist_layout.addWidget(QLabel("Machine:"))
        hist_layout.addWidget(self.machine_combo)
        self.hist_table = QTableWidget(0, 5)
        self.hist_table.setHorizontalHeaderLabels(["ID", "Timestamp", "Type", "Machine", "Results"])
        self.hist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_history)
        hist_layout.addWidget(refresh_btn)
        hist_layout.addWidget(self.hist_table)
        tabs.addTab(hist_tab, "History")

        tabs.addTab(self.calib, "Calibration")

    def toggle_mqtt(self):
        if not self.mqtt_enabled:
            broker = self.config.get('mqtt_broker', 'localhost')
            port = int(self.config.get('mqtt_port', '1883'))
            self.mqtt_client = MQTTClient(broker, port)
            if self.mqtt_client.connect():
                self.mqtt_enabled = True
                self.mqtt_btn.setText("MQTT On")
                self.log.log(f"MQTT connected to {broker}:{port}")
            else:
                QMessageBox.warning(self, "MQTT", "Failed to connect to broker")
        else:
            if self.mqtt_client:
                self.mqtt_client.disconnect()
            self.mqtt_enabled = False
            self.mqtt_btn.setText("MQTT Off")
            self.log.log("MQTT disconnected")

    def refresh_machines(self):
        if not self.db:
            return
        self.machine_combo.clear()
        self.machine_combo.addItem("None", None)
        for mid, name, _ in self.db.get_machines():
            self.machine_combo.addItem(name, mid)

    def start_acquisition(self):
        if hasattr(self, 'simulate_timer') and self.simulate_timer.isActive():
            self.simulate_timer.stop()
        port = self.port_combo.currentText()
        if port == "Simulated":
            self.simulate_timer = QTimer()
            self.simulate_timer.timeout.connect(self.simulate_data)
            self.simulate_timer.start(10)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            if self.log:
                self.log.log("Started simulated acquisition")
            return

        if not SERIAL_AVAILABLE:
            QMessageBox.warning(self, "Error", "PySerial not installed")
            return
        self.serial_reader = AsyncSerialReader(port)
        self.serial_reader.data_received.connect(self.on_serial_data)
        self.serial_reader.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        if self.log:
            self.log.log(f"Started serial on {port}")

    def simulate_data(self):
        if len(self.time_buffer) == 0:
            t = 0.0
        else:
            t = self.time_buffer[-1] + 1.0 / self.fs
        val = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.random.randn()
        self.on_serial_data(val)

    def check_alarms(self, rms, peak):
        if rms > self.alarm_rms:
            self.log.log(f"WARNING: High RMS vibration detected: {rms:.4f} > {self.alarm_rms}")
            QApplication.beep()
        if peak > self.alarm_peak:
            self.log.log(f"CRITICAL: Peak vibration exceeds limit: {peak:.4f} > {self.alarm_peak}")
            QApplication.beep()

    def on_serial_data(self, raw_val):
        cal_val = self.calib.calibrate(raw_val)
        t_now = len(self.time_buffer) / self.fs if self.time_buffer else 0
        self.time_buffer.append(t_now)
        self.data_buffer.append(cal_val)
        if len(self.time_buffer) > 2000:
            self.time_buffer = self.time_buffer[-2000:]
            self.data_buffer = self.data_buffer[-2000:]
        self.plot_update_counter += 1
        if self.plot_update_counter % 5 == 0 or len(self.time_buffer) < 100:
            self.live_curve.setData(self.time_buffer[-500:], self.data_buffer[-500:])
        if len(self.data_buffer) > 0 and len(self.data_buffer) % 10 == 0:
            data = np.array(self.data_buffer)
            stats = SignalAnalytics.compute_statistics(data)
            rms = stats['RMS']
            peak = stats['Peak']
            kurtosis = stats['Kurtosis']
            crest = stats['Crest Factor']
            self.rms_label.setText(f"{rms:.4f}")
            self.peak_label.setText(f"{peak:.4f}")
            self.kurtosis_label.setText(f"{kurtosis:.2f}")
            self.crest_label.setText(f"{crest:.2f}")
            self.check_alarms(rms, peak)
            if len(data) > 10:
                n = len(data)
                yf = fft(data)
                xf = fftfreq(n, 1/self.fs)
                pos = xf > 0
                mag = np.abs(yf[pos])
                idx = np.argmax(mag)
                dom_freq = xf[pos][idx]
                self.dom_freq_label.setText(f"{dom_freq:.1f}")
                self.fft_curve.setData(xf[pos][:n//2], mag[:n//2])
                if self.mqtt_enabled and self.mqtt_client:
                    mqtt_data = {
                        'timestamp': time.time(),
                        'rms': float(rms),
                        'peak': float(peak),
                        'dom_freq': float(dom_freq),
                        'kurtosis': float(kurtosis)
                    }
                    self.mqtt_client.publish(mqtt_data)

    def stop_acquisition(self):
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader = None
        if hasattr(self, 'simulate_timer') and self.simulate_timer.isActive():
            self.simulate_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.time_buffer.clear()
        self.data_buffer.clear()
        self.current_filtered = None
        if self.log:
            self.log.log("Acquisition stopped")

    def get_current_data(self):
        if self.current_filtered is not None:
            return np.array(self.current_filtered)
        elif len(self.data_buffer) > 0:
            return np.array(self.data_buffer)
        return None

    def set_manual_data(self, time, data):
        if len(time) != len(data) or len(time) < 2:
            return
        self.time_buffer = list(time)
        self.data_buffer = list(data)
        if len(time) > 1:
            dt_values = np.diff(time)
            if np.allclose(dt_values, dt_values[0], rtol=1e-5):
                self.fs = 1.0 / dt_values[0]
            else:
                self.fs = 1.0 / np.mean(dt_values)
                if self.log:
                    self.log.log(f"Non-uniform sampling, using mean fs={self.fs:.1f}Hz")
        else:
            self.fs = 1000.0
        self.live_curve.setData(self.time_buffer, self.data_buffer)
        if len(self.data_buffer) > 0:
            data_np = np.array(self.data_buffer)
            stats = SignalAnalytics.compute_statistics(data_np)
            self.rms_label.setText(f"{stats['RMS']:.4f}")
            self.peak_label.setText(f"{stats['Peak']:.4f}")
            self.kurtosis_label.setText(f"{stats['Kurtosis']:.2f}")
            self.crest_label.setText(f"{stats['Crest Factor']:.2f}")
            if len(data_np) > 10:
                n = len(data_np)
                yf = fft(data_np)
                xf = fftfreq(n, 1/self.fs)
                pos = xf > 0
                mag = np.abs(yf[pos])
                idx = np.argmax(mag)
                dom_freq = xf[pos][idx]
                self.dom_freq_label.setText(f"{dom_freq:.1f}")
                self.fft_curve.setData(xf[pos][:n//2], mag[:n//2])

    def apply_filter(self):
        data = self.get_current_data()
        if data is None or len(data) < 10:
            QMessageBox.warning(self, "No Data", "Capture data first.")
            return
        ftype = self.filter_type.currentText()
        cutoff1 = float(self.filter_cutoff.text())
        order = self.filter_order.value()
        nyq = 0.5 * self.fs
        if ftype == 'bandpass':
            cutoff2 = float(self.filter_cutoff2.text())
            normal_cutoff = [cutoff1 / nyq, cutoff2 / nyq]
            b, a = butter(order, normal_cutoff, btype='band')
        else:
            normal_cutoff = cutoff1 / nyq
            b, a = butter(order, normal_cutoff, btype=ftype)
        filtered = filtfilt(b, a, data)
        self.current_filtered = filtered
        self.adv_ax.clear()
        self.adv_ax.plot(self.time_buffer[-len(data):], filtered, 'g-')
        self.adv_ax.set_title("Filtered Signal")
        self.adv_canvas.draw()
        if self.log:
            self.log.log(f"Applied {ftype} filter cutoff={cutoff1}Hz order={order}")

    def compute_cwt_parallel(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "No Data", "No signal.")
            return
        if not WAVELET_AVAILABLE:
            QMessageBox.warning(self, "CWT", "PyWavelets not installed.")
            return
        scales = np.arange(1, 129)
        progress = QProgressDialog("Computing CWT...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        progress.setValue(10)
        if self.log:
            self.log.log("Starting CWT in separate process...")
        try:
            with mp.Pool(1) as pool:
                result = pool.apply_async(cwt_process, (data, scales))
                progress.setValue(50)
                coeffs = result.get(timeout=10)
                progress.setValue(100)
                if coeffs is None:
                    QMessageBox.warning(self, "CWT", "Processing failed.")
                    return
                self.adv_ax.clear()
                im = self.adv_ax.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
                self.adv_ax.set_xlabel("Time")
                self.adv_ax.set_ylabel("Scale")
                self.adv_fig.colorbar(im, ax=self.adv_ax)
                self.adv_ax.set_title("CWT (Parallel)")
                self.adv_canvas.draw()
                if self.log:
                    self.log.log("CWT completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"CWT failed: {e}")
        finally:
            progress.close()

    def compute_psd(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            return
        f, Pxx = welch(data, self.fs, nperseg=min(256, len(data)))
        self.adv_ax.clear()
        self.adv_ax.semilogy(f, Pxx, 'purple')
        self.adv_ax.set_xlabel("Frequency (Hz)")
        self.adv_ax.set_ylabel("PSD")
        self.adv_ax.set_title("Power Spectral Density")
        self.adv_ax.grid(True)
        self.adv_canvas.draw()

    def compute_spectrogram(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            return
        f, t, Sxx = spectrogram(data, self.fs, nperseg=256, noverlap=128)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        self.adv_ax.clear()
        self.adv_ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
        self.adv_ax.set_ylabel("Frequency (Hz)")
        self.adv_ax.set_xlabel("Time (s)")
        self.adv_ax.set_title("Spectrogram")
        self.adv_canvas.draw()

    def compute_envelope_spectrum(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            return
        f, env_psd = SignalAnalytics.envelope_spectrum(data, self.fs)
        self.adv_ax.clear()
        self.adv_ax.semilogy(f, env_psd, 'orange')
        self.adv_ax.set_xlabel("Frequency (Hz)")
        self.adv_ax.set_ylabel("Envelope PSD")
        self.adv_ax.set_title("Envelope Spectrum (for bearing faults)")
        self.adv_ax.grid(True)
        self.adv_canvas.draw()

    def capture_waterfall(self):
        data = self.get_current_data()
        if data is None or len(data) < 256:
            QMessageBox.warning(self, "No Data", "Need at least 256 points for PSD")
            return
        f, Pxx = welch(data, self.fs, nperseg=min(256, len(data)))
        self.waterfall.add_spectrum(f, Pxx)
        if self.log:
            self.log.log("Captured PSD for waterfall")

    def calculate_bearing_frequencies(self):
        rpm = self.bearing_rpm.value()
        n = self.bearing_balls.value()
        bd = self.bearing_ball_dia.value() / 1000.0
        pd = self.bearing_pitch_dia.value() / 1000.0
        angle = self.bearing_angle.value()
        bpfi, bpfo, bsf, ftf = SignalAnalytics.bearing_characteristic_frequencies(rpm, n, bd, pd, angle)
        self.bpfi_label.setText(f"{bpfi:.2f}")
        self.bpfo_label.setText(f"{bpfo:.2f}")
        self.bsf_label.setText(f"{bsf:.2f}")
        self.ftf_label.setText(f"{ftf:.2f}")

    def simulate_bearing_fault(self):
        rpm = self.bearing_rpm.value()
        n = self.bearing_balls.value()
        bd = self.bearing_ball_dia.value() / 1000.0
        pd = self.bearing_pitch_dia.value() / 1000.0
        angle = self.bearing_angle.value()
        bpfi, bpfo, bsf, ftf = SignalAnalytics.bearing_characteristic_frequencies(rpm, n, bd, pd, angle)
        fault_type = QMessageBox.question(self, "Fault Type", "Select fault type:\nYes=Outer Race, No=Inner Race, Cancel=Ball",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if fault_type == QMessageBox.Yes:
            fault = 'outer'
            freq = bpfo
        elif fault_type == QMessageBox.No:
            fault = 'inner'
            freq = bpfi
        else:
            fault = 'ball'
            freq = bsf
        duration = 5.0
        fs = 5000
        t, signal = BearingFaultSimulator.generate_bearing_signal(fs, duration, rpm, bpfi, bpfo, bsf, ftf, fault)
        self.set_manual_data(t, signal)
        self.log.log(f"Generated {fault} race bearing fault signal with fault frequency {freq:.2f} Hz")

    def compute_statistics(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "No Data", "No signal data")
            return
        stats = SignalAnalytics.compute_statistics(data)
        self.stats_table.setRowCount(len(stats))
        for i, (key, value) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(key))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

    def run_ai(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "No Data", "Acquire data first.")
            return
        diagnosis, conf = self.ai_model.predict(data)
        self.ai_result.setText(f"Diagnosis: {diagnosis} (conf {conf:.2f})")
        if self.log:
            self.log.log(f"AI diagnosis: {diagnosis}")

    def run_llm(self):
        metrics = {"rms": self.rms_label.text(), "peak": self.peak_label.text(), "dom_freq": self.dom_freq_label.text()}
        diagnosis = self.ai_result.text()
        machine_id = self.machine_combo.currentData()
        machine_name = self.machine_combo.currentText() if machine_id else "Unknown"
        text = self.ollama.generate_report(metrics, diagnosis, machine_name)
        self.llm_output.setText(text)
        if self.log:
            self.log.log("LLM report generated")

    def generate_pdf(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "No Data", "No data to report.")
            return
        test_info = {
            "type": "Vibration",
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {
                "RMS": self.rms_label.text(),
                "Peak": self.peak_label.text(),
                "Dominant Freq": self.dom_freq_label.text(),
                "Kurtosis": self.kurtosis_label.text(),
                "Crest Factor": self.crest_label.text(),
                "AI Diagnosis": self.ai_result.text()
            }
        }
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.plot(self.time_buffer, self.data_buffer, 'b-')
        ax.set_title("Signal")
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        if REPORTLAB_AVAILABLE:
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            y = height - 50
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "Vibration Lab Pro Report")
            y -= 30
            c.setFont("Helvetica", 12)
            for k, v in test_info["metrics"].items():
                c.drawString(50, y, f"{k}: {v}")
                y -= 20
            fig_path = "/tmp/plot.png"
            fig.savefig(fig_path)
            c.drawImage(ImageReader(fig_path), 50, y-200, width=500, height=180)
            c.save()
            QMessageBox.information(self, "Report", f"Saved as {filename}")
            if self.log:
                self.log.log(f"PDF report saved: {filename}")
        else:
            QMessageBox.warning(self, "Report", "ReportLab not installed.")

    def load_history(self):
        if not self.db:
            return
        machine_id = self.machine_combo.currentData()
        rows = self.db.get_tests(test_type="Vibration", machine_id=machine_id)
        self.hist_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row[:5]):
                self.hist_table.setItem(i, j, QTableWidgetItem(str(val)))

    def export_data(self):
        data = self.get_current_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", 
            "CSV (*.csv);;NPZ (*.npy);;HDF5 (*.h5);;Parquet (*.parquet)")
        if not path:
            return
        try:
            if path.endswith('.csv'):
                np.savetxt(path, np.column_stack([self.time_buffer, data]), 
                          delimiter=',', header='Time(s),Amplitude', comments='')
            elif path.endswith('.npy'):
                np.savez(path, time=self.time_buffer, data=data)
            elif path.endswith('.h5') and H5PY_AVAILABLE:
                with h5py.File(path, 'w') as f:
                    f.create_dataset('time', data=self.time_buffer)
                    f.create_dataset('data', data=data)
            elif path.endswith('.parquet') and PARQUET_AVAILABLE:
                df = pd.DataFrame({'Time': self.time_buffer, 'Amplitude': data})
                df.to_parquet(path)
            else:
                if H5PY_AVAILABLE and not path.endswith('.h5'):
                    path += '.h5'
                    with h5py.File(path, 'w') as f:
                        f.create_dataset('time', data=self.time_buffer)
                        f.create_dataset('data', data=data)
                else:
                    path += '.csv'
                    np.savetxt(path, np.column_stack([self.time_buffer, data]), 
                              delimiter=',', header='Time(s),Amplitude', comments='')
            self.log.log(f"Data exported to {path}")
            QMessageBox.information(self, "Export", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

class OllamaEngineer:
    def __init__(self, model="llama3.2", url="http://localhost:11434"):
        self.model = model
        self.url = url

    def generate_report(self, metrics, diagnosis, machine_info=""):
        if not OLLAMA_AVAILABLE:
            return "Ollama not available. Install requests and run 'ollama serve'."
        prompt = f"""You are a vibration analysis expert following ISO 20816 standards.
Machine: {machine_info}
Metrics: RMS={metrics.get('rms')}, Peak={metrics.get('peak')}, Dominant Frequency={metrics.get('dom_freq')} Hz
AI Diagnosis: {diagnosis}
Provide a concise technical assessment, possible root cause, and recommended action (max 5 lines)."""
        try:
            resp = requests.post(f"{self.url}/api/generate",
                                 json={"model": self.model, "prompt": prompt, "stream": False},
                                 timeout=30)
            if resp.status_code == 200:
                return resp.json().get('response', 'No response')
        except:
            return "Ollama request failed."
        return "Failed."

class HomeDashboard(QWidget):
    def __init__(self, parent=None, road_mod=None, vibe_mod=None, seismic_mod=None):
        super().__init__(parent)
        self.road_mod = road_mod
        self.vibe_mod = vibe_mod
        self.seismic_mod = seismic_mod
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("MANUAL DATA ENTRY")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00FFFF;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        s_layout = QVBoxLayout(sw)
        scroll.setWidget(sw)
        layout.addWidget(scroll)

        road_grp = QGroupBox("Road (Dist m, Elev mm)")
        road_layout = QVBoxLayout()
        self.road_table = QTableWidget(0, 2)
        self.road_table.setHorizontalHeaderLabels(["Distance (m)", "Elevation (mm)"])
        self.road_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        btn_row = QHBoxLayout()
        add_road = QPushButton("Add Row")
        del_road = QPushButton("Delete")
        analyze_road = QPushButton("Analyze")
        add_road.clicked.connect(lambda: self.add_row(self.road_table))
        del_road.clicked.connect(lambda: self.del_row(self.road_table))
        analyze_road.clicked.connect(self.analyze_road)
        btn_row.addWidget(add_road)
        btn_row.addWidget(del_road)
        btn_row.addStretch()
        btn_row.addWidget(analyze_road)
        road_layout.addWidget(self.road_table)
        road_layout.addLayout(btn_row)
        road_grp.setLayout(road_layout)
        s_layout.addWidget(road_grp)
        self.add_row(self.road_table, [0,0])
        self.add_row(self.road_table, [10,15])
        self.add_row(self.road_table, [20,8])

        vibe_grp = QGroupBox("Vibration (Time s, Amplitude)")
        vibe_layout = QVBoxLayout()
        self.vibe_table = QTableWidget(0, 2)
        self.vibe_table.setHorizontalHeaderLabels(["Time (s)", "Amplitude"])
        self.vibe_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        btn_row2 = QHBoxLayout()
        add_vibe = QPushButton("Add Row")
        del_vibe = QPushButton("Delete")
        analyze_vibe = QPushButton("Analyze")
        add_vibe.clicked.connect(lambda: self.add_row(self.vibe_table))
        del_vibe.clicked.connect(lambda: self.del_row(self.vibe_table))
        analyze_vibe.clicked.connect(self.analyze_vibe)
        btn_row2.addWidget(add_vibe)
        btn_row2.addWidget(del_vibe)
        btn_row2.addStretch()
        btn_row2.addWidget(analyze_vibe)
        vibe_layout.addWidget(self.vibe_table)
        vibe_layout.addLayout(btn_row2)
        vibe_grp.setLayout(vibe_layout)
        s_layout.addWidget(vibe_grp)
        self.add_row(self.vibe_table, [0,0])
        self.add_row(self.vibe_table, [0.01,0.5])
        self.add_row(self.vibe_table, [0.02,-0.3])

        seis_grp = QGroupBox("Seismic (Time s, Accel g)")
        seis_layout = QVBoxLayout()
        self.seis_table = QTableWidget(0, 2)
        self.seis_table.setHorizontalHeaderLabels(["Time (s)", "Acceleration (g)"])
        self.seis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        btn_row3 = QHBoxLayout()
        add_seis = QPushButton("Add Row")
        del_seis = QPushButton("Delete")
        analyze_seis = QPushButton("Analyze")
        add_seis.clicked.connect(lambda: self.add_row(self.seis_table))
        del_seis.clicked.connect(lambda: self.del_row(self.seis_table))
        analyze_seis.clicked.connect(self.analyze_seis)
        btn_row3.addWidget(add_seis)
        btn_row3.addWidget(del_seis)
        btn_row3.addStretch()
        btn_row3.addWidget(analyze_seis)
        seis_layout.addWidget(self.seis_table)
        seis_layout.addLayout(btn_row3)
        seis_grp.setLayout(seis_layout)
        s_layout.addWidget(seis_grp)
        self.add_row(self.seis_table, [0,0])
        self.add_row(self.seis_table, [0.1,0.2])
        self.add_row(self.seis_table, [0.2,-0.1])

    def add_row(self, table, values=None):
        row = table.rowCount()
        table.insertRow(row)
        if values is None:
            values = ["0","0"]
        for col, val in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(str(val)))

    def del_row(self, table):
        cur = table.currentRow()
        if cur >= 0:
            table.removeRow(cur)

    def get_table_data(self, table):
        rows = table.rowCount()
        cols = table.columnCount()
        if rows == 0:
            return [], []
        data = []
        for r in range(rows):
            row_data = []
            for c in range(cols):
                item = table.item(r, c)
                if item is None:
                    row_data.append(np.nan)
                else:
                    try:
                        row_data.append(float(item.text()))
                    except:
                        row_data.append(np.nan)
            data.append(row_data)
        data = np.array(data)
        if data.shape[1] >= 2:
            return data[:,0], data[:,1]
        return [], []

    def analyze_road(self):
        dist, elev = self.get_table_data(self.road_table)
        if len(dist) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 points for road analysis")
            return
        if self.road_mod:
            self.road_mod.set_data(dist, elev)
            main_window = self.window()
            if hasattr(main_window, 'stack'):
                main_window.stack.setCurrentIndex(1)

    def analyze_vibe(self):
        t, amp = self.get_table_data(self.vibe_table)
        if len(t) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 points for vibration analysis")
            return
        if self.vibe_mod:
            self.vibe_mod.set_manual_data(t, amp)
            main_window = self.window()
            if hasattr(main_window, 'stack'):
                main_window.stack.setCurrentIndex(2)

    def analyze_seis(self):
        t, acc = self.get_table_data(self.seis_table)
        if len(t) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 points for seismic analysis")
            return
        if self.seismic_mod:
            self.seismic_mod.set_data(t, acc)
            main_window = self.window()
            if hasattr(main_window, 'stack'):
                main_window.stack.setCurrentIndex(3)

class HelpModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("HELP")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FFFF;")
        layout.addWidget(title)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("""
Vibration Lab Pro - Engineering Suite

NEW FEATURES:
- Real-time FFT display alongside time waveform
- Bearing fault analysis with envelope spectrum
- Bearing characteristic frequency calculator (BPFI, BPFO, BSF, FTF)
- Bearing fault signal simulation
- Statistical features: Kurtosis, Crest Factor, Skewness, Clearance Factor
- MQTT remote monitoring (publish data to broker)
- HDF5/Parquet export for large datasets
- Bandpass filter option
- Enhanced AI diagnosis using kurtosis

Modules:
- HOME: Manual data entry tables
- ROAD: IRI from CSV (distance, elevation)
- VIBRATION: Real-time acquisition, FFT, filters, CWT, PSD, spectrogram, envelope spectrum, bearing analysis, statistics, AI, MQTT
- SEISMIC: PGA/PGV/PSD from CSV or simulation
- CANTILEVER: Beam deflection/stress/freq with unit selection
- HELP: This page

Keyboard shortcuts:
- Ctrl+S: Export current data
- Space: Start/stop acquisition

MQTT Setup:
- Configure broker in vibration_lab.ini
- Click MQTT button to connect
- Data published to topic 'vibration/data' as JSON

Optional libraries for full features:
- PyWavelets: CWT
- PyTorch/TensorFlow: advanced AI
- ReportLab: PDF export
- paho-mqtt: MQTT remote monitoring
- h5py, pandas: HDF5/Parquet export
        """)
        layout.addWidget(text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900)
        self.is_dark = True
        self.db = DatabaseManager()
        self.log_panel = LogPanel()
        self.config = Config()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(0)
        self.sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(20, 80, 20, 20)
        side_layout.setSpacing(15)
        menu = ["🏠 HOME", "🛣️ ROAD", "⚙️ VIBRATION", "🌍 SEISMIC", "📐 CANTILEVER", "❓ HELP"]
        for i, name in enumerate(menu):
            btn = QPushButton(name)
            btn.setFixedHeight(45)
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))
            side_layout.addWidget(btn)
        side_layout.addStretch()
        root_layout.addWidget(self.sidebar)

        self.content = QWidget()
        self.main_layout = QVBoxLayout(self.content)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(40, 30, 40, 10)
        self.title_label = QLabel("VIBRATION LAB PRO")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.theme_btn = QPushButton("DARK")
        self.theme_btn.setFixedSize(80, 50)
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.menu_btn = QPushButton("≡")
        self.menu_btn.setFixedSize(50, 50)
        self.menu_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.theme_btn)
        header_layout.addWidget(self.menu_btn)
        self.main_layout.addWidget(header)

        splitter = QSplitter(Qt.Vertical)
        self.stack = QStackedWidget()
        self.road_mod = RoadRoughnessModule(db=self.db, log=self.log_panel)
        self.vibe_mod = VibrationModule(db=self.db, log=self.log_panel)
        self.seismic_mod = SeismicModule(db=self.db, log=self.log_panel)
        self.cantilever_mod = CantileverBeamModule()
        self.help_mod = HelpModule()
        self.home = HomeDashboard(self, self.road_mod, self.vibe_mod, self.seismic_mod)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.road_mod)
        self.stack.addWidget(self.vibe_mod)
        self.stack.addWidget(self.seismic_mod)
        self.stack.addWidget(self.cantilever_mod)
        self.stack.addWidget(self.help_mod)
        splitter.addWidget(self.stack)
        splitter.addWidget(self.log_panel)
        splitter.setSizes([700, 200])
        self.main_layout.addWidget(splitter)
        root_layout.addWidget(self.content)

        self.animation = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.animation.setDuration(400)
        self.animation.setEasingCurve(QEasingCurve.InOutQuint)

        self.apply_theme()
        self.log_panel.log("Application started")
        self.setup_shortcuts()

    def setup_shortcuts(self):
        save_shortcut = QShortcut("Ctrl+S", self)
        save_shortcut.activated.connect(self.save_current_data)
        space_shortcut = QShortcut("Space", self)
        space_shortcut.activated.connect(self.toggle_acquisition)

    def save_current_data(self):
        if self.stack.currentWidget() == self.vibe_mod:
            self.vibe_mod.export_data()
        else:
            QMessageBox.information(self, "Info", "Switch to Vibration module to export data (Ctrl+S)")

    def toggle_acquisition(self):
        if self.stack.currentWidget() == self.vibe_mod:
            if self.vibe_mod.start_btn.isEnabled():
                self.vibe_mod.start_acquisition()
            else:
                self.vibe_mod.stop_acquisition()
        else:
            QMessageBox.information(self, "Info", "Switch to Vibration module to start/stop (Space)")

    def switch_view(self, index):
        self.stack.setCurrentIndex(index)
        if self.sidebar.width() > 0:
            self.toggle_sidebar()

    def toggle_sidebar(self):
        cur = self.sidebar.width()
        target = 260 if cur == 0 else 0
        self.animation.setStartValue(cur)
        self.animation.setEndValue(target)
        self.animation.start()

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.theme_btn.setText("DARK" if self.is_dark else "LIGHT")
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark:
            accent = "#00FFFF"
            bg = "#000000"
            side_bg = "#111111"
            text = "#00FFFF"
            table_bg = "#111111"
            table_text = "#FFFFFF"
            input_bg = "#222222"
            input_text = "#FFFFFF"
            label_text = "#FFFFFF"
        else:
            accent = "#0055FF"
            bg = "#FFFFFF"
            side_bg = "#F0F0F0"
            text = "#0055FF"
            table_bg = "#FFFFFF"
            table_text = "#000000"
            input_bg = "#FFFFFF"
            input_text = "#000000"
            label_text = "#000000"

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {bg}; }}
            QWidget {{ background-color: {bg}; color: {label_text}; }}
            QLabel {{ color: {label_text}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color: {input_bg}; color: {input_text}; border: 1px solid {accent}; padding: 2px; }}
            QTableWidget {{ background-color: {table_bg}; color: {table_text}; gridline-color: gray; }}
            QHeaderView::section {{ background-color: {side_bg}; color: {text}; }}
            QPushButton {{ background: transparent; color: {text}; border: 1px solid transparent; font-weight: bold; text-align: left; padding-left: 10px; }}
            QPushButton:hover {{ border: 1px solid {accent}; }}
            QGroupBox {{ border: 1px solid {accent}; margin-top: 10px; color: {text}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
            QTabWidget::pane {{ border: 1px solid {accent}; }}
            QTabBar::tab {{ background-color: {side_bg}; color: {text}; padding: 8px; }}
            QTabBar::tab:selected {{ background-color: {accent}; color: {bg}; }}
        """)
        self.sidebar.setStyleSheet(f"QFrame#sidebar {{ background-color: {side_bg}; border-right: 1px solid {accent}; }}")
        btn_style = f"background-color: {side_bg}; border: 1px solid {accent}; border-radius: 12px; color: {text}; font-weight: bold;"
        self.theme_btn.setStyleSheet(btn_style)
        self.menu_btn.setStyleSheet(btn_style + "font-size: 22px;")
        self.update_title_style(accent)

    def update_title_style(self, accent):
        size = max(20, min(int(self.width() / 25), 40))
        grad = "#000055" if self.is_dark else "#0022AA"
        self.title_label.setStyleSheet(
            f"font-family: 'Segoe UI'; font-size: {size}px; font-weight: 900; color: white; padding: 15px 40px; border-radius: 12px; "
            f"background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {grad}, stop:1 {accent}); border: 2px solid {accent};"
        )

    def resizeEvent(self, event):
        self.update_title_style("#00FFFF" if self.is_dark else "#0055FF")
        super().resizeEvent(event)

if __name__ == '__main__':
    mp.freeze_support()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())