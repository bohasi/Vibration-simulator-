#!/usr/bin/env python3

import sys
import os
import json
import sqlite3
import datetime
import time
import threading
import numpy as np
import configparser
from scipy.signal import welch, butter, filtfilt, spectrogram, hilbert
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

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