import cv2
import numpy as np
import random
import time
import threading
import subprocess
import os
import argparse


class DoomscrollDetector:
    def __init__(self, video_path="rickroll.mp4", roast_cooldown=3, detection_threshold=1, sensitivity=0.55):
        # Try MediaPipe first, then dlib, then OpenCV Haar Cascades
        self.use_mediapipe = False
        self.use_dlib = False

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request

            model_path = "face_landmarker.task"
            if not os.path.exists(model_path):
                print("Downloading MediaPipe face landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded.")

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.7,
                min_face_presence_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self.use_mediapipe = True
            print("Using MediaPipe Face Mesh for face tracking")
        except ImportError:
            try:
                import dlib
                self.use_dlib = True
                self.detector = dlib.get_frontal_face_detector()
                # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                print("Using dlib for face tracking")
            except ImportError:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'
                )
                print("Using OpenCV Haar Cascades for face tracking")

        # Roasting messages
        self.roasts = [
            "You'll fail if you don't stop!",
            "Your dreams called - they want your attention back!",
            "Scrolling won't make that deadline disappear!",
            "The phone can wait. Your future can't.",
            "Success doesn't scroll itself into existence!",
            "That screen won't study for you!",
            "Your goals > Your feed. Remember that.",
            "Your parents do not love you",
            "Future you is watching. They're disappointed.",
            "Every scroll is a step backward. Look up!",
            "The algorithm wins again. Pathetic.",
            "You will be alone forever",
            "Is this really more important than your goals?",
            "Your productivity just left the chat.",
            "Doomscrolling detected! You're better than this!",
            "PUT. THE. PHONE. DOWN. NOW.",
            "You re such a disappointment to your family",
            "This is why you're behind schedule."
        ]

        self.last_roast_time = 0
        self.roast_cooldown = roast_cooldown
        self.current_roast = ""
        self.sensitivity = sensitivity  # pitch ratio threshold (lower = more sensitive)

        # Rickroll video
        self.rickroll_path = video_path
        self.rickroll_process = None
        self.is_rickrolling = False

        # Detection state tracking
        self.doomscroll_count = 0
        self.normal_count = 0
        self.detection_threshold = detection_threshold  # frames to trigger doomscroll (fast)
        self.recovery_threshold = 20  # frames of face detection needed to stop music (slow)

        # Performance: skip frames to reduce CPU load
        self.frame_count = 0
        self.last_raw_detection = False

    def _update_roast(self):
        """Pick a new roast message if cooldown has passed."""
        current_time = time.time()
        if current_time - self.last_roast_time > self.roast_cooldown:
            self.current_roast = random.choice(self.roasts)
            self.last_roast_time = current_time

    def detect_doomscroll_mediapipe(self, frame):
        """Detect doomscrolling using MediaPipe iris tracking.
        Uses iris position within the eye to determine gaze direction.
        Falls back to face presence if iris landmarks are unavailable.
        """
        h, w = frame.shape[:2]

        # Resize to 320x240 for faster MediaPipe processing
        small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return True  # No face detected = looking away = doomscrolling

        landmarks = result.face_landmarks[0]

        # Iris landmarks require 478 points (indices 468-477)
        # Left iris center: 468 | Right iris center: 473
        # Left eye:  top=159, bottom=145
        # Right eye: top=386, bottom=374
        if len(landmarks) < 478:
            return False  # No iris data, assume OK

        left_iris_y   = landmarks[468].y
        left_top_y    = landmarks[159].y
        left_bottom_y = landmarks[145].y
        left_gaze     = (left_iris_y - left_top_y) / (abs(left_bottom_y - left_top_y) + 1e-6)

        right_iris_y   = landmarks[473].y
        right_top_y    = landmarks[386].y
        right_bottom_y = landmarks[374].y
        right_gaze     = (right_iris_y - right_top_y) / (abs(right_bottom_y - right_top_y) + 1e-6)

        avg_gaze = (left_gaze + right_gaze) / 2

        # Sanity check: discard garbage readings from false face detections (e.g. hair)
        if not (0.0 <= avg_gaze <= 2.0):
            return False

        # avg_gaze > 0.65 means iris is in lower portion of eye = looking down
        gaze_threshold = 0.65 - (self.sensitivity - 0.55) * 1.5
        is_looking_down = avg_gaze > gaze_threshold

        # Draw iris indicators on both eyes
        color = (0, 0, 255) if is_looking_down else (0, 255, 0)
        for idx in [468, 473]:
            pt = landmarks[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Debug: show gaze value so we can calibrate the threshold
        cv2.putText(frame, f"gaze: {avg_gaze:.2f} / thresh: {gaze_threshold:.2f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return is_looking_down

    def detect_doomscroll_dlib(self, frame, gray):
        """Detect doomscrolling using dlib landmarks."""
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            nose_tip       = (landmarks.part(30).x, landmarks.part(30).y)
            chin           = (landmarks.part(8).x,  landmarks.part(8).y)
            forehead_approx = (landmarks.part(27).x, landmarks.part(27).y)

            left_eye_points  = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_eye_top    = (left_eye_points[1][1] + left_eye_points[2][1]) / 2
            left_eye_bottom = (left_eye_points[4][1] + left_eye_points[5][1]) / 2
            left_eye_center = (left_eye_points[0][1] + left_eye_points[3][1]) / 2

            right_eye_top    = (right_eye_points[1][1] + right_eye_points[2][1]) / 2
            right_eye_bottom = (right_eye_points[4][1] + right_eye_points[5][1]) / 2
            right_eye_center = (right_eye_points[0][1] + right_eye_points[3][1]) / 2

            left_ratio  = abs(left_eye_center - left_eye_top)  / (abs(left_eye_bottom  - left_eye_top)  + 1e-6)
            right_ratio = abs(right_eye_center - right_eye_top) / (abs(right_eye_bottom - right_eye_top) + 1e-6)
            eye_ratio   = (left_ratio + right_ratio) / 2

            head_tilt = (chin[1] - nose_tip[1]) / (nose_tip[1] - forehead_approx[1] + 1e-6)
            is_looking_down = head_tilt > 1.3 or eye_ratio < 0.35

            cv2.circle(frame, nose_tip, 3, (0, 255, 0), -1)
            cv2.circle(frame, chin, 3, (255, 0, 0), -1)
            for pt in left_eye_points + right_eye_points:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

            return is_looking_down

        return False

    def detect_doomscroll_opencv(self, frame, gray):
        """Detect doomscrolling using OpenCV Haar Cascades."""
        scale = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        small_gray = cv2.equalizeHist(small_gray)

        faces = self.face_cascade.detectMultiScale(small_gray, 1.1, 3, minSize=(60, 60))

        for (sx, sy, sw, sh) in faces:
            x, y, w, h = int(sx/scale), int(sy/scale), int(sw/scale), int(sh/scale)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray  = gray[y:y+int(h*0.6), x:x+w]
            roi_color = frame[y:y+int(h*0.6), x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            detection_score = 0

            face_center_y      = y + h//2
            face_position_ratio = face_center_y / frame.shape[0]

            if face_position_ratio > self.sensitivity + 0.03:
                detection_score += 2
            elif face_position_ratio > self.sensitivity:
                detection_score += 1

            if h / w < 1.1:
                detection_score += 1

            if len(eyes) >= 2:
                eye_y_positions    = [y + ey + eh//2 for (ex, ey, ew, eh) in eyes]
                avg_eye_y          = sum(eye_y_positions) / len(eye_y_positions)
                eye_position_in_face = (avg_eye_y - y) / h

                if eye_position_in_face > 0.6:
                    detection_score += 2
                elif eye_position_in_face > 0.52:
                    detection_score += 1

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                detection_score += 1

            return detection_score >= 3

        return False

    def play_rickroll(self):
        """Play rickroll video (only if not already playing)."""
        if self.is_rickrolling or not os.path.exists(self.rickroll_path):
            return

        self.is_rickrolling = True
        video_path = os.path.abspath(self.rickroll_path)

        def start_video():
            if os.name == 'nt':  # Windows
                # Try known players directly so we get a trackable process handle
                win_players = [
                    ['vlc', '--play-and-exit', video_path],
                    [r'C:\Program Files\VideoLAN\VLC\vlc.exe', '--play-and-exit', video_path],
                    [r'C:\Program Files\Windows Media Player\wmplayer.exe', video_path],
                    [r'C:\Program Files (x86)\Windows Media Player\wmplayer.exe', video_path],
                ]
                for cmd in win_players:
                    try:
                        self.rickroll_process = subprocess.Popen(
                            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        return
                    except (FileNotFoundError, OSError):
                        continue
                # Fallback: open with default player (no direct handle)
                os.startfile(video_path)
            elif os.uname().sysname == 'Darwin':  # macOS
                escaped_path = video_path.replace('"', '\\"')
                self.rickroll_process = subprocess.Popen(
                    ['osascript',
                     '-e', f'tell application "QuickTime Player" to open POSIX file "{escaped_path}"',
                     '-e', 'tell application "QuickTime Player" to play front document'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux
                try:
                    self.rickroll_process = subprocess.Popen(
                        ['vlc', '--play-and-exit', video_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                    self.rickroll_process = subprocess.Popen(['xdg-open', video_path])

        threading.Thread(target=start_video, daemon=True).start()

    def stop_rickroll(self):
        """Stop rickroll video (runs in background to avoid blocking the main loop)."""
        if not self.is_rickrolling:
            return

        self.is_rickrolling = False
        process_snapshot = self.rickroll_process
        self.rickroll_process = None
        threading.Thread(target=self._kill_video, args=(process_snapshot,), daemon=True).start()

    def _kill_video(self, process):
        """Actually kill the video player process (called from background thread)."""
        if process:
            try:
                if os.name == 'nt':  # Windows
                    # 1. Kill process tree by PID
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    # 2. Kill common video players by name (cmd /c spawns orphan processes)
                    for player in ['wmplayer', 'vlc', 'Video.UI', 'MPC-HC',
                                   'PotPlayerMini64', 'PotPlayerMini', 'mpv']:
                        subprocess.run(
                            ['taskkill', '/F', '/IM', f'{player}.exe'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                    # 3. PowerShell fallback: kill anything with the video filename in window title
                    video_name = os.path.splitext(os.path.basename(self.rickroll_path))[0]
                    escaped = video_name.replace("'", "''")
                    subprocess.run(
                        ['powershell', '-Command',
                         f"Get-Process | Where-Object {{$_.MainWindowTitle -like '*{escaped}*'}} | Stop-Process -Force"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                elif os.uname().sysname == 'Darwin':  # macOS
                    subprocess.run(['killall', 'QuickTime Player'], stderr=subprocess.DEVNULL)
                process.terminate()
            except Exception:
                pass

    def show_roast(self, frame):
        """Display roasting message on frame."""
        self._update_roast()

        overlay = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "DOOMSCROLLING DETECTED!", (w//2 - 250, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, self.current_roast, (w//2 - 300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        """Main loop."""
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Doomscrolling Blocker Started!")
        print("Looking for your face...")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            frame = cv2.flip(frame, 1)

            # Detect doomscrolling every 3 frames, reuse last result in between
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                if self.use_mediapipe:
                    self.last_raw_detection = self.detect_doomscroll_mediapipe(frame)
                elif self.use_dlib:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.last_raw_detection = self.detect_doomscroll_dlib(frame, gray)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.last_raw_detection = self.detect_doomscroll_opencv(frame, gray)
            raw_detection = self.last_raw_detection

            # Stabilize detection with frame counting to avoid flickering
            if raw_detection:
                self.doomscroll_count += 1
                self.normal_count = 0
            else:
                self.normal_count += 1
                self.doomscroll_count = 0

            # Trigger fast, recover slow — prevents music from looping on brief face detections
            is_doomscrolling = self.doomscroll_count >= self.detection_threshold
            is_normal        = self.normal_count >= self.recovery_threshold

            if is_doomscrolling:
                self.show_roast(frame)
                self.play_rickroll()
            elif is_normal:
                cv2.putText(frame, "Good posture! Keep it up!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.stop_rickroll()
            else:
                cv2.putText(frame, "Monitoring...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Doomscrolling Blocker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_rickroll()
        if self.use_mediapipe:
            self.face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doomscrolling Blocker')
    parser.add_argument('--video', default='rickroll.mp4',
                        help='Path to punishment video (default: rickroll.mp4)')
    parser.add_argument('--cooldown', type=float, default=3.0,
                        help='Seconds between roast messages (default: 3)')
    parser.add_argument('--threshold', type=int, default=1,
                        help='Frames needed to confirm state change (default: 1)')
    parser.add_argument('--sensitivity', type=float, default=0.55,
                        help='Detection sensitivity 0.0-1.0, lower = more sensitive (default: 0.55)')
    args = parser.parse_args()

    detector = DoomscrollDetector(
        video_path=args.video,
        roast_cooldown=args.cooldown,
        detection_threshold=args.threshold,
        sensitivity=args.sensitivity
    )
    detector.run()
