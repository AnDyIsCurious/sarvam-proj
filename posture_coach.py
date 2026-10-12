"""
Posture Coach Application
Combines webcam posture tracking (OpenCV & MediaPipe Pose) with Sarvam AI's
Text-to-Speech (Bulbul) and Speech-to-Text (Saaras) APIs.
"""

import base64
import os
import tempfile
import threading
import time
import wave
import cv2
import mediapipe as mp
import numpy as np
import pygame
import requests
import sounddevice as sd

# ==============================================================================
# Configuration
# ==============================================================================
SARVAM_API_KEY = "YOUR_SARVAM_API_KEY_HERE"F"

TTS_ENDPOINT = "https://api.sarvam.ai/text-to-speech"
STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

ALERT_WAV_FILE = "alert.wav"
AUDIO_SAMPLE_RATE = 16000
AUDIO_RECORD_DURATION = 3.0  # seconds
SLOUCH_THRESHOLD_RATIO = 0.15  # 15% drop from upright baseline
SLOUCH_TIME_LIMIT = 4.0  # 4 continuous seconds triggers alert
SNOOZE_DURATION = 60.0  # 60 seconds pause on voice dismissal

# Voice dismissal keywords
DISMISS_KEYWORDS = ["snooze", "stop", "ok", "okay", "theek", "thik", "dismiss", "cancel", "chup", "ruko"]

# ==============================================================================
# Global State & Thread Synchronization
# ==============================================================================
state_lock = threading.Lock()

# Calibration state
is_calibrated = False
baseline_nose_y = 0.0
baseline_shoulder_mid_y = 0.0
baseline_posture_height = 0.0  # vertical distance between shoulders midpoint and nose

# Slouch tracking state
slouch_start_time = None
is_currently_slouching = False

# Alerts and Snooze state
snooze_until = 0.0
is_alert_active = False
is_recording_voice = False
last_transcript = "None yet"
voice_status_text = "Idle (Press [S] to speak)"
calibrated_banner_until = 0.0


def initialize_audio():
    """Initialize pygame mixer for playing alert sounds."""
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[Warning] Failed to initialize pygame mixer: {e}")


def play_alert_sound(filepath):
    """Play alert.wav using pygame.mixer and wait until playback finishes."""
    try:
        if not os.path.exists(filepath):
            print(f"[Warning] Audio file {filepath} not found.")
            return

        sound = pygame.mixer.Sound(filepath)
        channel = sound.play()
        while channel and channel.get_busy():
            time.sleep(0.05)
    except Exception as e:
        print(f"[Error] Audio playback failed: {e}")


def trigger_sarvam_tts(text="Please sit upright and fix your posture."):
    """
    Calls Sarvam Text-to-Speech (Bulbul:v3) API, decodes base64 audios[0],
    saves to alert.wav, and plays using pygame.mixer.
    """
    global is_alert_active, voice_status_text
    print("\n[TTS] Triggering Sarvam Text-to-Speech alert...")

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "language_code": "en-IN",
        "model": "bulbul:v3",
        "speaker": "shubh"
    }

    try:
        with state_lock:
            voice_status_text = "Playing Voice Alert..."

        response = requests.post(TTS_ENDPOINT, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            audios = data.get("audios", [])
            if audios:
                audio_bytes = base64.b64decode(audios[0])
                with open(ALERT_WAV_FILE, "wb") as f:
                    f.write(audio_bytes)
                print(f"[TTS] Saved alert to {ALERT_WAV_FILE}. Playing...")
                play_alert_sound(ALERT_WAV_FILE)
            else:
                print("[TTS Error] Response did not contain 'audios' field.")
        else:
            print(f"[TTS Error] API returned status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[TTS Error] Request failed: {e}")


def record_and_transcribe():
    """
    Records 3 seconds from the microphone using sounddevice at 16000 Hz,
    saves to a temporary WAV file, and POSTs to Sarvam Speech-to-Text (Saaras:v4).
    If dismiss keywords are detected, pauses alerts for 60 seconds.
    """
    global is_recording_voice, last_transcript, voice_status_text, snooze_until

    with state_lock:
        if is_recording_voice:
            return  # Already recording
        is_recording_voice = True
        voice_status_text = "Listening for 3 seconds..."

    temp_wav_path = None
    try:
        print(f"\n[STT] Recording {AUDIO_RECORD_DURATION} seconds at {AUDIO_SAMPLE_RATE} Hz...")
        recording = sd.rec(
            int(AUDIO_RECORD_DURATION * AUDIO_SAMPLE_RATE),
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()
        print("[STT] Recording complete. Saving temporary WAV file...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav_path = temp_file.name

        with wave.open(temp_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(recording.tobytes())

        with state_lock:
            voice_status_text = "Transcribing with Saaras..."

        print("[STT] Sending audio to Sarvam Saaras API...")
        headers = {
            "api-subscription-key": SARVAM_API_KEY
        }
        data = {
            "model": "saaras:v4"
        }

        with open(temp_wav_path, "rb") as audio_file:
            files = {
                "file": ("voice.wav", audio_file, "audio/wav")
            }
            response = requests.post(STT_ENDPOINT, headers=headers, data=data, files=files, timeout=12)

        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            print(f"[STT] Transcribed text: '{transcript}'")

            with state_lock:
                last_transcript = transcript if transcript else "(No speech detected)"
                transcript_lower = transcript.lower()

                # Check for dismissal keywords
                matched_keyword = None
                for keyword in DISMISS_KEYWORDS:
                    if keyword in transcript_lower:
                        matched_keyword = keyword
                        break

                if matched_keyword:
                    snooze_until = time.time() + SNOOZE_DURATION
                    voice_status_text = f"Snoozed by Voice! ('{matched_keyword}')"
                    print(f"[STT] Alert paused for 60s! Triggered by keyword '{matched_keyword}'.")
                else:
                    voice_status_text = f"Heard: '{last_transcript}'"
        else:
            print(f"[STT Error] API returned status {response.status_code}: {response.text}")
            with state_lock:
                voice_status_text = f"STT Error ({response.status_code})"

    except Exception as e:
        print(f"[STT Error] Voice dismissal failed: {e}")
        with state_lock:
            voice_status_text = "Mic / STT Error"
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass
        with state_lock:
            is_recording_voice = False


def alert_cycle_worker():
    """
    Background worker that runs TTS alert and immediately follows up
    with automatic voice dismissal listening.
    """
    global is_alert_active, slouch_start_time
    try:
        trigger_sarvam_tts("Please sit upright and fix your posture.")
        # Automatic voice dismissal after alert
        record_and_transcribe()
    finally:
        with state_lock:
            is_alert_active = False
            # Reset slouch start time so another full 4 continuous seconds are needed
            slouch_start_time = time.time()


def trigger_alert_in_background():
    """Launches TTS alert and subsequent voice recording in a background thread."""
    global is_alert_active
    with state_lock:
        if is_alert_active or is_recording_voice:
            return
        is_alert_active = True

    t = threading.Thread(target=alert_cycle_worker, daemon=True)
    t.start()


def trigger_manual_voice_recording():
    """Launches voice recording in a background thread when 's' key is pressed."""
    global is_recording_voice
    with state_lock:
        if is_recording_voice:
            print("[STT] Voice recording is already in progress.")
            return

    t = threading.Thread(target=record_and_transcribe, daemon=True)
    t.start()


def draw_hud(frame, pose_landmarks, width, height):
    """Draws sleek on-screen UI overlays, guidelines, and status indicators."""
    global is_calibrated, baseline_nose_y, baseline_shoulder_mid_y, baseline_posture_height
    global is_currently_slouching, slouch_start_time, snooze_until, calibrated_banner_until

    curr_time = time.time()
    overlay = frame.copy()

    # 1. Top Bar Banner
    cv2.rectangle(overlay, (0, 0), (width, 42), (25, 25, 25), -1)
    # Blend top bar banner with slight transparency
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    top_bar_text = "Press [C] to Calibrate | Press [S] to Speak Voice Command | Press [Q] to Quit"
    cv2.putText(frame, top_bar_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Calibration Guideline and Landmark Visuals
    if pose_landmarks:
        # Landmark 0: Nose, 11: Left Shoulder, 12: Right Shoulder
        nose = pose_landmarks.landmark[0]
        ls = pose_landmarks.landmark[11]
        rs = pose_landmarks.landmark[12]

        nose_px = (int(nose.x * width), int(nose.y * height))
        ls_px = (int(ls.x * width), int(ls.y * height))
        rs_px = (int(rs.x * width), int(rs.y * height))
        shoulder_mid_px = (int((ls_px[0] + rs_px[0]) / 2), int((ls_px[1] + rs_px[1]) / 2))

        # Draw shoulder connection and midpoint
        cv2.line(frame, ls_px, rs_px, (200, 150, 50), 2, cv2.LINE_AA)
        cv2.circle(frame, ls_px, 5, (255, 180, 0), -1)
        cv2.circle(frame, rs_px, 5, (255, 180, 0), -1)
        cv2.circle(frame, shoulder_mid_px, 6, (0, 200, 255), -1)

        # Draw line from nose to shoulder midpoint
        cv2.line(frame, nose_px, shoulder_mid_px, (180, 180, 180), 2, cv2.LINE_AA)

        # Draw nose point (Green if good, Red if slouching)
        nose_color = (0, 0, 255) if is_currently_slouching else (0, 255, 0)
        cv2.circle(frame, nose_px, 8, nose_color, -1)
        cv2.circle(frame, nose_px, 10, (255, 255, 255), 1, cv2.LINE_AA)

        # If calibrated, draw horizontal baseline and slouch threshold line
        if is_calibrated:
            base_y_px = int(baseline_nose_y * height)
            threshold_y_px = int((baseline_nose_y + SLOUCH_THRESHOLD_RATIO * baseline_posture_height) * height)

            # Baseline line (Cyan)
            cv2.line(frame, (20, base_y_px), (width - 20, base_y_px), (255, 230, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Baseline", (width - 95, base_y_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 0), 1, cv2.LINE_AA)

            # Threshold line (Red/Orange dashed effect)
            cv2.line(frame, (20, threshold_y_px), (width - 20, threshold_y_px), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Slouch Limit (-15%)", (width - 150, threshold_y_px + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # 3. Status Display (Top Left area)
    # Background panel for status
    cv2.rectangle(frame, (15, 55), (380, 130), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 55), (380, 130), (60, 60, 60), 1)

    if not is_calibrated:
        cv2.putText(frame, "Status: Not Calibrated", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Sit straight and press [C]", (25, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        # Check if snooze is currently active
        is_snoozed = curr_time < snooze_until
        snooze_rem = int(snooze_until - curr_time) if is_snoozed else 0

        if is_currently_slouching:
            elapsed = curr_time - slouch_start_time if slouch_start_time else 0.0
            status_text = f"Slouching: {elapsed:.1f}s"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "Posture: Good"
            status_color = (0, 255, 0)  # Green

        cv2.putText(frame, status_text, (25, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)

        # Sub-status details
        if is_snoozed:
            sub_text = f"Snoozed by Voice! ({snooze_rem}s remaining)"
            sub_color = (255, 200, 0)
        elif is_currently_slouching:
            sub_text = f"Alert in: {max(0.0, SLOUCH_TIME_LIMIT - elapsed):.1f}s"
            sub_color = (0, 165, 255)
        else:
            sub_text = "Baseline locked | Monitoring posture"
            sub_color = (180, 220, 180)

        cv2.putText(frame, sub_text, (25, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, sub_color, 1, cv2.LINE_AA)

    # 4. Calibration Banner Popup
    if curr_time < calibrated_banner_until:
        cv2.rectangle(frame, (width // 2 - 160, 52), (width // 2 + 160, 92), (0, 140, 0), -1)
        cv2.putText(frame, "Calibrated Successfully!", (width // 2 - 130, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 5. Voice & Snooze Status Bar (Bottom Panel)
    cv2.rectangle(frame, (15, height - 70), (width - 15, height - 15), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, height - 70), (width - 15, height - 15), (60, 60, 60), 1)

    with state_lock:
        curr_voice_status = voice_status_text
        curr_transcript = last_transcript
        snooze_active = curr_time < snooze_until

    # Voice status icon/indicator
    if is_recording_voice:
        rec_dot_color = (0, 0, 255)  # Pulsing/solid red for recording
        cv2.circle(frame, (35, height - 52), 6, rec_dot_color, -1)
    else:
        cv2.circle(frame, (35, height - 52), 6, (0, 200, 0), -1)

    cv2.putText(frame, f"Voice Status: {curr_voice_status}", (50, height - 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    transcript_display = f"Last Words (Saaras): \"{curr_transcript}\""
    cv2.putText(frame, transcript_display, (50, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)

    # Prominent "Snoozed by Voice!" badge if snooze is active
    if snooze_active:
        cv2.rectangle(frame, (width - 240, height - 63), (width - 25, height - 23), (120, 80, 0), -1)
        cv2.rectangle(frame, (width - 240, height - 63), (width - 25, height - 23), (255, 200, 0), 1)
        cv2.putText(frame, "Snoozed by Voice!", (width - 225, height - 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def run_posture_coach():
    """Main loop combining camera capture, pose tracking, and event handling."""
    global is_calibrated, baseline_nose_y, baseline_shoulder_mid_y, baseline_posture_height
    global slouch_start_time, is_currently_slouching, calibrated_banner_until

    initialize_audio()

    print("\n=======================================================")
    print("           AI POSTURE COACH - SARVAM AI               ")
    print("=======================================================")
    print("Controls:")
    print("  [C] : Calibrate upright posture height")
    print("  [S] : Speak voice command (Saaras STT)")
    print("  [Q] : Quit application")
    print("=======================================================\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam at index 0.")
        return

    # Use standard 640x480 resolution for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    cv2.namedWindow('Posture Coach', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Posture Coach', 800, 600)
    cv2.setWindowProperty('Posture Coach', cv2.WND_PROP_TOPMOST, 1)

    first_frame_rendered = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to grab webcam frame.")
                break

            # Flip horizontally for a natural mirror view
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            curr_nose_y = None
            curr_shoulder_mid_y = None
            curr_posture_height = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[0]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]

                # Check landmark visibility
                if nose.visibility > 0.4 and left_shoulder.visibility > 0.4 and right_shoulder.visibility > 0.4:
                    curr_nose_y = nose.y
                    curr_shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
                    curr_posture_height = abs(curr_shoulder_mid_y - curr_nose_y)

            # Slouch Detection Logic
            if is_calibrated and curr_nose_y is not None and curr_posture_height is not None:
                # Upright baseline reference
                # In image coordinates, Y increases downwards.
                # A drop in posture height means curr_posture_height drops below 85% of baseline (15% drop).
                # Additionally, head lowering means nose_y drops downwards by > 15% of baseline posture height.
                height_drop_condition = curr_posture_height < (baseline_posture_height * (1.0 - SLOUCH_THRESHOLD_RATIO))
                nose_drop_condition = curr_nose_y > (baseline_nose_y + SLOUCH_THRESHOLD_RATIO * baseline_posture_height)

                if height_drop_condition or nose_drop_condition:
                    # User is slouching
                    is_currently_slouching = True
                    if slouch_start_time is None:
                        slouch_start_time = time.time()

                    elapsed_slouch = time.time() - slouch_start_time
                    # If slouching continues for 4 seconds and alerts are not snoozed
                    if elapsed_slouch >= SLOUCH_TIME_LIMIT:
                        with state_lock:
                            can_alert = (time.time() >= snooze_until) and (not is_alert_active) and (not is_recording_voice)

                        if can_alert:
                            trigger_alert_in_background()
                else:
                    # User is upright! Reset slouch timer
                    is_currently_slouching = False
                    slouch_start_time = None
            else:
                is_currently_slouching = False
                slouch_start_time = None

            # Render Heads-Up Display
            draw_hud(frame, results.pose_landmarks, width, height)

            cv2.imshow('Posture Coach', frame)

            if not first_frame_rendered:
                print('Frame rendered successfully')
                first_frame_rendered = True

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n[Exit] 'Q' pressed. Exiting Posture Coach...")
                break

            elif key == ord('c') or key == ord('C'):
                # Calibrate upright posture
                if curr_nose_y is not None and curr_shoulder_mid_y is not None:
                    baseline_nose_y = curr_nose_y
                    baseline_shoulder_mid_y = curr_shoulder_mid_y
                    baseline_posture_height = max(0.05, abs(curr_shoulder_mid_y - curr_nose_y))
                    is_calibrated = True
                    slouch_start_time = None
                    is_currently_slouching = False
                    calibrated_banner_until = time.time() + 2.0
                    print(f"\n[Calibration] Upright baseline set!")
                    print(f"  Nose Y: {baseline_nose_y:.3f}")
                    print(f"  Shoulder Mid Y: {baseline_shoulder_mid_y:.3f}")
                    print(f"  Baseline Posture Height: {baseline_posture_height:.3f}")
                else:
                    print("[Calibration] No clear pose detected. Sit upright facing the camera and press 'C' again.")

            elif key == ord('s') or key == ord('S'):
                # Voice dismissal / command
                print("\n[Voice] 'S' pressed. Initiating voice recording...")
                trigger_manual_voice_recording()

    finally:
        print("[Cleanup] Releasing camera and closing windows...")
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        print("[Done] Posture Coach terminated cleanly.")


if __name__ == "__main__":
    run_posture_coach()
