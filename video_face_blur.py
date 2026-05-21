"""
Video Face Blur Pro - Professional Face Blurring for Videos
Author: Yuseph Alvandi
Description: Two-pass face detection with stabilizer, multi-scale, and audio preservation.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import subprocess
import tempfile
import shutil

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VideoFaceBlurApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Video Face Blur Pro")
        self.window.geometry("1000x700")
        self.window.configure(fg_color="#0a0a0a")
        
        self.video_path = None
        self.output_path = None
        
        self.blur_strength = ctk.IntVar(value=25)
        self.confidence = ctk.DoubleVar(value=0.3)
        
        self.proto_path = os.path.expanduser("~/python_projects/image_processing/models/deploy.prototxt")
        self.model_path = os.path.expanduser("~/python_projects/image_processing/models/res10_300x300_ssd_iter_140000.caffemodel")
        
        self.previous_faces = []
        self.stability_counter = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.pack(fill="x", pady=(20, 10), padx=30)
        ctk.CTkLabel(header, text="Video Face Blur Pro", font=ctk.CTkFont(size=32, weight="bold"), text_color="#1E90FF").pack()
        ctk.CTkLabel(header, text="Two-pass detection with stabilizer and multi-scale", font=ctk.CTkFont(size=14), text_color="#AAAAAA").pack()
        
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        
        left = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        
        ctk.CTkLabel(left, text="Controls", font=ctk.CTkFont(size=18, weight="bold"), text_color="#1E90FF").pack(pady=(20, 15))
        
        ctk.CTkButton(left, text="Open Video", command=self.open_video, height=40, font=ctk.CTkFont(size=14)).pack(pady=10, padx=20, fill="x")
        
        self.file_label = ctk.CTkLabel(left, text="No video selected", text_color="#888888", font=ctk.CTkFont(size=12))
        self.file_label.pack(pady=5)
        
        ctk.CTkLabel(left, text="Blur Strength", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkSlider(left, from_=5, to=55, variable=self.blur_strength, width=250, command=self.update_blur_label).pack()
        self.blur_label = ctk.CTkLabel(left, text="25", font=ctk.CTkFont(size=11), text_color="#1E90FF")
        self.blur_label.pack()
        
        ctk.CTkLabel(left, text="Detection Confidence", font=ctk.CTkFont(size=14, weight="bold"), text_color="#CCCCCC").pack(pady=(15, 5))
        ctk.CTkLabel(left, text="Recommended: 0.3", font=ctk.CTkFont(size=10), text_color="#888888").pack()
        ctk.CTkSlider(left, from_=0.1, to=0.9, variable=self.confidence, width=250, command=self.update_conf_label).pack()
        self.conf_label = ctk.CTkLabel(left, text="0.3", font=ctk.CTkFont(size=11), text_color="#1E90FF")
        self.conf_label.pack()
        
        self.btn_process = ctk.CTkButton(left, text="Start Processing", command=self.process_video, height=45, fg_color="#E67E22", font=ctk.CTkFont(size=15, weight="bold"))
        self.btn_process.pack(pady=20, padx=20, fill="x")
        
        self.progress_label = ctk.CTkLabel(left, text="", text_color="#FFAA33", font=ctk.CTkFont(size=12))
        self.progress_label.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(left, text="Ready", text_color="#4CAF50", font=ctk.CTkFont(size=11))
        self.status_label.pack(pady=10)
        
        right = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12)
        right.pack(side="right", fill="both", expand=True)
        
        self.preview_label = ctk.CTkLabel(right, text="No Video Loaded", font=ctk.CTkFont(size=16), text_color="#555555")
        self.preview_label.pack(expand=True)
    
    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        
        self.video_path = path
        self.file_label.configure(text=os.path.basename(path))
        self.status_label.configure(text="Video loaded. Recommended confidence: 0.3", text_color="#4CAF50")
        
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self.show_frame(frame)
        cap.release()
    
    def update_blur_label(self, value):
        v = int(float(value))
        if v % 2 == 0:
            v += 1
        self.blur_label.configure(text=str(v))
    
    def update_conf_label(self, value):
        self.conf_label.configure(text=f"{float(value):.1f}")
    
    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        preview_w = 550
        ratio = preview_w / pil_img.width
        preview_h = int(pil_img.height * ratio)
        pil_img = pil_img.resize((preview_w, preview_h))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img
    
    def detect_faces_multi_scale(self, frame, net):
        """
        Two-pass detection:
        Pass 1: High confidence (0.5) for clear faces only — avoids false positives.
        Pass 2: User confidence for distant faces in unoccupied areas.
        Pass 3: Zoomed center for very distant faces.
        """
        (h, w) = frame.shape[:2]
        all_detections = []
        occupied_regions = []
        
        # PASS 1: High confidence — only clear faces, no false positives
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sX, sY, eX, eY) = box.astype("int")
                sX, sY = max(0, sX), max(0, sY)
                eX, eY = min(w, eX), min(h, eY)
                all_detections.append(((sX, sY, eX, eY), conf))
                occupied_regions.append((sX, sY, eX, eY))
        
        # PASS 2: User confidence — find smaller faces in unoccupied areas
        blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob2)
        detections2 = net.forward()
        
        user_conf = self.confidence.get()
        for i in range(0, detections2.shape[2]):
            conf = detections2[0, 0, i, 2]
            if conf > user_conf:
                box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sX, sY, eX, eY) = box.astype("int")
                sX, sY = max(0, sX), max(0, sY)
                eX, eY = min(w, eX), min(h, eY)
                
                overlaps = False
                for (ox1, oy1, ox2, oy2) in occupied_regions:
                    ix1, iy1 = max(sX, ox1), max(sY, oy1)
                    ix2, iy2 = min(eX, ox2), min(eY, oy2)
                    if ix2 > ix1 and iy2 > iy1:
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        face_area = (eX - sX) * (eY - sY)
                        if face_area > 0 and intersection / face_area > 0.3:
                            overlaps = True
                            break
                
                if not overlaps:
                    all_detections.append(((sX, sY, eX, eY), conf))
                    occupied_regions.append((sX, sY, eX, eY))
        
        # PASS 3: Zoomed center for very distant faces
        zoom_h = int(h * 0.5)
        zoom_w = int(w * 0.5)
        start_y = (h - zoom_h) // 2
        start_x = (w - zoom_w) // 2
        zoomed = frame[start_y:start_y+zoom_h, start_x:start_x+zoom_w]
        
        if zoomed.size > 0:
            blob_z = cv2.dnn.blobFromImage(cv2.resize(zoomed, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob_z)
            det_z = net.forward()
            
            for i in range(0, det_z.shape[2]):
                conf = det_z[0, 0, i, 2]
                if conf > user_conf + 0.05:
                    box = det_z[0, 0, i, 3:7] * np.array([zoom_w, zoom_h, zoom_w, zoom_h])
                    (sX, sY, eX, eY) = box.astype("int")
                    sX += start_x
                    sY += start_y
                    eX += start_x
                    eY += start_y
                    sX, sY = max(0, sX), max(0, sY)
                    eX, eY = min(w, eX), min(h, eY)
                    
                    overlaps = False
                    for (ox1, oy1, ox2, oy2) in occupied_regions:
                        ix1, iy1 = max(sX, ox1), max(sY, oy1)
                        ix2, iy2 = min(eX, ox2), min(eY, oy2)
                        if ix2 > ix1 and iy2 > iy1:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            face_area = (eX - sX) * (eY - sY)
                            if face_area > 0 and intersection / face_area > 0.3:
                                overlaps = True
                                break
                    
                    if not overlaps:
                        all_detections.append(((sX, sY, eX, eY), conf))
        
        return all_detections
    
    def stabilize_faces(self, current_faces):
        if not self.previous_faces:
            return current_faces
        
        stabilized = []
        used_prev = set()
        
        for curr_box, curr_conf in current_faces:
            cx1, cy1, cx2, cy2 = curr_box
            best_match = None
            best_iou = 0
            
            for pi, (prev_box, _) in enumerate(self.previous_faces):
                if pi in used_prev:
                    continue
                px1, py1, px2, py2 = prev_box
                
                ix1, iy1 = max(cx1, px1), max(cy1, py1)
                ix2, iy2 = min(cx2, px2), min(cy2, py2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    curr_area = (cx2 - cx1) * (cy2 - cy1)
                    prev_area = (px2 - px1) * (py2 - py1)
                    iou = intersection / float(curr_area + prev_area - intersection)
                    
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match = pi
            
            if best_match is not None:
                px1, py1, px2, py2 = self.previous_faces[best_match][0]
                smooth_box = (
                    int(0.7 * cx1 + 0.3 * px1),
                    int(0.7 * cy1 + 0.3 * py1),
                    int(0.7 * cx2 + 0.3 * px2),
                    int(0.7 * cy2 + 0.3 * py2)
                )
                stabilized.append((smooth_box, max(curr_conf, self.previous_faces[best_match][1])))
                used_prev.add(best_match)
            else:
                stabilized.append((curr_box, curr_conf))
        
        for pi, (prev_box, prev_conf) in enumerate(self.previous_faces):
            if pi not in used_prev:
                key = f"{prev_box[0]},{prev_box[1]}"
                self.stability_counter[key] = self.stability_counter.get(key, 0) + 1
                if self.stability_counter.get(key, 0) < 4:
                    stabilized.append((prev_box, prev_conf * 0.7))
        
        return stabilized
    
    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file!")
            return
        
        output = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
        if not output:
            return
        
        self.output_path = output
        self.btn_process.configure(state="disabled", text="Processing...")
        self.progress_label.configure(text="Starting...")
        self.window.update()
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
            
            blur_k = self.blur_strength.get()
            if blur_k % 2 == 0:
                blur_k += 1
            
            self.previous_faces = []
            self.stability_counter = {}
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    detections = self.detect_faces_multi_scale(frame, net)
                    stabilized = self.stabilize_faces(detections)
                    
                    for (startX, startY, endX, endY), _ in stabilized:
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(width, endX), min(height, endY)
                        
                        if endX > startX and endY > startY:
                            face = frame[startY:endY, startX:endX]
                            face = cv2.GaussianBlur(face, (blur_k, blur_k), 30)
                            frame[startY:endY, startX:endX] = face
                    
                    self.previous_faces = [(box, conf) for (box, conf) in stabilized if conf > self.confidence.get()]
                    
                    out.write(frame)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        progress = min(int((frame_count / total_frames) * 100), 100) if total_frames > 0 else 0
                        self.progress_label.configure(text=f"Frame {frame_count} — {progress}%")
                        self.show_frame(frame)
                        self.window.update()
                        
                except Exception:
                    out.write(frame)
                    frame_count += 1
                    continue
            
            cap.release()
            out.release()
            
            # Merge audio
            self.progress_label.configure(text="Merging audio...")
            self.window.update()
            
            try:
                temp_audio = tempfile.NamedTemporaryFile(suffix=".aac", delete=False).name
                subprocess.run(["ffmpeg", "-i", self.video_path, "-vn", "-acodec", "copy", temp_audio, "-y"], check=True, capture_output=True)
                subprocess.run(["ffmpeg", "-i", temp_video, "-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest", output, "-y"], check=True, capture_output=True)
                os.unlink(temp_audio)
            except:
                shutil.copy2(temp_video, output)
            
            os.unlink(temp_video)
            
            self.btn_process.configure(state="normal", text="Start Processing")
            self.progress_label.configure(text="")
            self.status_label.configure(text=f"Done! {frame_count} frames with audio.", text_color="#4CAF50")
            messagebox.showinfo("Complete", f"Video processed!\n{frame_count} frames\nAudio preserved\nOutput: {os.path.basename(output)}")
            
        except Exception as e:
            self.btn_process.configure(state="normal", text="Start Processing")
            self.progress_label.configure(text="")
            self.status_label.configure(text=f"Error: {str(e)[:50]}", text_color="#FF5555")
            messagebox.showerror("Error", f"Processing failed:\n{e}")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = VideoFaceBlurApp()
    app.run()