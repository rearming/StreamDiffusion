"""
StreamDiffusion Real-time GUI for RTX 5080
- Live camera feed → diffusion → output
- Change prompt, guidance, strength in real-time
- Model selection
"""
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class StreamDiffusionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("StreamDiffusion RT — RTX 5080")
        self.root.configure(bg="#1a1a2e")

        # State
        self.running = False
        self.stream = None
        self.cap = None
        self.fps = 0.0
        self.frame_count = 0
        self.loading = False

        # Defaults
        self.current_model = "stabilityai/sd-turbo"
        self.current_prompt = "a beautiful oil painting"
        self.current_negative = "ugly, blurry, low quality"
        self.current_guidance = 1.2
        self.current_delta = 0.5
        self.current_strength_idx = 1  # index into t_index presets
        self.prompt_dirty = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#00d4ff", font=("Segoe UI", 14, "bold"))
        style.configure("FPS.TLabel", background="#1a1a2e", foreground="#00ff88", font=("Consolas", 12, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TScale", background="#1a1a2e")

        # Main layout: left=controls, right=video
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - controls
        left = ttk.Frame(main, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)

        ttk.Label(left, text="StreamDiffusion RT", style="Header.TLabel").pack(pady=(5, 10))

        # Model selection
        ttk.Label(left, text="Model:").pack(anchor=tk.W, padx=5)
        self.model_var = tk.StringVar(value=self.current_model)
        model_combo = ttk.Combobox(left, textvariable=self.model_var, width=35, values=[
            "stabilityai/sd-turbo",
            "stabilityai/sdxl-turbo",
            "runwayml/stable-diffusion-v1-5",
            "KBlueLeaf/kohaku-v2.1",
        ])
        model_combo.pack(padx=5, pady=(0, 10))

        # Prompt
        ttk.Label(left, text="Prompt:").pack(anchor=tk.W, padx=5)
        self.prompt_entry = tk.Text(left, height=3, width=35, bg="#16213e", fg="#e0e0e0",
                                     insertbackground="#00d4ff", font=("Segoe UI", 10), wrap=tk.WORD)
        self.prompt_entry.insert("1.0", self.current_prompt)
        self.prompt_entry.pack(padx=5, pady=(0, 5))
        self.prompt_entry.bind("<KeyRelease>", self._on_prompt_change)

        # Negative prompt
        ttk.Label(left, text="Negative prompt:").pack(anchor=tk.W, padx=5)
        self.neg_entry = tk.Text(left, height=2, width=35, bg="#16213e", fg="#e0e0e0",
                                  insertbackground="#00d4ff", font=("Segoe UI", 10), wrap=tk.WORD)
        self.neg_entry.insert("1.0", self.current_negative)
        self.neg_entry.pack(padx=5, pady=(0, 10))
        self.neg_entry.bind("<KeyRelease>", self._on_prompt_change)

        # Guidance scale
        ttk.Label(left, text="Guidance scale:").pack(anchor=tk.W, padx=5)
        self.guidance_var = tk.DoubleVar(value=self.current_guidance)
        self.guidance_label = ttk.Label(left, text=f"{self.current_guidance:.1f}")
        self.guidance_label.pack(anchor=tk.E, padx=5)
        guidance_scale = ttk.Scale(left, from_=0.0, to=5.0, variable=self.guidance_var,
                                    command=self._on_guidance_change)
        guidance_scale.pack(fill=tk.X, padx=5, pady=(0, 10))

        # Delta (noise strength)
        ttk.Label(left, text="Delta (noise):").pack(anchor=tk.W, padx=5)
        self.delta_var = tk.DoubleVar(value=self.current_delta)
        self.delta_label = ttk.Label(left, text=f"{self.current_delta:.2f}")
        self.delta_label.pack(anchor=tk.E, padx=5)
        delta_scale = ttk.Scale(left, from_=0.0, to=2.0, variable=self.delta_var,
                                 command=self._on_delta_change)
        delta_scale.pack(fill=tk.X, padx=5, pady=(0, 10))

        # Diffusion strength preset
        ttk.Label(left, text="Diffusion strength:").pack(anchor=tk.W, padx=5)
        self.strength_var = tk.StringVar(value="Medium")
        strength_combo = ttk.Combobox(left, textvariable=self.strength_var, width=35, values=[
            "Subtle [45]",
            "Medium [32, 45]",
            "Strong [22, 32, 45]",
            "Very Strong [0, 16, 32, 45]",
        ], state="readonly")
        strength_combo.current(1)
        strength_combo.pack(padx=5, pady=(0, 10))

        # Seed
        ttk.Label(left, text="Seed:").pack(anchor=tk.W, padx=5)
        self.seed_var = tk.StringVar(value="42")
        seed_entry = ttk.Entry(left, textvariable=self.seed_var, width=15)
        seed_entry.pack(anchor=tk.W, padx=5, pady=(0, 10))

        # Camera selection
        ttk.Label(left, text="Camera:").pack(anchor=tk.W, padx=5)
        self.camera_var = tk.IntVar(value=0)
        cam_frame = ttk.Frame(left)
        cam_frame.pack(anchor=tk.W, padx=5, pady=(0, 10))
        for i in range(3):
            ttk.Radiobutton(cam_frame, text=str(i), variable=self.camera_var, value=i).pack(side=tk.LEFT, padx=3)

        # Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self._start)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, foreground="#888888").pack(pady=5)

        # FPS counter
        self.fps_label = ttk.Label(left, text="0.0 FPS", style="FPS.TLabel")
        self.fps_label.pack(pady=5)

        # Right panel - video
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Input video
        ttk.Label(right, text="Camera Input").pack()
        self.input_canvas = tk.Canvas(right, width=512, height=512, bg="#0a0a1a", highlightthickness=0)
        self.input_canvas.pack(pady=(0, 5))

        # Output video
        ttk.Label(right, text="Diffusion Output").pack()
        self.output_canvas = tk.Canvas(right, width=512, height=512, bg="#0a0a1a", highlightthickness=0)
        self.output_canvas.pack()

    def _on_prompt_change(self, event=None):
        self.prompt_dirty = True

    def _on_guidance_change(self, val):
        self.current_guidance = float(val)
        self.guidance_label.config(text=f"{self.current_guidance:.1f}")

    def _on_delta_change(self, val):
        self.current_delta = float(val)
        self.delta_label.config(text=f"{self.current_delta:.2f}")

    def _get_t_index_list(self):
        presets = {
            "Subtle [45]": [45],
            "Medium [32, 45]": [32, 45],
            "Strong [22, 32, 45]": [22, 32, 45],
            "Very Strong [0, 16, 32, 45]": [0, 16, 32, 45],
        }
        return presets.get(self.strength_var.get(), [32, 45])

    def _start(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.loading = True
        self.status_var.set("Loading model...")
        threading.Thread(target=self._init_and_run, daemon=True).start()

    def _stop(self):
        self.running = False
        self.stop_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("Stopped")
        if self.cap:
            self.cap.release()
            self.cap = None

    def _init_and_run(self):
        try:
            from streamdiffusion import StreamDiffusionWrapper

            model = self.model_var.get()
            t_list = self._get_t_index_list()
            seed = int(self.seed_var.get()) if self.seed_var.get().isdigit() else 42

            self.root.after(0, lambda: self.status_var.set(f"Loading {model}..."))

            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model,
                t_index_list=t_list,
                frame_buffer_size=1,
                width=512,
                height=512,
                warmup=10,
                acceleration="tensorrt",
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="self" if self.current_guidance > 1.0 else "none",
                seed=seed,
                use_lcm_lora=False,
                use_tiny_vae=True,
                engine_dir="engines/gui",
            )

            prompt = self.prompt_entry.get("1.0", tk.END).strip()
            neg = self.neg_entry.get("1.0", tk.END).strip()

            self.stream.prepare(
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=50,
                guidance_scale=self.current_guidance,
                delta=self.current_delta,
            )

            # Open camera
            cam_id = self.camera_var.get()
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                self.root.after(0, lambda: self.status_var.set(f"Camera {cam_id} not found!"))
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

            # Warmup
            self.root.after(0, lambda: self.status_var.set("Warming up..."))
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((512, 512))
                img_tensor = self.stream.preprocess_image(pil_img)
                for _ in range(self.stream.batch_size):
                    self.stream(image=img_tensor)

            self.running = True
            self.loading = False
            self.prompt_dirty = False
            self.root.after(0, lambda: self.status_var.set("Running"))

            # Main loop
            fps_time = time.time()
            fps_count = 0

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_input = Image.fromarray(frame_rgb).resize((512, 512))

                # Update prompt if changed
                if self.prompt_dirty:
                    prompt = self.prompt_entry.get("1.0", tk.END).strip()
                    neg = self.neg_entry.get("1.0", tk.END).strip()
                    self.stream.prepare(
                        prompt=prompt,
                        negative_prompt=neg,
                        num_inference_steps=50,
                        guidance_scale=self.current_guidance,
                        delta=self.current_delta,
                    )
                    self.prompt_dirty = False

                # Run diffusion
                img_tensor = self.stream.preprocess_image(pil_input)
                output_image = self.stream(image=img_tensor)

                # Update canvases
                fps_count += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    self.fps = fps_count / (now - fps_time)
                    fps_count = 0
                    fps_time = now
                    self.root.after(0, lambda f=self.fps: self.fps_label.config(text=f"{f:.1f} FPS"))

                self.root.after(0, self._update_canvas, self.input_canvas, pil_input, "input_photo")
                self.root.after(0, self._update_canvas, self.output_canvas, output_image, "output_photo")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)[:60]}"))
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.running = False

    def _update_canvas(self, canvas, pil_img, attr_name):
        if not self.running and not self.loading:
            return
        tk_img = ImageTk.PhotoImage(pil_img)
        setattr(self, attr_name, tk_img)  # prevent GC
        canvas.create_image(256, 256, image=tk_img)

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StreamDiffusionGUI()
    app.run()
