# ndi_utils.py
import numpy as np
from threading import Thread
from queue import Queue
import time

try:
    import NDIlib as ndi
except ImportError:
    print("NDIlib not found. NDI functionality will not be available.")
    ndi = None

class NDIUtils:
    def __init__(self, sender_name="StreamDiffusion_NDI_out", input_name=None, control_name=None):
        self.sender_name = sender_name
        self.input_name = input_name
        self.control_name = control_name
        self.ndi_send = None
        self.input_queue = Queue(maxsize=1)
        self.control_queue = Queue(maxsize=1) if control_name else None
        self.transmit_queue = Queue(maxsize=1)
        self.running = False
        self.input_thread = None
        self.control_thread = None
        self.transmit_thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.input_thread = Thread(target=self._capture_loop, args=(self.input_name, self.input_queue))
            self.input_thread.start()
            if self.control_name:
                self.control_thread = Thread(target=self._capture_loop, args=(self.control_name, self.control_queue))
                self.control_thread.start()
            self.transmit_thread = Thread(target=self._transmit_loop)
            self.transmit_thread.start()

    def stop(self):
        self.running = False
        if self.input_thread:
            self.input_thread.join()
        if self.control_thread:
            self.control_thread.join()
        if self.transmit_thread:
            self.transmit_thread.join()
        if self.ndi_send:
            ndi.send_destroy(self.ndi_send)

    def _capture_loop(self, source_name, queue):
        print(f"Starting capture loop for source: {source_name}")
        
        while self.running:
            try:
                ndi_find = ndi.find_create_v2()
                if ndi_find is None:
                    print(f"Failed to create NDI finder for {source_name}")
                    time.sleep(1)
                    continue

                sources = ndi.find_get_current_sources(ndi_find)
                
                if sources:
                    print(f"Available sources: {[s.ndi_name for s in sources]}")
                    source = next((s for s in sources if source_name in s.ndi_name), None)
                    
                    if source:
                        print(f"Found matching source: {source.ndi_name} for {source_name}")
                        ndi_recv = ndi.recv_create_v3()
                        if ndi_recv is None:
                            print(f"Failed to create receiver for {source_name}")
                            continue

                        ndi.recv_connect(ndi_recv, source)

                        while self.running:
                            t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
                            if t == ndi.FRAME_TYPE_VIDEO:
                                try:
                                    # Use the actual video frame dimensions!
                                    print(f"Frame dimensions: {v.xres}x{v.yres}")
                                    frame = np.fromstring(v.data, dtype=np.uint8).reshape((v.yres, v.xres, 4))
                                    print(f"Frame shape after reshape: {frame.shape}")
                                    
                                    # Convert to RGB
                                    frame = frame[:, :, :3].copy()
                                    print(f"Final frame shape: {frame.shape}")
                                    
                                    if queue.full():
                                        queue.get()
                                    queue.put(frame)
                                except Exception as e:
                                    print(f"Frame error for {source_name}: {str(e)}")
                                    print(f"Data size: {len(v.data)}")
                                    print(f"Video info - xres: {v.xres}, yres: {v.yres}")
                                finally:
                                    ndi.recv_free_video_v2(ndi_recv, v)

                        ndi.recv_destroy(ndi_recv)
                else:
                    time.sleep(0.1)
                
                ndi.find_destroy(ndi_find)

            except Exception as e:
                print(f"Loop error for {source_name}: {str(e)}")
                time.sleep(0.1)
                
    def _transmit_loop(self):
        print(f"!!!!NDI !!!!!!\nSender Name: {self.sender_name}")
        ndi_send_settings = ndi.SendCreate()
        ndi_send_settings.ndi_name = self.sender_name
        self.ndi_send = ndi.send_create(ndi_send_settings)

        while self.running:
            if not self.transmit_queue.empty():
                frame = self.transmit_queue.get()
                video_frame = ndi.VideoFrameV2()
                video_frame.data = frame
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
                ndi.send_send_video_v2(self.ndi_send, video_frame)
            else:
                time.sleep(0.001)

    def capture_input_frame(self):
        print(f"!!!!NDI !!!!!!\nInput Queue Size: {self.input_queue.qsize()}")
        if not self.input_queue.empty():
            return self.input_queue.get()
        return None

    def capture_control_frame(self):
        print(f"!!!!NDI !!!!!!\nControl Queue Size: {self.control_queue.qsize()}")
        if self.control_queue and not self.control_queue.empty():
            return self.control_queue.get()
        return None

    def transmit_frame(self, frame):
        if self.transmit_queue.full():
            self.transmit_queue.get()
        self.transmit_queue.put(frame)
