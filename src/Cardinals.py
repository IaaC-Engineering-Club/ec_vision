import cv2
import numpy as np

class BuoyNavigator:
    def __init__(self):
        # HSV thresholds
        self.red_lower1   = np.array([0, 100, 100]);   self.red_upper1   = np.array([10, 255, 255])
        self.red_lower2   = np.array([160, 100, 100]); self.red_upper2   = np.array([180, 255, 255])
        self.green_lower  = np.array([40,  50,  50]);  self.green_upper  = np.array([90, 255, 255])
        self.yellow_lower = np.array([20, 100, 100]);  self.yellow_upper = np.array([30, 255, 255])

    def detect_buoys(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # masks
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        detections = []

        # helper to process each mask
        def process_mask(mask, color, default_label=None):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 500:  # too small
                    continue
                x, y, w, h = cv2.boundingRect(c)
                label = default_label if default_label else "Unknown"
                detections.append({
                    'box': (x, y, w, h),
                    'area': area,
                    'mask_color': color,
                    'label': label,
                    'mask': mask[y:y+h, x:x+w]
                })

        # detect lateral buoys
        process_mask(red_mask, (0,0,255), "Red Buoy")
        process_mask(green_mask, (0,255,0), "Green Buoy")

        # detect and classify cardinal buoys
        cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # classify orientation
            roi = hsv[y:y+h, x:x+w]
            ym = cv2.inRange(roi, self.yellow_lower, self.yellow_upper) > 0
            # split into 3 horizontal bands
            h3 = max(h // 3, 1)
            bands = [ym[i*h3:(i+1)*h3, :] for i in range(3)]
            cols = []
            for band in bands:
                cols.append("Yellow" if np.sum(band) / band.size > 0.5 else "Black")
            # match pattern
            if cols == ["Black","Yellow","Yellow"]:
                lbl = "North Buoy"
            elif cols == ["Yellow","Yellow","Black"]:
                lbl = "South Buoy"
            elif cols == ["Black","Yellow","Black"]:
                lbl = "East Buoy"
            elif cols == ["Yellow","Black","Yellow"]:
                lbl = "West Buoy"
            else:
                lbl = "Cardinal Buoy"
            detections.append({
                'box': (x, y, w, h),
                'area': area,
                'mask_color': (0,255,255),
                'label': lbl,
                'mask': yellow_mask[y:y+h, x:x+w]
            })

        return detections

    def process_frame(self, frame):
        detections = self.detect_buoys(frame)
        # sort by descending area (closest first)
        detections.sort(key=lambda d: d['area'], reverse=True)

        # decision based on nearest buoy
        if detections:
            top = detections[0]['label']
            if top == "Red Buoy":
                decision = "Turn starboard!"
            elif top == "Green Buoy":
                decision = "Turn port!"
            elif top.endswith("Buoy"):  # North/South/East/West Buoy
                direction = top.split()[0]
                decision = f"Go {direction}!"
            else:
                decision = "Keep course!"
        else:
            decision = "Stop!"

        # draw detections
        for det in detections:
            x, y, w, h = det['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), det['mask_color'], 2)
            cv2.putText(frame, det['label'], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, det['mask_color'], 2)

        # overlay decision in corner
        cv2.rectangle(frame, (5,5), (250,50), (0,0,0), -1)
        cv2.putText(frame, decision, (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return frame

    def run_on_video(self, inp_path, out_path):
        cap = cv2.VideoCapture(inp_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = self.process_frame(frame)
            writer.write(out_frame)

        cap.release()
        writer.release()
        return out_path

# Example usage:
navigator = BuoyNavigator()
input_video = "/content/video bouys 2.mp4"  # your 3D simulated path video
output_video = "video_bouys_detection.mp4"
navigator.run_on_video(input_video, output_video)


