# Definin tha navigator (with red, green & yellow-cardinal buoys)
from vision import visionNav as vn
class BuoyNavigator (vn):
    def __init__(self):
        self.red_lower1   = np.array([0, 100, 100]);   self.red_upper1   = np.array([10, 255, 255])
        self.red_lower2   = np.array([160, 100, 100]); self.red_upper2   = np.array([180, 255, 255])
        self.green_lower  = np.array([40, 50, 50]);    self.green_upper  = np.array([ 90, 255, 255])
        self.yellow_lower = np.array([20, 100, 100]);  self.yellow_upper = np.array([30, 255, 255])
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # masks
        red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red  = cv2.bitwise_or(red1, red2)
        green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        # contours → boxes
        boxes = []
        for mask, col, label in [(red,(0,0,255),"Red Buoy"),
                                 (green,(0,255,0),"Green Buoy"),
                                 (yellow,(0,255,255),"",)]:  # yellow label handled below
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 300: continue
                x,y,ww,hh = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+ww,y+hh),col,2)
                if label:
                    cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
                else:
                    # cardinal logic:
                    cx,cy = x+ww//2, y+hh//2
                    midx,midy = w//2, h//2
                    if cy < midy-50: txt="North Buoy"
                    elif cy > midy+50: txt="South Buoy"
                    elif cx < midx-50: txt="West Buoy"
                    else: txt="East Buoy"
                    cv2.putText(frame,txt,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        return frame

    def run_on_video(self, inp, outp):
        cap = cv2.VideoCapture(inp)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w,h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(5) or 10
        writer = cv2.VideoWriter(outp, fourcc, fps, (w,h))
        while True:
            ret,frm = cap.read()
            if not ret: break
            writer.write(self.process_frame(frm))
        cap.release(); writer.release()
        return outp

# 3️⃣ Run it
navigator = BuoyNavigator()
input_video  = "your_uploaded_input.mp4"                  # <- upload this via Colab’s file UI
output_video = "with_cardinals.mp4"
navigator.run_on_video(input_video, output_video)

# 4️⃣ Trigger download
files.download(output_video)
