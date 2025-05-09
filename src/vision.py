import cv2 as cv
import numpy as np

class visionNav:
    def __init__(self, video=None):

        self.video = video
        self.image = None

        self.hsv_color = None
        self.mask_r = None
        self.mask_g = None

        self.middle_x = None
        self.height = None
        self.width = None
        self.distance = None

    def text_size(self, width ,direction):
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        thickness = 4
        text_size = cv.getTextSize(direction, font, scale, thickness)[0]
        cv.putText(self.image, direction, ((width - text_size[0])//2, 50), font, scale, (0, 0, 0), thickness, cv.LINE_AA)

    def generate_masks(self):
        if self.image is not None:

            image_bilateral = cv.bilateralFilter(self.image, 25, 400, 400)
            self.hsv_color = cv.cvtColor(image_bilateral, cv.COLOR_BGR2HSV)

            red = {
                "lower1": np.array([0, 40, 40]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([170, 0, 20]),
                "upper2": np.array([180, 255, 255])
            }
            green = {
                "lower": np.array([30, 40, 0]),
                "upper": np.array([90, 255, 255])
            }

            # green mask
            self.mask_g = cv.inRange(self.hsv_color, green["lower"], green["upper"])
            
            # red mask
            mask_r1 = cv.inRange(self.hsv_color, red["lower1"], red["upper1"])
            mask_r2 = cv.inRange(self.hsv_color, red["lower2"], red["upper2"])
            self.mask_r = mask_r1 | mask_r2

            # Morphological operations
            self.mask_g = self.morphops(self.mask_g)
            self.mask_r = self.morphops(self.mask_r)
        else:
            print("No image loaded.")

    def morphops(self, mask):
        kernel = np.ones((5,5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,kernel)
        mask = cv.erode(mask, kernel, iterations=4)
        return mask

    def detect(self, mask, min_area, color, description):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea) if contours else None
        x,y,w,h= cv.boundingRect(biggest_contour)
        position = x + w // 2
        box = cv.rectangle(self.image, (x,y), (x+w, y+h), color, 5)
        cv.putText(self.image, f"{description} BUOY \n width:{w} height:{h}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return (True, position, x, y, w, h)
    
    def line_style(self, centerp, p1, p2, color, thickness, description):
        cv.line(self.image, p1, p2, color, thickness)
        cv.circle(self.image, centerp, 5, color, thickness)
        self.text_size(self.width,description)
    
    def distance_between(self, x_green, x_red, y_green, y_red, green_w, green_h, red_w, red_h):
        green_point = ((x_green + (green_w // 2)), (y_green + (green_h // 2)))
        green_missing = (0,(y_red + (red_h // 2)))
        red_point = ((x_red + (red_w // 2)), (y_red + (red_h // 2)))
        red_missing = (self.width, (y_green + (green_h // 2)))
        distance = x_green - x_red
        self.middle_x = abs((x_green - x_red) // 2)
        center_line = (((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2, ((y_green + (green_h // 2)) + (y_red + (red_h // 2))) // 2)

        if distance < 0:

            if x_green == 0 and y_green == 0:
                if x_red == 0 and y_red == 0:
                    self.line_style(red_missing,red_missing,green_missing,(0, 0, 0),1,"CLEAR! FOLLOW GPS!")
                else:
                    self.line_style(green_missing,green_missing,red_point,(0, 255, 0),3,"Turn Starboard!") 
            
            else:
                # Create points correctly with brackets
                point1 = np.array([((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2, self.height // 2])
                point2 = np.array([self.width // 2, self.height // 2])
                self.distance = cv.norm(point1, point2)
                if (((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2) > self.width // 2:
                    self.line_style(center_line,green_point, red_point, (0, 255, 255), 3, f"Keep Route! Move right: {int(self.distance)}")
                elif (((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2) < self.width // 2:
                    self.line_style(center_line,green_point, red_point, (0, 255, 255), 3, f"Keep Route! Move left: {int(self.distance)}")
                #line between buoys and center of the frame
                cv.line(self.image, center_line,(((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2, self.height // 2), (0, 255, 255),2)
                cv.line(self.image,(((x_green + (green_w // 2)) + (x_red + (red_w // 2))) // 2, self.height // 2),(self.width // 2, self.height // 2), (0, 255, 255),2)
        else:
            if x_red == 0 and y_red == 0:
                if x_green == 0 and y_green == 0:
                    self.line_style(green_missing,green_missing,red_missing,(0, 0, 0),1,"CLEAR! FOLLOW GPS!")
                else:
                    self.line_style(red_missing,green_point,red_missing,(0, 0, 255),3,"Turn Port!")
            else:
                self.line_style(center_line,green_point, red_point, (0, 0, 255), 3, f"Turn Around!")
            
    def json(self, gx, gy, gw, gh, gtype, rx, ry, rw, rh, rtype):
        if rx == 0 and ry == 0:
            data = {
                "green_buoy": {
                    "x": gx,
                    "y": gy,
                    "width": gw,
                    "height": gh,
                    "type": gtype
                }
            }
            text = f"GREEN BUOY:\nx: {data['green_buoy']['x']}\ny: {data['green_buoy']['y']}\nsize: {data['green_buoy']['width']}x{data['green_buoy']['height']}"
        elif gx == 0 and gy == 0:
            data = {
                "red_buoy": {
                    "x": rx,
                    "y": ry,
                    "width": rw,
                    "height": rh,
                    "type": rtype
                }
            }
            text = f"RED BUOY:\nx: {data['red_buoy']['x']}\ny: {data['red_buoy']['y']}\nsize: {data['red_buoy']['width']}x{data['red_buoy']['height']}"
        else:
            data = {
                "green_buoy": {
                    "x": gx,
                    "y": gy,
                    "width": gw,
                    "height": gh,
                    "type": gtype
                },
                "red_buoy": {
                    "x": rx,
                    "y": ry,
                    "width": rw,
                    "height": rh,
                    "type": rtype
                }
            }
            text = f"GREEN BUOY:\nx: {data['green_buoy']['x']}\ny: {data['green_buoy']['y']}\nsize: {data['green_buoy']['width']}x{data['green_buoy']['height']}\n\nRED BUOY:\nx: {data['red_buoy']['x']}\ny: {data['red_buoy']['y']}\nsize: {data['red_buoy']['width']}x{data['red_buoy']['height']}"
        return text, data

    def put_text_multiline(self,img, text, org, font, font_scale, color, thickness=1, line_spacing=10):
        x, y = org
        for i, line in enumerate(text.split('\n')):
            y_pos = y + i * (line_spacing + thickness)
            cv.putText(img, line, (x, y_pos), font, font_scale, color, thickness)
        return img

    def detect_buoys(self, min_area = 1000):

        green_detected, green_position, green_x, green_y, green_w, green_h = self.detect(self.mask_g, min_area, (0, 255, 0),"GREEN") if self.detect(self.mask_g, min_area, (0, 255, 0), "GREEN") else (False, None)
        red_detected, red_position, red_x, red_y, red_w, red_h = self.detect(self.mask_r, min_area, (0, 0, 255),"RED") if self.detect(self.mask_r, min_area, (0, 0, 255), "RED") else (False, None)
        
        height, width, _ = self.image.shape
        self.height = height
        self.width = width

        self.distance_between(green_x, red_x, green_y, red_y,green_w, green_h, red_w, red_h)
        
        middle_frame = width // 2
        distance = middle_frame - self.middle_x

        text, data = self.json(green_x, green_y, green_w, green_h, "GREEN", red_x, red_y, red_w, red_h, "RED")

        text_lines = text.count('\n') + 1
        rect_height = text_lines * 20 + 10
        cv.rectangle(self.image, (10, 10), (200, 10 + rect_height), (0, 0, 0), -1)

        # Draw the text
        self.put_text_multiline(self.image, text, (15, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 20)
    
    def run_on_video(self, output_path):
        width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv.CAP_PROP_FPS)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                self.image = frame
                self.generate_masks()
                self.detect_buoys()
                out.write(self.image)
                cv.imshow("Processed Frame", self.image)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.video.release()
            out.release()
            cv.destroyAllWindows()

        return output_path
    

    
class CardinalBuoys(visionNav):
    def __init__(self, sift_images, video=None):
        super().__init__(video)
        self.sift_images = sift_images
        self.y_mask = None
        self.b_mask = None

    def turn_gray(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray_image
    

    def detect_cardinal(self,frame, img_buoy):

        sift = cv.SIFT_create(
            nfeatures=0,
            contrastThreshold=0.03,
            edgeThreshold=10,
            sigma=1.6,
        )
        bf = cv.BFMatcher()

        img_matches = None
        M = None
        gray_buoy = self.turn_gray(img_buoy)
        gray_frame = self.turn_gray(frame)

        kp1, des1 = sift.detectAndCompute(gray_buoy, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        if des2 is not None:
            
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

            if len(good_matches) > 10:
                
                img_matches = cv.drawMatches(img_buoy, kp1, gray_frame, kp2, good_matches, None, flags=2) # cv2.drawMatches

                # keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calculate homography
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        return img_matches, M
    
    def draw_buoy(self, frame, M, img_buoy, desc):
        # If there are enough matches
        if M is not None:
            h, w = img_buoy.shape[:2]

            # Define the four corners of the robot image
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            # Transform the points to locate the robot in the video frame
            dst = cv.perspectiveTransform(pts, M)

            # Draw a blue polygon around the detected robot in the frame
            frame = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
            cv.putText(frame, desc, (int(dst[0][0][0]), int(dst[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        return frame
    
    def run_on_video(self, output_path):
        width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv.CAP_PROP_FPS)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                self.image = frame

                img_matches_e, M_e = self.detect_cardinal(frame, self.sift_images[0])
                img_matches_w, M_w = self.detect_cardinal(frame, self.sift_images[1])

                frame =  self.draw_buoy(frame, M_e, self.sift_images[0], "EAST BUOY")
                frame =  self.draw_buoy(frame, M_w, self.sift_images[1], "WEST BUOY")

                self.generate_masks()
                self.detect_buoys()

                out.write(self.image)
                cv.imshow("Processed Frame", self.image)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.video.release()
            out.release()
            cv.destroyAllWindows()

        return output_path


########################################################################################