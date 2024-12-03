from configparser import ConfigParser
from detection_model import Detection
from tracker.byte_tracker import BYTETracker
import cv2
import numpy as np

# SETUP
config = ConfigParser()
config.read('tracker.cfg')
cap = cv2.VideoCapture(config.get('video', 'video_path'))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_tracking = cv2.VideoWriter(config.get('video', 'video_out_tracking'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(config.get('video', 'video_out_detect'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# test_size = [416, 416]
aspect_ratio_thresh = 0.6
min_box_area = 100

class_name = ['ball', 'goalkeeper', 'player', 'referee']
class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def main():
    weights = "weights/best.pt"
    model = Detection(weights)
    args = TrackerArgs(
        track_thresh=config.getfloat('Tracker', 'track_thresh'),
        track_buffer=config.getint('Tracker', 'track_buffer'),
        match_thresh=config.getfloat('Tracker', 'match_thresh'),
        fuse_score=config.getboolean('Tracker', 'fuse_score')
    )
    tracker = BYTETracker(
        args)
    frame_id = 0
    temp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        results = model.detect(frame)
        detections = []
        frame_detected = frame.copy()
        for result in results:
            for boxes in result.boxes:
                label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                print(label, conf, bbox)
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)
                detections.append([x1, y1, x2, y2, conf])
                fps = ' FPS: ' + str(fps) + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
                cv2.putText(frame_detected, fps, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame_detected, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_detected, f"{class_name[class_id]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out_detect.write(frame_detected)

        
        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        
        # Update tracker with detections format, frame size, and test size
        online_targets = tracker.update(detections, [height, width], [height, width])

        # Draw tracked objects
        frame_tracked = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
            # if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            if tlwh[2] * tlwh[3] > min_box_area:
                # save results
                temp.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
                # Draw the bounding box
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame_tracked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_tracked, f'ID: {tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # frame_id += 1
        # if frame_id == 1:
        #     break

        print(temp)
        # Write and display the frame

        out_tracking.write(frame_tracked)
        cv2.imshow('frame', frame_tracked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_detect.release()
    out_tracking.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()