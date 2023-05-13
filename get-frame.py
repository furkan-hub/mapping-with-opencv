import cv2

def extract_frames(video_path, output_path, frame_interval):
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_count % frame_interval == 0:
            output_file = f"{output_path}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)

        frame_count += 1

        for _ in range(frame_interval - 1):
            video.read()

    video.release()

video_path = "video4.mp4"  
output_path = "frames"  
frame_interval = 10
extract_frames(video_path, output_path, frame_interval)
