import cv2


class VideoReader:

    def __init__(self, path: str):

        self.path = path

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {path}"
            )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        self.width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        self.frame_count = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if self.fps <= 0:
            raise RuntimeError(
                "Invalid video FPS."
            )

    @property
    def duration(self):
        return self.frame_count / self.fps

    def read(self):

        return self.cap.read()

    def release(self):

        self.cap.release()

    def frame_time(self, frame_number):

        return frame_number / self.fps