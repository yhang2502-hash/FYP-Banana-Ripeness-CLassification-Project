import cv2


def test_resolution(width, height, name):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Dell webcam

    if not cap.isOpened():
        print("Error: Cannot open Dell webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"\nTesting {name}...")
    print("Press Q to continue.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        text = f"{name} | Actual: {actual_width}x{actual_height}"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize only for display, not for capture
        display_frame = cv2.resize(frame, (1280, 720))

        cv2.imshow("Resolution Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_resolution(2560, 1440, "2K QHD")
    test_resolution(1920, 1080, "Full HD")
    test_resolution(1280, 720, "HD")

# The Dell WB3023 webcam was successfully tested in Python OpenCV at HD
# (1280×720), Full HD (1920×1080), and 2K QHD (2560×1440). Although 2K provided
# higher image detail, Full HD was selected for real-time experiments due to its more
# practical balance between image quality and processing efficiency.
