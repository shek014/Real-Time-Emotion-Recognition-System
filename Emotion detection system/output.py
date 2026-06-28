from ultralytics import YOLO

def main():
    model = YOLO("best.pt")  # or last.pt
    model.predict(
        source=0,       # webcam
        show=True,      # show the window
        conf=0.25,       # confidence threshold
        device=0        # GPU (use 'cpu' if no GPU)
    )

if __name__ == "__main__":
    main()
