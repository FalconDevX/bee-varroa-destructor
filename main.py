from ultralytics import YOLO

model = YOLO("last.pt")

results = model("test_bee.jpg", show=True)

results[0].save(filename="output.jpg", conf=0.05)
