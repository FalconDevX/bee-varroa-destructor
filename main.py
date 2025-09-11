from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test_bee.jpg", show=True)

results[0].save(filename="output.jpg", conf=0.2)
