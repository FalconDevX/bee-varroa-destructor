from ultralytics import YOLO

# załaduj model
model = YOLO("last.pt")

# predykcja na jednym obrazie
results = model("test_bee.jpg", show=True)

# zapisz wynik do pliku
results[0].save(filename="output.jpg", conf=0.05)
