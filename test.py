from ultralytics import YOLO
 
def main():
    data = "ultralytics/cfg/datasets/dior-r.yaml"
    model = YOLO("train/dior-r/lbgd/weights/best.pt")
    metrics = model.val(split="test", imgsz=800, device=0, batch=1, workers=4, project="test/dior-r", name="lbgd")
    map50 = metrics.box.map50
    map75 = metrics.box.map75
    map = metrics.box.map
    metrics.box.maps
    print(map50, map75, map)
if __name__ == '__main__':
    main()
