from ultralytics import YOLO
 
def main():
    model = YOLO('train/dior-r/lbgd/weights/best.pt')
    source = '/DIOR-R/test/images/'
    project = 'detect/dior-r'
    name = 'lbgd'
    batch = 1 # batch only set 1
    predict = model(source=source, save_conf=True, imgsz=800, save=True, device=1, batch=batch, workers=1, save_txt=False, project=project, name=name)

if __name__ == '__main__':
    main()
