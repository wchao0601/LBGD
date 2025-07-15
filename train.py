import os  
from ultralytics import YOLO

os.environ["WANDB_API_KEY"] = "edf43c0b228beb9fec06dccca946a950b1c53dc0"
os.environ["WANDB_MODE"] = "offline"

def main():
    t_weight = "weights/dior-r/teacher.pt"  # teacher weights
    model_t = YOLO(t_weight)
    model_t.model.model[-1].set_Distillation = True

    data = "'ultralytics/cfg/datasets/dior-r.yaml'" # dataset setting
    yaml = "ultralytics/cfg/models/v8/yolov8n-obb.yaml" # student yaml
    project = "train/dior-r/"
    name = "lbgd"
    model_s = YOLO(yaml)
    model_s.train(data=data, epochs=100, imgsz=800, batch=16, device=1, workers=4, project=project, name=name, model_t=model_t.model)
if __name__ == '__main__':
    main()




 
