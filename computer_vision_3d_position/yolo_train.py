from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/home/leanhchien/computer_vision/cuoi_ky/data_train/data_train.yaml', epochs=100, imgsz=320, batch = 70)