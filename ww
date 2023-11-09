import torch
from torchvision import models
import fastapi

# 加载模型
model_path = "./resnet18.pth"
model = torch.load(model_path)
model.eval()

# 定义 FastAPI 应用
app = fastapi.FastAPI()

# 定义图像分类路由
@app.post("/predict")
async def predict(image: bytes):
    # 解码图像
    image = torch.from_numpy(image.numpy())
    image = image.unsqueeze(0)
    image = image.float()

    # 进行预测
    outputs = model(image)
    _, preds = torch.max(outputs, 1)

    # 返回预测结果
    return {"label": classes[preds[0].item()]}

# 启动应用
if __name__ == "__main__":
    app.run(debug=True)
