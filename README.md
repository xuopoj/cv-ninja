# CV Ninja

一个CV工具集，用来做图片标注的格式转换，标注切分和合并，图片的切分，图片推理，推理结果评测等。

## 功能特性

### 1. 标注格式转换

支持的图片标注格式：
- LabelMe
- LabelStudio
- Pascal VOC
- COCO
- 私有推理结果格式

### 2. 图片和标注切分/合并

某些场景比如钢板表面检查场景下，原始的图片比较大（4096*3000），需要一些切分和合并操作：
- 训练场景下，人工标注基于原始图片，在训练之前需要对图片和标注进行切分
- 推理场景下，需要先对原始图片进行切分，再将标注重新拼接，返回基于原始图片的标注信息

### 3. 模型推理 API 客户端

通过 HTTP API 调用外部 CV 模型进行推理，支持：
- 多种认证方式（API Key、IAM Token）
- 多种数据传输格式（Form-Data、Binary Upload）
- 单张和批量图片处理
- 多种输出格式（LabelStudio、VOC、COCO）

## 安装

从源码安装：

```bash
git clone https://github.com/yourusername/cv-ninja.git
cd cv-ninja
pip install -e .
```

## 配置

### 环境变量配置

创建 `.env` 文件配置 API 访问参数：

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

`.env` 文件内容：

```bash
# 推理 API 端点 URL
PREDICTION_API_URL=https://api.example.com/predict

# 认证方式 1: API Key (Bearer token)
PREDICTION_API_KEY=your_api_key_here

# 认证方式 2: IAM Token (X-Auth-Token)
PREDICTION_IAM_URL=https://iam-apigateway-proxy.cloud.nisco.cn/v3/auth/tokens
PREDICTION_USERNAME=your_username
PREDICTION_PASSWORD=your_password
PREDICTION_IAM_DOMAIN=your_domain
PREDICTION_IAM_PROJECT=your_project
```

### 混合配置（推荐用于多 endpoint 测试）

CV-Ninja 支持 **`.env` + `endpoints.yaml`** 混合配置，方便管理和测试多个 API endpoint：

**配置优先级**（从高到低）：
1. CLI 选项
2. YAML Profile 配置
3. `.env` 环境变量
4. 默认值

**创建 `endpoints.yaml` 文件**：

```bash
cp endpoints.yaml.example endpoints.yaml
```

`endpoints.yaml` 示例：

```yaml
endpoints:
  prod:
    api_url: https://api.prod.example.com
    mode: binary
    endpoint: /upload
    iam_url: https://iam.prod.example.com/v3/auth/tokens

  test:
    api_url: https://api.test.example.com/predict
    mode: formdata

  local:
    api_url: http://localhost:5000
    mode: binary
    endpoint: /upload
```

**使用方式**：

```bash
# 使用 prod profile
cv-ninja predict image test.jpg --profile prod

# 切换到 test profile
cv-ninja predict image test.jpg --profile test

# 临时覆盖配置
cv-ninja predict image test.jpg --profile prod --api-url https://override.com
```

**优点**：
- 凭证（用户名/密码）集中在 `.env`（不提交到 git）
- Endpoint 配置在 YAML（可以提交到 git）
- 快速切换不同 endpoint 进行对比测试

## 使用方法

### 标注格式转换

#### Pascal VOC 转 LabelStudio

```bash
cv-ninja convert voc-to-labelstudio dataset/pascal_voc \
  -o output.json \
  -p "/data/local-files/?d=orig/data/surface_defect/"
```

### 模型推理

#### 1. 单张图片推理

**使用 .env 配置：**

```bash
cv-ninja predict image path/to/image.jpg
```

**使用命令行参数覆盖配置：**

```bash
cv-ninja predict image path/to/image.jpg \
  -u https://api.example.com/predict \
  --api-key YOUR_API_KEY \
  -c 0.5 \
  -o predictions.json \
  -f labelstudio
```

**使用 IAM 认证：**

```bash
cv-ninja predict image path/to/image.jpg \
  --iam-url https://iam.example.com/v3/auth/tokens \
  --username user \
  --password pass \
  --iam-domain default \
  --iam-project my-project
```

**使用二进制上传模式（带查询参数）：**

```bash
cv-ninja predict image path/to/image.jpg \
  --binary \
  --endpoint /upload \
  --params '{"Station_id": "station123", "extra_param": "value"}'
```

**使用 Profile 配置：**

```bash
# 使用预定义的 endpoint 配置
cv-ninja predict image path/to/image.jpg --profile prod

# 在不同 endpoint 之间快速切换进行测试
cv-ninja predict image path/to/image.jpg --profile test
cv-ninja predict image path/to/image.jpg --profile staging
```

**大图自动切分（Tiling）：**

```bash
# 启用自动切分（适合大于 1386x1516 的图片）
cv-ninja predict image large_image.jpg --tile

# 自定义切分大小和重叠区域
cv-ninja predict image large_image.jpg --tile --tile-size 2048x2048 --tile-overlap 64

# 配合 profile 使用
cv-ninja predict image large_image.jpg --profile prod --tile -v
```

#### 2. 批量图片推理

**基本用法：**

```bash
cv-ninja predict batch ./images
```

**递归处理子目录：**

```bash
cv-ninja predict batch ./images -r
```

**指定输出格式：**

```bash
# LabelStudio 格式（默认）
cv-ninja predict batch ./images -f labelstudio -o predictions.json

# Pascal VOC 格式（每张图片一个 XML 文件）
cv-ninja predict batch ./images -f voc -o ./output/

# COCO 格式
cv-ninja predict batch ./images -f coco -o coco_results.json
```

**使用自定义 .env 文件：**

```bash
cv-ninja predict batch ./images --env-file /path/to/.env
```

**详细输出：**

```bash
cv-ninja predict batch ./images -v
```

#### 3. 参数说明

**通用参数：**
- `-u, --api-url`: API 端点 URL
- `-c, --confidence`: 置信度阈值（0-1，默认 0.5）
- `-o, --output`: 输出文件路径
- `-f, --format`: 输出格式（labelstudio/voc/coco）
- `-v, --verbose`: 详细输出

**认证参数：**
- `--api-key`: API Key（Bearer token 认证）
- `--iam-url`: IAM 服务 URL
- `--username`: IAM 用户名
- `--password`: IAM 密码
- `--iam-domain`: IAM 域名
- `--iam-project`: IAM 项目名称

**二进制上传模式参数：**
- `--binary`: 启用二进制上传模式
- `--endpoint`: API 端点路径（默认 /upload）
- `--params`: JSON 格式的查询参数

**批量处理参数：**
- `-r, --recursive`: 递归处理子目录

**Profile 配置参数：**
- `--profile`: Profile 名称（从 YAML 配置加载，如 prod、test、staging）
- `--config-file`: YAML 配置文件路径（默认 cv-ninja.yaml 或 endpoints.yaml）

**图片切分参数：**
- `--tile`: 启用自动图片切分（适合超过 1386x1516 的大图）
- `--tile-size`: 切分尺寸，格式 WIDTHxHEIGHT（默认 1386x1516）
- `--tile-overlap`: 切片重叠像素数（默认 32px，用于边界目标检测）

### 大图切分处理

当图片尺寸超过模型输入限制时（如 4096x3000），可以启用自动切分：

**工作原理：**
1. 自动检测图片是否超过设定尺寸
2. 将大图切分为多个重叠的小图块
3. 对每个图块分别进行推理
4. 合并所有图块的检测结果
5. 使用 NMS（非极大值抑制）去除重复检测

**使用场景：**
- 钢板表面检测（原始图片 4096x3000）
- 卫星图像分析
- 大幅面文档处理
- 高分辨率工业检测

**示例：**

```bash
# 处理 4096x3000 的大图
cv-ninja predict image steel_plate.jpg \
  --tile \
  --tile-size 1386x1516 \
  --tile-overlap 32 \
  --profile prod \
  -v

# 批量处理大图
cv-ninja predict batch ./large_images \
  --tile \
  --profile prod \
  -v
```

**性能说明：**
- 切分会增加推理时间（图块数量 = (width/stride) × (height/stride)）
- 32px 重叠确保边界目标被正确检测
- NMS 阈值默认 0.5，可调整以控制去重程度

### Python API 使用

#### 使用 Form-Data 上传（多部分表单）

```python
from cv_ninja.predictors.base import FormDataPredictor
from cv_ninja.predictors.auth import APIKeyAuth

# 创建认证处理器
auth = APIKeyAuth("your_api_key")

# 创建预测客户端
client = FormDataPredictor(
    api_url="https://api.example.com/predict",
    auth_handler=auth
)

# 单张图片推理
result = client.predict_from_file(
    image_path="path/to/image.jpg",
    confidence_threshold=0.5
)

print(result)
```

#### 使用二进制上传

```python
from cv_ninja.predictors.base import BinaryPredictor
from cv_ninja.predictors.auth import IAMTokenAuth

# 创建 IAM 认证处理器
auth = IAMTokenAuth(
    iam_url="https://iam.example.com/v3/auth/tokens",
    username="user",
    password="pass",
    domain="default",
    project="my-project"
)

# 创建二进制预测客户端
client = BinaryPredictor(
    api_url="https://api.example.com",
    auth_handler=auth,
    endpoint="/upload"
)

# 推理
result = client.predict_from_file(
    image_path="path/to/image.jpg",
    params={"Station_id": "station123"}
)

print(result)
```

#### 从字节数据推理

```python
# 读取图片字节
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# 使用字节数据推理
result = client.predict_from_bytes(
    image_data=image_bytes,
    params={"Station_id": "station123"}
)
```

#### 使用图片切分

```python
from cv_ninja.predictors.base import FormDataPredictor
from cv_ninja.predictors.auth import IAMTokenAuth
from cv_ninja.predictors.tiling import ImageTiler
from cv_ninja.predictors.formats import FormDataFormatConverter
from PIL import Image

# 创建预测客户端（slim，只做预测）
auth = IAMTokenAuth(iam_url, username, password, domain, project)
predictor = FormDataPredictor(api_url="https://api.example.com", auth_handler=auth)

# 创建格式转换器和图片切分器
converter = FormDataFormatConverter()
tiler = ImageTiler(
    tile_size=(1386, 1516),      # 切片大小
    overlap=32                    # 重叠像素
)

# 加载图片
img = Image.open("large_image.jpg")

# 使用 tiler 处理大图（tiler 会调用 predictor）
if tiler.needs_tiling(img):
    result = tiler.predict_tiled(
        predictor,
        converter,
        img,
        model_name="default",
        confidence_threshold=0.5
    )
else:
    # 直接预测小图
    result = predictor.predict_from_file("large_image.jpg", "default", 0.5)

# 结果会自动合并
print(f"Total detections: {result['total_detections']}")
print(f"Number of tiles: {result['num_tiles']}")
```

## 架构设计

### 关注点分离 (Separation of Concerns)

CV-Ninja 遵循单一职责原则，将不同功能解耦到独立组件：

**核心组件：**
1. **Predictor（预测器）**- 专注于 API 调用和预测
   - 轻量级，只负责发送请求和接收响应
   - 不包含切分、合并等复杂逻辑
   - 示例：`FormDataPredictor`, `BinaryPredictor`

2. **FormatConverter（格式转换器）**- 标准化不同 API 格式
   - 将 API 格式转换为 COCO 标准格式
   - 将 COCO 格式转换回 API 格式
   - 示例：`FormDataFormatConverter`, `BinaryFormatConverter`

3. **ImageTiler（图片切分器）**- 编排大图处理流程
   - 切分图片为小块
   - 调用 predictor 处理每个切片
   - 使用 converter 标准化格式
   - 合并结果并应用 NMS

**架构流程：**
```
用户代码
  ↓
ImageTiler.predict_tiled(predictor, converter, image)
  ↓
切分图片 → 对每个切片调用 → Predictor.predict_from_bytes()
                                ↓
                            API 响应（原始格式）
  ↓
Converter.to_coco() → COCO 标准格式 → 合并坐标 + NMS
  ↓
Converter.from_coco() → API 格式
  ↓
返回给用户
```

**优势：**
- **可测试性**：每个组件独立测试
- **可复用性**：Tiler 可用于任何 Predictor
- **可扩展性**：新增 API 只需实现 Predictor + Converter
- **清晰性**：职责明确，易于理解和维护

### COCO 格式作为标准交换格式

CV-Ninja 使用 **COCO 格式**作为内部标准交换格式，确保不同 API 预测结果的一致性和互操作性。

**设计理念：**
- 每个预测器（FormDataPredictor、BinaryPredictor）使用各自的 API 格式与外部服务通信
- 内部使用统一的 COCO 格式处理标注数据（切分、合并、NMS 等）
- 通过格式转换器（FormatConverter）在 API 格式和 COCO 格式之间转换

**COCO 格式示例：**
```json
{
    "images": [
        {"id": 1, "width": 1920, "height": 1080, "file_name": "image.jpg"}
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1148, 689, 45, 154],
            "area": 6930,
            "score": 0.866,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 1, "name": "jiaza"}
    ],
    "metadata": {
        "dataset_id": "1377606572385112064",
        "num_tiles": 4,
        "total_detections": 25
    }
}
```

### 使用格式转换器

```python
from cv_ninja.predictors.formats import FormDataFormatConverter

# 创建转换器
converter = FormDataFormatConverter()

# API 响应（FormData 格式）
api_response = {
    'dataset_id': '1377606572385112064',
    'result': [
        {'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {
            'Box': {'X': 1148, 'Y': 689, 'Width': 45, 'Height': 154, 'Angle': 0},
            'Score': 0.8662109375,
            'label': 'jiaza'
        }
    ],
    'image_width': 1920,
    'image_height': 1080
}

# 转换为 COCO 格式
coco_data = converter.to_coco(api_response, image_id=1)

# 处理 COCO 数据（如 NMS、坐标变换等）
# ...

# 转换回 API 格式
formdata_result = converter.from_coco(coco_data)
```

## 开发

### 添加新的传输格式

类似于 `AuthHandler` 的设计，可以轻松扩展新的传输格式：

1. **创建新的预测器类：**

```python
from cv_ninja.predictors.base import PredictionClient

class Base64Predictor(PredictionClient):
    """发送 base64 编码的 JSON 格式"""

    def predict_from_file(self, image_path: str, **kwargs):
        # 实现 base64 编码逻辑
        pass

    def predict_from_bytes(self, image_data: bytes, **kwargs):
        # 实现 base64 编码逻辑
        pass
```

2. **创建格式转换器：**

```python
from cv_ninja.predictors.formats import FormatConverter

class Base64FormatConverter(FormatConverter):
    """Base64 格式转换器"""

    def to_coco(self, predictions: Dict[str, Any], image_id: int = 1) -> Dict[str, Any]:
        """将 Base64 预测格式转换为 COCO 格式"""
        # 实现转换逻辑
        pass

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """将 COCO 格式转换回 Base64 预测格式"""
        # 实现转换逻辑
        pass
```

3. **使用 Tiler 编排预测流程：**

```python
from cv_ninja.predictors.tiling import ImageTiler
from PIL import Image

# 创建预测器和转换器
predictor = Base64Predictor(api_url="...", auth_handler=auth)
converter = Base64FormatConverter()

# 创建 Tiler
tiler = ImageTiler(tile_size=(1386, 1516), overlap=32)

# 加载图片
img = Image.open("large_image.jpg")

# Tiler 会调用 predictor 并使用 converter 标准化格式
if tiler.needs_tiling(img):
    result = tiler.predict_tiled(predictor, converter, img, **predict_kwargs)
else:
    result = predictor.predict_from_file("large_image.jpg", **predict_kwargs)
```

**关键点：**
- Predictor 保持轻量级，只负责 API 调用
- Converter 负责格式转换
- Tiler 负责编排整个流程（切分 → 预测 → 合并）
- 三者解耦，独立测试和扩展

