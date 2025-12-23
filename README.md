# CV Ninja

一个CV工具集，用来做图片标注的格式转换，标注切分和合并，图片的切分，图片推理，推理结果评测等。

## 标注格式转换

需要支持的图片标注格式：
1. LabelMe;
2. LabelStudio;
3. Pascal VOC;
4. 一种私有的推理结果格式;

某些场景比如钢板表面检查场景下，原始的图片比较大（4096*3000），需要一些切分和合并操作：
1. 训练场景下，人工标注基于原始图片，在训练之前需要对图片和标注进行切分；
2. 推理场景下，需要先对原始图片进行切分，再将标注重新拼接，返回基于原始图片的标注信息。

## Usage

### Convert Pascal VOC to LabelStudio

```
python voc_to_labelstudio.py dataset/pascal_voc -o output.json -p "/data/local-files/?d=orig/data/surface_defect/"
```
