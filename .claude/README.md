# CV-Ninja Development Summary

## Project Overview
CV-Ninja is a professional CLI tool for computer vision model prediction and annotation format conversion. It provides seamless integration with external prediction APIs and supports multiple output formats including Label Studio, COCO, and Pascal VOC.

## Recent Changes (Latest Session)

### 1. Parameter Cleanup
**Removed unnecessary parameters across the codebase:**
- `model_name`: Removed from CLI commands, input types, and predictor classes
- `confidence_threshold`: Removed from all prediction interfaces

**Rationale**: These parameters were not needed for the core prediction workflow and added unnecessary complexity to the API surface.

**Files Modified**:
- `src/cv_ninja/cli/commands/predict_commands.py`
- `src/cv_ninja/predictors/base.py`
- `src/cv_ninja/predictors/input_types.py`
- `src/cv_ninja/predictors/http_handler.py`
- `README.md`

### 2. Filename Metadata Extraction
**Added intelligent filename parsing to extract validation metadata:**

**New Fields**:
- `review_label`: FN (False Negative) or FP (False Positive)
- `target_class`: Expected defect type (e.g., jieba, maobian, huashang)

**Filename Format**: `{REVIEW_LABEL}_{TARGET_CLASS}_rest_of_name.ext`

**Examples**:
- `FN_jieba_image001.jpg` → review_label="FN", target_class="jieba"
- `FP_maobian_test.png` → review_label="FP", target_class="maobian"
- `normal_image.jpg` → no metadata extracted

**Implementation**:
- Added `_parse_filename_metadata()` static method to `PredictionOutputFormatter`
- Metadata automatically added to Label Studio output when present
- Only filename (not full path) is used in outputs

**Files Modified**:
- `src/cv_ninja/predictors/output_formatter.py`
- `src/cv_ninja/predictors/base.py`

### 3. URL Prefix Support
**Added `--prefix` option to prediction commands:**

**Purpose**: Allows prefixing image paths for Label Studio local file serving

**Example Usage**:
```bash
cv-ninja predict image image.jpg --prefix "/data/local-files/?d=images/"
cv-ninja predict batch ./images --prefix "/data/local-files/?d=dataset/"
```

**Output**:
```json
{
  "data": {
    "image": "/data/local-files/?d=images/FN_jieba_image001.jpg",
    "review_label": "FN",
    "target_class": "jieba"
  }
}
```

**Files Modified**:
- `src/cv_ninja/cli/commands/predict_commands.py`

### 4. Bug Fixes
- Fixed COCO format reference: Changed `result.get("detections")` to `result.get("annotations")`
- Removed undefined `confidence` variable references
- Updated all format converter calls to use consistent interfaces

## Architecture

### Prediction Flow
1. **CLI Layer** (`predict_commands.py`): Handles user input and orchestrates prediction
2. **Predictor Layer** (`base.py`): Manages API communication (FormData/Binary modes)
3. **Tiling Layer** (`tiling.py`): Splits large images and combines results
4. **Format Layer** (`formats.py`): Converts API responses to COCO format
5. **Output Layer** (`output_formatter.py`): Converts to Label Studio/VOC/COCO formats

### Key Components

**Predictors**:
- `FormDataPredictor`: Multipart form-data requests (most common)
- `BinaryPredictor`: Raw binary uploads with query parameters

**Format Converters**:
- `FormDataFormatConverter`: Converts proprietary API format to COCO
- `BinaryFormatConverter`: Converts binary API responses to COCO

**Output Formatters**:
- `to_labelstudio()`: Label Studio JSON format
- `to_voc()`: Pascal VOC XML format
- `to_coco()`: COCO JSON format

**Tiling**:
- `ImageTiler`: Handles large image splitting and NMS combination
- Default tile size: 1386x1516 with 32px overlap
- Automatic NMS to remove duplicate detections at boundaries

## Configuration

### Authentication
Supports two auth methods:
1. **API Key**: Bearer token authentication
2. **IAM Token**: OpenStack-style token authentication

### Configuration Sources (Priority Order)
1. CLI arguments (highest)
2. YAML profile configuration
3. .env file variables (lowest)

### YAML Profile Structure
```yaml
profiles:
  prod:
    api_url: "https://prod-api.example.com"
    mode: "formdata"
  test:
    api_url: "https://test-api.example.com"
    mode: "binary"
    endpoint: "/v2/predict"
```

## Testing

To test the filename parsing:
```python
from src.cv_ninja.predictors.output_formatter import PredictionOutputFormatter

filename = "FN_jieba_image001.jpg"
basename, review_label, target_class = PredictionOutputFormatter._parse_filename_metadata(filename)
print(f"basename={basename}, review={review_label}, target={target_class}")
# Output: basename=FN_jieba_image001.jpg, review=FN, target=jieba
```

## Future Improvements

Potential enhancements to consider:
1. Add support for additional review labels beyond FN/FP (e.g., TP, TN)
2. Support multiple target classes per image
3. Add confidence score filtering in output formatters
4. Implement batch processing progress bars
5. Add retry logic for failed API calls
6. Support for rotated bounding boxes in VOC output

## Development Guidelines

1. **Keep predictors simple**: Focus on API communication only
2. **Use COCO as interchange format**: All conversions go through COCO
3. **Validate early**: Use input_types for validation at CLI boundary
4. **No business logic in formatters**: Pure transformation only
5. **Test with real API responses**: Use debug output to verify formats

## Dependencies

Core dependencies:
- `click`: CLI framework
- `requests`: HTTP client
- `Pillow`: Image processing
- `pyyaml`: Configuration files
- `python-dotenv`: Environment variables

## Notes

- Python version requirement updated to >=3.10 (from 3.12)
- All debug print statements should be removed in production
- Image tiling is optional but recommended for large images
- NMS IoU threshold is configurable (default: 0.5)
