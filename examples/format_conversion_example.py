"""Example: Using COCO format as standard exchange format.

This example demonstrates how CV-Ninja uses COCO format internally
to standardize prediction results from different APIs.
"""

from cv_ninja.predictors.formats import FormDataFormatConverter


def main():
    # Example API response in FormData format
    formdata_response = {
        'dataset_id': '1377606572385112064',
        'result': [
            {'RegisterMatrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {
                'Box': {'X': 1148, 'Y': 689, 'Width': 45, 'Height': 154, 'Angle': 0},
                'Score': 0.8662109375,
                'label': 'jiaza'
            },
            {
                'Box': {'X': 200, 'Y': 300, 'Width': 60, 'Height': 80, 'Angle': 0},
                'Score': 0.9234375,
                'label': 'defect'
            }
        ],
        'image_width': 1920,
        'image_height': 1080
    }

    # Create converter
    converter = FormDataFormatConverter()

    # Convert FormData to COCO format
    print("Converting FormData format to COCO format...")
    coco_data = converter.to_coco(formdata_response, image_id=1, file_name="example.jpg")

    print("\nCOCO Format Result:")
    print(f"Images: {coco_data['images']}")
    print(f"Categories: {coco_data['categories']}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    print(f"Annotations: {coco_data['annotations']}")
    print(f"Metadata: {coco_data.get('metadata', {})}")

    # Convert back to FormData format
    print("\n" + "="*50)
    print("Converting COCO format back to FormData format...")
    formdata_result = converter.from_coco(coco_data)

    print("\nFormData Format Result:")
    print(f"Dataset ID: {formdata_result['dataset_id']}")
    print(f"Image size: {formdata_result['image_width']}x{formdata_result['image_height']}")
    print(f"Number of detections: {len(formdata_result['result']) - 1}")  # -1 for RegisterMatrix
    print(f"Result: {formdata_result['result']}")

    # Verify round-trip conversion
    print("\n" + "="*50)
    print("Verification: Round-trip conversion test")

    # Check if the original detections match
    original_detections = [item for item in formdata_response['result'] if 'Box' in item]
    converted_detections = [item for item in formdata_result['result'] if 'Box' in item]

    if len(original_detections) == len(converted_detections):
        print(f"✓ Detection count matches: {len(original_detections)} detections")
    else:
        print(f"✗ Detection count mismatch!")

    # Check bounding boxes
    for i, (orig, conv) in enumerate(zip(original_detections, converted_detections)):
        orig_box = orig['Box']
        conv_box = conv['Box']
        if (orig_box['X'] == conv_box['X'] and
            orig_box['Y'] == conv_box['Y'] and
            orig_box['Width'] == conv_box['Width'] and
            orig_box['Height'] == conv_box['Height']):
            print(f"✓ Detection {i+1} bbox matches")
        else:
            print(f"✗ Detection {i+1} bbox mismatch!")


if __name__ == "__main__":
    main()
