import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the TensorFlow model - keep it chill and make sure the paths are like zen gardens
detection_model = tf.saved_model.load('path_to_your_chill_model')
category_index = label_map_util.create_category_index_from_labelmap('path_to_the_munchies_labels.pbtxt')

# Function to turn images into stoner cartoons
def cartoonify_image(image):
    # Convert to grayscale, like old-school TVs
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median blur, like your vision at 4:20
    gray_blurred = cv2.medianBlur(gray, 5)
    # Edge mask, for that comic book feel
    edges = cv2.adaptiveThreshold(gray_blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    # Back to color, cause life's too short for just gray
    color = cv2.bilateralFilter(image, 9, 300, 300)
    # Mix edges and color for the final toon look
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Function to run inference and spot 'TheFanger' in the wild
def run_inference(image_np, detection_model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = detection_model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # Unbatchify the outputs
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # Ensure detection classes are integers, not floaty things
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Visualize the results like a psychedelic trip
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    return image_np

# Fire up the camera and let's get this party started
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Cartoonify the frame for that trippy effect
        cartoon_frame = cartoonify_image(frame)

        # Run inference and look for 'TheFanger'
        output_frame = run_inference(cartoon_frame, detection_model)

        # Show the tooned-up, Fangtuned vision
        cv2.imshow('Fangtune AI Vision', output_frame)

        # Press 'q' to bail, cause all good things must come to an end
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
