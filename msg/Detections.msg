# Simulated camera detection, in camera frame
# TODO: having detections as vectors makes no damn sense. 
# just put them as XYZ coordinates in a frame that's specified by the header
Header header

uint16[] target_ids
uint16[] classes
string[] class_labels
float32[] headings # heading of the ship
geometry_msgs/Point[] target_camera_vectors  # projected camera ray onto the ocean plane, in camera frame