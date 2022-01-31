import tensorflow as tf
import numpy as np
import logging
import cv2

class ImageOperator():
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def transform_images(self, x_train, size=416):
        """Perform processing on images

        Args:
            x_train: 
        """
        x_train = tf.image.resize(x_train, (size, size))
        x_train = x_train / 255
        return x_train

    def draw_outputs(self, img, outputs, class_names):
        boxes, objectness, classes, nums = outputs
        wh = np.flip(img.shape[0:2])
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            width = abs(x1y1[0]-x2y2[0])/wh
            height = abs(x1y1[1]-x2y2[1])/wh
            hw_ratio = height/width
            self.logger.write(logging.INFO, 'Detected object with height {} and hw ratio {}'.format(height, hw_ratio))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        return img

    def filter_detections(self, nums, classes, scores, boxes, targets):
        classes = classes[0][:nums[0]]
        scores = scores[0][:nums[0]]
        boxes = boxes[0][:nums[0]]
        filter_classes = np.nonzero(np.isin(classes, targets))
        filter_scores = np.nonzero(scores > 0.56)
        filter_indices = np.intersect1d(filter_classes, filter_scores)

        num_out = filter_indices.size
        detected_classes = classes[filter_indices]
        detected_classes = detected_classes.astype('int')
        detected_boxes = boxes[filter_indices]
        detected_scores = scores[filter_indices]
        classes = classes.astype('int')  # convert classes to ints

        def filter_boxes(class_id, box):
            if class_id != 0:
                return True
            x1y1 = tuple((np.array(box[0:2])))
            x2y2 = tuple((np.array(box[2:4])))
            width = abs(x1y1[0]-x2y2[0])
            height = abs(x1y1[1]-x2y2[1])
            hw_ratio = height/width
            return height > self.config['min_detection_height'] and hw_ratio > self.config['min_hw_ratio']

        res = np.array(list(map(filter_boxes, detected_classes, detected_boxes))).astype('bool')
        filter_indices = filter_indices[res]
        num_out = filter_indices.size
        detected_classes = classes[filter_indices]
        detected_boxes = boxes[filter_indices]
        detected_scores = scores[filter_indices]
        return num_out, classes, detected_classes, detected_boxes, detected_scores
