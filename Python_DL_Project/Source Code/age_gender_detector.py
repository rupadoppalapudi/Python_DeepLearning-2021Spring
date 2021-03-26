# importing required modules
import cv2 as cv
import time

# extracting face of the person from image
def extract_face(net, image, conf_threshold=0.7):
    frame = image.copy()
    f_height = frame.shape[0]  # frame height
    f_width = frame.shape[1]   # frame width

    # deep neural network library
    # blobfromimage method to set scalefactor, size, mean, swapRB, crop, ddepth of the image
    blob_img = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * f_width)
            y1 = int(detections[0, 0, i, 4] * f_height)
            x2 = int(detections[0, 0, i, 5] * f_width)
            y2 = int(detections[0, 0, i, 6] * f_height)
            b_boxes.append([x1, y1, x2, y2])
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(f_height / 150)), 8)
    return frame, b_boxes

face_Proto = "models/opencv_face_detector.pbtxt"  # protocol buffer
face_Model = "models/opencv_face_detector_uint8.pb"

age_Proto = "models/age_deploy.prototxt" # deploys age model
age_Model = "models/age_net.caffemodel"  # defines the the internal states of layer parameters

gender_Proto = "models/gender_deploy.prototxt" # deploys gender model
gender_Model = "models/gender_net.caffemodel"  # defines the the internal states of layer parameters

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# set age and gender category
age_category = ['(0-3)', '(4-7)', '(8-15)', '(16-23)', '(24-33)', '(34-45)', '(46-54)', '(55-100)']
gender_category = ['Male', 'Female']

# loading the network - face, age and gender network
face_network = cv.dnn.readNet(face_Model, face_Proto)
age_network = cv.dnn.readNet(age_Model, age_Proto)
gender_network = cv.dnn.readNet(gender_Model, gender_Proto)

padding = 20

# age and gender detection of the person based on the image
def age_gender_detector(image):
    # reading image
    t = time.time()
    frame_face, b_boxes = extract_face(face_network, image)
    for bbox in b_boxes:
        face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_network.setInput(blob)
        gender_pred = gender_network.forward()
        gender = gender_category[gender_pred[0].argmax()]

        # Display detected gender of the input image on to console
        print("Gender Output: {}, conf = {:f}".format(gender, gender_pred[0].max()))

        age_network.setInput(blob)
        age_pred = age_network.forward()
        age = age_category[age_pred[0].argmax()]

        # Display detected age of the input image on to console
        print("Age Output : {}".format(age_pred))
        print("Age : {}, conf = {:f}".format(age, age_pred[0].max()))

        frame_label = "{},{}".format(age, gender)
        font = cv.FONT_ITALIC
        color = (0, 0, 255)

        # putText renders the specified text string in the image
        cv.putText(frame_face, frame_label, (bbox[0], bbox[1] - 10), font, 0.8, color, 2,
                   cv.FILLED)
    return frame_face

# displaying the output image along with age and gender indication
input_image = cv.imread("pp.jpg")
output_image = age_gender_detector(input_image)
cv.imshow("image", output_image)
cv.waitKey(0)