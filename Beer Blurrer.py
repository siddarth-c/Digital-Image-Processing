"""
Beer Blurrer

COM503 : Digital Image Processing, IIITDM, Kancheepuram

Authors:
A Navaas Roshan
B Gokulapriyan
C Siddarth
D Balajee

"""


import time
import os
import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def blur_beer(org_img, config_path, weights_path, labels_path):

    """
    Blurs the detected alcohol bottles
    Inputs: Input image, Configurateion file path, Weights of YOLO model, Label file of YOLO
    Output: None
    """
    try:
        # PART 1 - Detection of bottles

        my_bar = st.progress(0)

        CONFIDENCE = 0.5
        SCORE_THRESHOLD = 0.5
        IOU_THRESHOLD = 0.5
        extra = 20
        # the neural network configuration
        config_path = config_path

        # the YOLO net weights file
        weights_path = weights_path

        # loading all the class labels (objects)
        labels = open(labels_path).read().strip().split("\n")
        # generating colors for each object for later plotting
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        # load the YOLO network
        net = cv.dnn.readNetFromDarknet(config_path, weights_path)

        # path_name = img_path

        padded = org_img.copy()
        padded = cv.copyMakeBorder(padded, 20, 20, 20, 20, cv.BORDER_REFLECT)

        image = padded.copy()
        h, w = image.shape[:2]

        # create 4D blob
        blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        # sets the blob as the input of the network
        net.setInput(blob)

        # get all the layer names
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # feed forward (inference) and get the network output
        # measure how much it took in seconds
        start = time.perf_counter()
        layer_outputs = net.forward(ln)
        time_took = time.perf_counter() - start

        font_scale = 1
        thickness = 1
        boxes, confidences, class_ids = [], [], []

        # loop over each of the layer outputs
        for output in layer_outputs:

            # loop over each of the object detections
            for detection in output:

                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # discard out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:

                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        my_bar.progress(10)

        # loop over the indexes we are keeping
        for i in range(len(boxes)):

            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv.FILLED)

            # add opacity (transparency to the box)
            image = cv.addWeighted(overlay, 0.6, image, 0.4, 0)

            # now put the text (label: confidence %)
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        my_bar.progress(15)
        tl_cord = []
        br_cord = []

        # perform the non maximum suppression given the scores defined before
        idxs = cv.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():

                # extract the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw a bounding box rectangle and label on the image
                tl_cord.append((x,y))
                br_cord.append((x+w, y+h))
                color = [int(c) for c in colors[class_ids[i]]]
                cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv.FILLED)

                # add opacity (transparency to the box)
                image = cv.addWeighted(overlay, 0.6, image, 0.4, 0)

                # now put the text (label: confidence %)
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)


        crop_imgs = []

        for i in range(len(tl_cord)):
            crop_imgs.append(padded[tl_cord[i][1] - extra:br_cord[i][1] + extra, tl_cord[i][0] - extra:br_cord[i][0] + extra])

        my_bar.progress(25)


        # PART 2 - Segmentation of detected bottles


        output_img = padded.copy()
        for i, crop_img in enumerate(crop_imgs):
            orig = crop_img
            img = cv.blur(orig,(7,11))

            b_bg, g_bg, r_bg = cv.split(img)

            b1 = []
            g1 = []
            r1 = []

            h = img.shape[0]
            w = img.shape[1]

            # Aggregate the background pixels
            for color1, color2  in zip([b_bg, g_bg, r_bg], [b1, g1, r1]):
                for strip2d in [color1[:w//5,:], color1[:,:w//5], color1[-w//5:,:], color1[:,-w//5:]]:
                    for strip1d in strip2d:
                        for val in strip1d:
                            color2.append(val)


            r1 = np.array(r1)
            g1 = np.array(g1)
            b1 = np.array(b1)

            # Histogram of the background pixels
            bg_hist1,bins = np.histogram(b1.ravel(),256,[0,256])
            bg_hist2,bins = np.histogram(g1.ravel(),256,[0,256])
            bg_hist3,bins = np.histogram(r1.ravel(),256,[0,256])
            my_bar.progress(35)

            b_bg, g_bg, r_bg = cv.split(img)

            b2 = []
            g2 = []
            r2 = []

            # Agregate the object pixels
            for color1, color2  in zip([b_bg, g_bg, r_bg], [b2, g2, r2]):
                for strip1d in [color1[h//8:h-h//8, w//4:-w//4]]:
                    for val in strip1d:
                        color2.append(val)

            r2 = np.array(r2)
            g2 = np.array(g2)
            b2 = np.array(b2)

            # Histogram of object pixels
            obj_hist1,bins = np.histogram(b2.ravel(),256,[0,256])
            obj_hist2,bins = np.histogram(g2.ravel(),256,[0,256])
            obj_hist3,bins = np.histogram(r2.ravel(),256,[0,256])
            my_bar.progress(40)

            b_all, g_all, r_all = cv.split(img)

            # Probability of being background
            prob_being_background = np.array(bg_hist1)[list(b_all)]/np.sum(bg_hist1) + np.array(bg_hist2)[list(g_all)]/np.sum(bg_hist2) + np.array(bg_hist3)[list(r_all)]/np.sum(bg_hist3)

            # Probability of being the object
            prob_being_object = np.array(obj_hist1)[list(b_all)]/np.sum(obj_hist1) + np.array(obj_hist2)[list(g_all)]/np.sum(obj_hist2) + np.array(obj_hist3)[list(r_all)]/np.sum(obj_hist3)

            y = np.array(range(w))

            # Weighted absolute distance of pixel from the center line
            dist_from_center = 0.035*np.abs(y - w/2)/w

            # Weighted absolute distance of pixel from the edge
            dist_from_edge = 0.035*(w/2 - np.abs(y - w/2))/w

            # Each pixel mapped to its probable output
            prob_matrix = (dist_from_edge + prob_being_object > dist_from_center+prob_being_background) * 255


            # For a smoother output
            prob_matrix_blurred = cv.blur(prob_matrix, (1,1))

            disp = orig.copy()
            blurred = cv.blur(orig,(50,50))

            # Background mask
            bg_mask = (cv.blur(prob_matrix_blurred,(25,35)) < 35) * 1
            bg_mask = np.stack((bg_mask, bg_mask, bg_mask), axis = 2)

            # Object mask
            obj_mask = (cv.blur(prob_matrix_blurred,(25,35)) > 35) * 1
            obj_mask = np.stack((obj_mask, obj_mask, obj_mask), axis = 2)
            final = np.multiply(bg_mask,disp) + np.multiply(obj_mask,blurred)

            # Replace the detected segment with the blurred segment
            output_img[tl_cord[i][1]-extra:br_cord[i][1]+extra, tl_cord[i][0]-extra:br_cord[i][0]+extra] = final

        my_bar.progress(100)

        st.image(output_img, caption='Final Image.', use_column_width=True, channels="BGR")
        return 'Done!'

    except:
        return 'Try another image!!'

# All related paths (local to local machine)
config_path = "C:\\Users\Gokul\Downloads\yolov3_custom.cfg"
weights_path = "C:\\Users\Gokul\Downloads\yolov3_custom_last.weights"
labels_path = "C:\\Users\Gokul\Downloads\obj.names"
poster_file = "C:\\Users\Gokul\Downloads\Poster.png"
opencv_image = cv.imread(poster_file)



st.set_page_config(page_title="Beer Blurrer", page_icon=":beers:")
st.markdown('# Blur the Beer :beers::beers:')

my_expander = st.beta_expander("About Beer Blurrer", expanded=False)

with my_expander:
    'The Indian cinema censor board demands to blur alcohol bottles in movies but it is a tedious job to do it manually.' \
    ' Pertaining to this issue we have devised a computer vision model which automatically detects the presence ' \
    'of alcohol bottles in the frame/image. The region of interest is thus obtained, which undergoes a process of ' \
    'image segmentation to accurately obtain the pixels that are contributing to the bottle. Those are the pixels ' \
    'that will be prone to gaussian blur.'
    """
    Source: [Github](https://github.com/siddarth-c/Digital-Image-Processing)
    """
    st.image(opencv_image, channels="BGR")




uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    in_image = cv.imdecode(file_bytes, 1)

    # Display loaded image
    st.image(in_image, caption='Uploaded Image.', use_column_width=True, channels="BGR")

    st.write("")

    st.info("Blurring...")

    status = blur_beer(in_image, config_path, weights_path, labels_path)

    if status == 'Done!':
        st.success(status)
    else:
        st.error(status)