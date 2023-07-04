import cv2
# import pytesseract
import easyocr

import numpy as np

SILENCE_THRESHOLD=400
SILENCE_RATIO=100

vidcap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'])
# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def process_image():
        ret, frame = vidcap.read()
        print("Processing image")
        # edge = cv2.Canny(frame, 50, 150)
        # cv2.imshow("Frame", edge)
        cv2.waitKey(1)
        results = reader.readtext(frame)
        for (bbox, text, prob) in results:
                # display the OCR'd text and associated probability
                print("[INFO] {:.4f}: {}".format(prob, text))
                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                # cleanup the text and draw the box surrounding the text along
                # with the OCR'd text itself
                text = cleanup_text(text)
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                cv2.putText(frame, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", frame)
        # text = pytesseract.image_to_string(frame)
        print(results)

print("Running")
while True and vidcap.isOpened():
    process_image()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vidcap.release()
cv2.destroyAllWindows()


    