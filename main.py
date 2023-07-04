import cv2
import easyocr
import numpy as np
import os
import openai
from gtts import gTTS 
import playsound

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

openai.api_key = os.getenv("OPENAI_API_KEY")
# import keyboard

# create high level objects 
vidcap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'])

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def capture_image():
    print("Capturing image...")
    ret, frame = vidcap.read()
    return frame

def process_image(image):
        print("Processing image...")
        cv2.waitKey(1)
        results = reader.readtext(image)
        texts = []
        for (bbox, text, prob) in results:
            if(prob < 0.99):
                continue
            texts.append("[INFO] {:.4f}: {}".format(prob, text))

                # display the OCR'd text and associated probability
                # unpack the bounding box
        #     (tl, tr, br, bl) = bbox
        #     tl = (int(tl[0]), int(tl[1]))
        #     tr = (int(tr[0]), int(tr[1]))
        #     br = (int(br[0]), int(br[1]))
        #     bl = (int(bl[0]), int(bl[1]))
        #     # cleanup the text and draw the box surrounding the text along
        #     # with the OCR'd text itself
        #     text = cleanup_text(text)
        #     cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        #     cv2.putText(image, text, (tl[0], tl[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # # show the output image
        # cv2.imshow("Image", image)
        
        return texts

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")


def capture_audio():
    print("Capturing Audio...")
#     stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=1024)
    myrecording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='int16')
    print("Recording started")
    sd.wait()
    print("Recording stopped")
#     frames = []
# #     try:
#     while True:
#         frames.append(stream.read(1024))
#     except KeyboardInterrupt:
#         pass
    
    audio = np.frombuffer(myrecording, np.int16).astype(np.float32)*(1/32768.0)
    return audio

def process_audio(audio):
    print("Processing audio...")
    segments, info = model.transcribe(audio, beam_size=5, language='en')
#     for segment in segments:
        #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return next(segments).text

def format(prompt, context, history):
    template = """ 
    EXTRACTED TEXT:
    {context}.
    CONVERSATION HISTORY:
    {history}.

    Question: {human_input}
    Chatbot:""".format(human_input=prompt, context=context, history=history)
    return template

def query_openai(prompt):
    intro = "Given the following extracted parts of text and history of converstation, create a response in two sentances."
    
    response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": intro+prompt},        
                ]
    )
    return response["choices"][0]["message"]["content"]


history = ""
while True:
        input("Press Enter to continue...")
        audio = capture_audio()
        image = capture_image()
        
        prompt = process_audio(audio)
        texts = process_image(image)

        prompt = format(prompt, texts, history)
        response= query_openai(prompt)
        
        history += prompt
        history += response
        print(history)
        
        speech = gTTS(text = response, lang = 'en', slow = False)
        speech.save("response.mp3")
        playsound.playsound("response.mp3")

vidcap.release()
cv2.destroyAllWindows()




    