# Pyhands

* Clone the repo

* Initiate your tensorflow environment, if not already existent:
  > pip install tensorflow
  
* Add OpenCV and Pyserial
  > pip install opencv-python pyserial
  
* If a web stream or local IP cam is used, specify the stream url over at line 55 in obj_webcam.py
  example: video = cv2.VideoCapture('192.168.X.X:XXXX')
  > python obj_webcam.py
  
* A basic Arduino boilerplate has been provided in the .ino file and the states to be written can be modified over at interface.py from line 106 onwards.

### Supported Gestures:
* Open hand with palm facing the cam lens.
* Closed fist
* Peace or a count of 2
* The german like hand gesture for representing a count of 3
* Super or the standard non-germanlike three finger gesture
* Spider or rock
* Ok / thumbs up
* One  

These can further be modded over at label_map.pbtxt in the training folder.

#### Thanks to:
* Evan Juras
* https://github.com/EdjeElectronics
* https://github.com/datitran  

For contributing towards this project indirectly.
