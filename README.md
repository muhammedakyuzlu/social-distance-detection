# social distance detection


Before running the app :

Dependencies:
  python3
  pafy 
  youtube-dl
  opencv-python
  numpy

get the yolov4.weights from the link and but him in the same folder 
https://drive.google.com/drive/folders/1iRga-qAcBRWe05dtOlLN6s4FUdPLmcpq?usp=sharing

How to run :
     python3 Social-distance-detection.py [options] path

  options:
     -v for video
     -c for camera
     -y for youtube
     -i for image
     -h for help
 
  path:
    path/url to the surce
    
EXAMPLES :   

python3 Social-distance-detection.py -y https://youtu.be/hTUyzF4v9KA  

python3 Social-distance-detection.py -c 0    