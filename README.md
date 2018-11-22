# facial-feature-extractor
APIs to extract facial feature from given image.

## Requirements

- face_recognition
- matplotlib
- numpy

## Experiments

### Face Locations

- Pure `face_recognition` implementation  

    - `model='hog'`  
        Fast, acceptable performance  

        ![hog](./img/face_locations/hog.jpg)  

    - `model='cnn'`  
        **Extremely** slow, but significantly better on sideway face cases  

        ![cnn](./img/face_locations/cnn.jpg)  

### Face Landmarks

- Pure `face_recognition` implementation  

    - `model=large` (72 points)  
        Medium in speed, tricky on sideway face cases  

        ![large](./img/face_landmarks/large.jpg)  
