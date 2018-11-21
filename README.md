# facial-feature-extractor
APIs to extract facial feature from given image.

## Requirements

- face_recognition
- matplotlib
- numpy

## Experiments

### Pure `face_recognition` model

- `model='hog'`  
    Fast, acceptable performance  

    ![hog](./img/face_locations/hog.jpg)  

- `model='cnn'`  
    **Extremely** slow, but significantly better performance  

    ![cnn](./img/face_locations/cnn.jpg)  
