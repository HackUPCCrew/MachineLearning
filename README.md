# Crime Prediction

The aim of this project is to predict crime based on a give time. We use an open data set from [here](https://www.kaggle.com/c/sf-crime/data). Steps taken to build our model below:
- Import Time, Lat and Long from the data set
- One Hot encode hours and week  and use them as featuers
- Using K-means clustering, cluster lat long into 10 cluster zones
- Use these cluster zonses as target
- Keras to build a deep learning model
- Serve results using the flask API

![alt text](http://url/to/img.png)
![Watch the video](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](http://youtu.be/vt5fpE0bzSY)
