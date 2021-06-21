# Beer Blurrer

The Indian cinema censor board demands to blur alcohol bottles in movies but it is a tedious job to do it manually. Pertaining to this issue we have devised a computer vision model which automatically detects the presence of alcohol bottles in the frame/image. The region of interest is thus obtained, which undergoes a process of image segmentation to accurately obtain the pixels that are contributing to the bottle. Those are the pixels that will be prone to gaussian blur.

## Poster
![Beer Blurrer Poster](https://github.com/siddarth-c/Digital-Image-Processing/blob/main/Poster.png "Title")

## StreamLit Output
![Beer Blurrer Poster](https://github.com/siddarth-c/Digital-Image-Processing/blob/main/StreamLit.png "Title")

## Result
Achieved a MAP (mean average precision) score of 99.19%. <br>
Average processing time per image: 0.75 sec <br>
YOLO training Graph: <br>
<img src = "https://github.com/siddarth-c/Digital-Image-Processing/blob/main/yolo_train_graph.png" width="600">

## Team Members: <br>

[B Gokulapriyan](https://github.com/Gokulapriyan-B/Machine-Learning)
<br>
D Balajee
<br>
A Navaas Roshan
