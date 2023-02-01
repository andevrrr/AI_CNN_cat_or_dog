# Convolutional neural network prediction algorithm 

## Work description and analysis:

The dataset: https://www.kaggle.com/c/dogs-vs-cats/data
Images of houses can simply be searched on the Internet.

The data “cats and dogs ” was trained successfully. The result of accuracy that I got is 82.73 and the plot looks accordingly:

![alt text](https://github.com/andevrrr/AI_CNN_cat_or_dog/blob/main/imagies/plot.png?raw=true)

I added two more Conv2D layers with MaxPooling2D layers. The number of output filters in the convolution in both cases I set as 64. Moreover, for more precise training I set the input_shape and target_size numbers to (300, 300, 3) and (300, 300) accordingly. The last thing, I implemented an augmentation structure.

This can be called successful training because looking at the graph the curves get closer to each other with crossing point.
By the way, it took 3 hours for my laptop to train the data.

## Summarization:
The more you train your data, the more accurate it will be, we can see this in the example above. In addition, the images that are used to train the data also affect the final result, they should be more diverse so that the model has a better idea of what, for example, cats look like.

## To start

- download the files
- run "files-todirectories.py" file first
- run "train-with-baseline-model.py" file
- run "predict-with-baseline-model.py" file

Inside the files I have added clear, descriptive comments to understand how the codes work
