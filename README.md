# Everis Machine Learning Prize
This repository contains our project for the Everis Machine Learning Prize competition. This project was developed by [João Figueira](https://github.com/joaoperfig) and [Tiago Almeida](https://github.com/ForbiddenOne).

The objective was to create a dataset and train a neural network that could indentify multiple (possibly different) fruits in images.

We implemented three different solutions. The first one use uses a YOLO model trained with the Microsoft COCO library. This library only contains 3 classes of fruits (Apples, Bananas and Oranges). Results are show in the images below (more examples can be found in the folder [results](https://github.com/joaoperfig/everismlprize/tree/master/Results/YOLO)).


<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/YOLO/table.jpg">
</p>
<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/YOLO/oi.jpg">
</p>

The second approach is a convolutional neural network created by us and trained using our own dataset to be able to identify a broader variety of fruits. The images below show the achieved results with our model (more examples can be found in the folder [results](https://github.com/joaoperfig/everismlprize/blob/master/Results/Our%20NN)).

<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/Our%20NN/childrenstesting2.jpg">
</p>
<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/Our%20NN/oiNEW.jpg">
</p>

The third approach is a mix of the previous ones, a YOLO model trained with our dataset. This combines the detection capabilities of the YOLO model with the diversity of fruits of our dataset (much larger than in the Microsoft COCO library).

<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/Final%20results/Picture2.jpg">
</p>
<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/Final%20results/Picture3.jpg">
</p>
<p align="center">
  <img src="https://github.com/joaoperfig/everismlprize/blob/master/Results/Final%20results/Picture4.jpg">
</p>

For more information read our [presentation](https://github.com/joaoperfig/everismlprize/blob/master/Presentation/Tutti_Viewti_Presentation.pdf) or our [paper](https://github.com/joaoperfig/everismlprize/blob/master/Everis_Prize.pdf) (paper does not include latest approach) where you can also find links to Google Colab notebooks containing working versions of the code.
