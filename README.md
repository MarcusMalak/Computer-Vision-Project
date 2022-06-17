# Computer-Vision-Project

This is the repository for the project for CS4245 Seminar Computer Vision by Deep Learning. The goal of the project is 3d-printer failure detection. To that end, a dataset was created from scratch with videos created by the team as well as videos found on the internet. A simple CNN as well as ResNet18 are used to demonstrate the application.

A requirements.txt file is provided for easy instalation of the dependencies of the project. The following command installs all of them:
```console
pip install -r requirements.txt
```

### Important files
The [main.py](https://github.com/MarcusMalak/Computer-Vision-Project/blob/main/printfailure/main.py) file contains the code that was run to generate the experimental results.

The [dataset_creation.py](https://github.com/MarcusMalak/Computer-Vision-Project/blob/main/printfailure/data/dataset_creation.py) creates the data augmentation code.

The [dataset](https://github.com/MarcusMalak/Computer-Vision-Project/tree/main/printfailure/data/dataset) folder contains the annotated dataset created for the project