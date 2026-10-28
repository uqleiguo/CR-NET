# **CR-NET for Clutter Signal Removal in Medical Microwave Sensing**
This repository includes the codes that implement a clutter removal neural network (CR-NET) proposed in the paper titled "Physics-Guided Coupled Neural Networks for
Clutter Extraction in Medical Microwave Sensing" written by [Lei Guo](https://about.uq.edu.au/experts/15936), [Alina Bialkowski](https://about.uq.edu.au/experts/19747), Indu Rajmohan, [Syed Naqvi](https://about.uq.edu.au/experts/33730), and [Amin Abbosh](https://about.uq.edu.au/experts/1576).
## **Environment**
The python codes are based on torch 2.8.0, numpy 2.0.2, matplotlib 3.9.4, and python 3.9. 
## **Codes**
This repository includes codes for two medical microwave sensing tasks: head stroke classification and skin thickness estimation.

Codes for head stroke classification task are in "codes_head".

Codes for skin thickness estimation task are in "codes_skin"
### **"codes_head"**
- "train_clutter_extractor_head.py" and "test_clutter_extractor_head.py" are the training and testing codes for the clutter extractor.
- "train_sensing_module_head.py" and "test_sensing_module_head.py" are the training and testing codes for the sensing module.
- "sensing_module_head.py" is the neural network used in the sensing module.
### **"codes_skin"**
- "train_clutter_extractor_skin.py" and "test_clutter_extractor_skin.py" are the training and testing codes for the clutter extractor.
- "train_sensing_module_skin.py" and "test_sensing_module_skin.py" are the training and testing codes for the sensing module.
- "sensing_module_skin.py" is the neural network used in the sensing module.
## **Data**
Some of the data may be made available upon reasonable request. Please contact l.guo3@uq.edu.au
## **Models**
Trained models are given in [Trained Models](https://drive.google.com/drive/u/0/folders/1DiBQEUx5KsFXvwVToNfNrrmuHVP1AopV) (Access will be given by contacting l.guo3@uq.edu.au)
## **Citation**
If you find our work helpful in your research, please consider citing our papers:
```
