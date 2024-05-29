# Model Detect Person Wearing Safety Helmet
Our project, “Model Detect Person Wearing Safety Helmet,” utilizes machine learning and computer vision technology to detect individuals wearing safety helmets. The primary goal of this project is to enhance safety in work environments such as construction sites, manufacturing plants, and other industrial areas.

Our model is trained using the TensorFlow Object Detection API, a powerful framework that allows for object detection in images. We have used a large dataset containing images of individuals wearing safety helmets in various environments and angles to ensure the model can operate effectively in real-world scenarios.

This model not only detects individuals in images but also determines whether they are wearing a safety helmet. This allows organizations to quickly identify and address potentially dangerous situations, ensuring compliance with occupational safety regulations.

We have also developed a web page using Django, allowing users to upload images and process them with our model. The end result is a comprehensive system, from the server-side to the user interface, that effectively monitors and ensures occupational safety.
# Install and Set up all dependencies
## Create and activate environment variables
```
python -m venv myvenv
.\myvenv\Scripts\activate
```
## Install jupyter notebook, ipykernel, add environment to jupyter notebook and open jupyter notebook in terminal
```
pip install jupyter ipykernel
python -m ipykernel install --user --name=myvenv
jupyter notebook
```
## Open Setup.ipynb and click change kernel
![examples](setup_images/changekernel.png)
## Change kernel to myvenv and run all cells
![examples](setup_images/selectenvir.png)