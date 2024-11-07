# Machine Learning: Improved Electron Orbital Classification using TensorFlow

In a previous project (see ojthomas7/electron-orbital-classification), I:

- Modelled the wavefunctions and electron orbitals of the hydrogen atom
- Built, trained and tested a ML model for electron orbital classification

Here, I took the model, which did not work and was inaccurate at classifying orbitals included in the training data, and built on it in order to create an accurate classification model.

## Project Evolution
I decided to make the following changes to improve its accuracy:

### What Changed?
1. **Cleaned Up the Model**

The original project used a far more complicated CNN architecture than was necessary. The new architecture took the form:
     ```python
   model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
Keeping the architecture neat and focused.

2. **Improved Training Data**
   Initially the model was only trained on quantum states from n = 1 to n = 5, which only consisted of 35 images. The new model was trained on n = 1 to n = 7, increasing the training data to 89 images, however this is still a comparably small data set.

## Results
The model shows significant improvement from my previous attempt. Where before it struggled to classify orbitals within its own training set, this version successfully identifies principle quantum numbers of test images, demonstrating it has learned meaningful features from the training data.

## What's Next?

Moving forward, I would like to explore baking the quantum mechanical equations into the architecture of the model to created a Physics-informed ML model that will be able to accurately classify not just the principle quantum number n, but the angular and magnetic quantum states l and m too. 
