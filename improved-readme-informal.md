# Machine Learning: Electron Orbital Classification using TensorFlow
Here I explore how machine learning can be used to classify quantum states/ electron orbitals about the hydrogen atom, based on the principle quantum number n.

## Project Evolution
After my previous attempt at building a model wasn't very successful, I decided to make some key changes to improve its accuracy:

### What Changed?
1. **Simplified the Problem**
   - Instead of trying to classify n, l, and m quantum numbers all at once, I focused just on n
   - This helped the model learn the most fundamental quantum property first
   - Turns out simpler is sometimes better!

2. **Cleaned Up the Model**
   - Built a more straightforward CNN with:
     ```python
     model = models.Sequential([
         layers.Conv2D(32, (3, 3), activation='relu'),
         layers.MaxPooling2D((2, 2)),
         layers.Conv2D(64, (3, 3), activation='relu'),
         layers.MaxPooling2D((2, 2)),
         layers.Conv2D(64, (3, 3), activation='relu'),
         layers.MaxPooling2D((2, 2)),
         layers.Flatten(),
         layers.Dense(64, activation='relu'),
         layers.Dropout(0.5),
         layers.Dense(num_classes, activation='softmax')
     ])
     ```
   - Added dropout to prevent overfitting
   - Kept the architecture neat and focused

3. **Better Data Handling**
   - Standardized all images to 128x128 pixels
   - Improved how quantum numbers are extracted from filenames
   - Made sure all images are processed consistently

## Results
The model shows significant improvement from my previous attempt. Where before it struggled to classify orbitals within its own training set, this version successfully identifies principle quantum numbers of test images, demonstrating it has learned meaningful features from the training data.

## What's Next?
- Could try adding back l and m classification once n is solid
- Maybe explore different model architectures
- Definitely want to expand the training dataset

## Dependencies
- TensorFlow 2.x
- NumPy
- PIL (Python Imaging Library)
- Matplotlib

## Notes
All code is available in `improved-electron-orbital-classification.ipynb`. The training data consists of electron orbital visualizations, with the quantum numbers encoded in the filenames.
