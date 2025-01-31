# Image-Generation-using-Variational-Auto-Encoder-VAE:
Built and trained a Variational Autoencoder (VAE) on the CelebA dataset to generate realistic celebrity faces. Used a CNN-based encoder-decoder, KL divergence, and reconstruction loss to learn facial representations. Explored latent space interpolation for smooth feature transitions.

### Variational Autoencoder for Celebrity Face Generation

### Project Overview:
In this project, I designed and trained a Variational Autoencoder (VAE) using the CelebA (Celebrity Faces) dataset to generate realistic human faces. The goal was to learn a meaningful latent representation of facial features and enable the generation of new, high-quality synthetic faces that resemble real celebrities.

**Key Contributions:**
	• Preprocessed and Augmented the CelebA Dataset: Cleaned and normalized over 200,000 celebrity images, resized them to a suitable resolution for training.
**•Built and Trained a VAE Architecture:**	
	• Implemented encoder-decoder networks using Convolutional Neural Networks (CNNs) in TensorFlow/Keras/PyTorch.
	•Utilized a latent space bottleneck with mean and variance estimation for probabilistic encoding.
	•Trained the Model for High-Quality Face Generation:
	•Used KL divergence loss and reconstruction loss (MSE/BCE) to balance learning.
	•	Trained with GPU acceleration for efficient optimization.
	•	Generated and Visualized New Faces:
	•	Sampled from the learned latent space to generate new celebrity-like faces.
	•	Used latent space interpolation to observe smooth transformations between facial features.
	•	Evaluation and Optimization:
	•	Improved generation quality by tuning hyperparameters, adjusting latent space size, and applying batch normalization/dropout.
	•	Visualized latent space representations to understand how the model learned different facial attributes.

###Technologies & Tools Used:
✅ Python, TensorFlow/PyTorch, Keras
✅ OpenCV, NumPy, Matplotlib, Seaborn
✅ CNN-based Encoder-Decoder architecture
✅ KL-Divergence & Reconstruction Loss
✅ CelebA Dataset for real-world facial features

###Outcome & Impact:
	•	Successfully trained a VAE capable of generating highly realistic celebrity faces.
	•	Demonstrated how unsupervised learning and generative modeling can create novel image data.
	•	Gained valuable insights into latent space disentanglement and deep learning-based image synthesis.

This project showcases expertise in deep learning, generative modeling, and computer vision, making it applicable to tasks such as image synthesis, style transfer, and anomaly detection.
