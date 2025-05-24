# Torch Lite Classifier

A lightweight, high-accuracy image classification model built with PyTorch and EfficientNet Lite0. Trained on CIFAR-10 with **99.54% accuracy**, this project includes support for both standard and quantized models optimized for efficient CPU inference.

## Overview

Torch Lite Classifier leverages the EfficientNet Lite0 architecture to achieve excellent accuracy while maintaining low computational overhead. The model is trained on CIFAR-10 images resized to 224×224 pixels and fine-tuned for optimal performance. This repository includes scripts for data preparation, training, evaluation, and dynamic quantization for deployment on resource-constrained devices.

## Features

- State-of-the-art EfficientNet Lite0 backbone pretrained and fine-tuned on CIFAR-10  
- Achieves **99.54% classification accuracy**  
- Dynamic quantization to reduce model size and speed up inference on CPU  
- Automated dataset extraction and preparation from compressed archives  
- Training pipeline with PyTorch and timm library  
- Easy-to-use inference API and Gradio demo interface available  

## Live Demo

Try the model in action on the live demo at [ELiteVision] https://elitevision.onrender.com 

## Usage

- Extract and organize CIFAR-10 images into class folders automatically  
- Train model with configurable hyperparameters and data augmentations  
- Save and quantize the trained model for CPU deployment  
- Use provided FastAPI or Gradio interfaces for inference  

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open issues or submit pull requests on GitHub.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
