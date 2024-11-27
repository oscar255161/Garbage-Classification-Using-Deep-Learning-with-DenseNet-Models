# Garbage Classification Using Deep Learning with DenseNet Models

## Introduction  
Garbage classification has become a global environmental challenge. From simple waste disposal methods to today’s recycling and waste reduction strategies, significant progress has been made. However, with increasing diversity in human activities and consumption patterns, the types and quantities of waste have also risen, ranging from household garbage to industrial waste, making efficient classification and disposal more complex and challenging.

Currently, waste classification largely depends on manual sorting and basic mechanical processing, which are inefficient and prone to errors, affecting resource recovery and recycling. Moreover, improper waste management leads to severe environmental pollution, such as soil and water contamination, as well as greenhouse gas emissions, threatening human health and the planet's sustainability.

This project aims to provide an innovative solution by utilizing technology to automate waste classification. By analyzing and identifying images of waste, this system improves classification efficiency and accuracy. The TrashNet dataset from Kaggle is used for training the model, employing advanced convolutional neural networks for deep learning to achieve efficient waste recognition and classification.

---

## Methodology  
In the modern era of rapid technological advancements, deep learning has become a key tool for addressing numerous challenges. This project, **"Garbage Classification,"** leverages deep learning techniques to develop a model capable of accurately classifying waste images, thereby enhancing the efficiency of waste sorting.

### Steps:  
1. **Dataset Preparation**:  
   - TrashNet dataset from Kaggle.  
   - 70% of the data is used as the training set, while the remaining 30% is reserved for validation.  
   - Data augmentation techniques such as rotation, flipping, and scaling are applied to increase diversity and improve model generalization.

2. **Model Architecture**:  
   - **DenseNet-121** and **DenseNet-169** are employed for training.  
   - DenseNet utilizes a "dense connection" structure where each layer connects directly to all previous layers, enhancing feature reuse, reducing parameter count, and improving efficiency.

3. **Training Setup**:  
   - **Hardware**: NVIDIA GTX 1080 GPU and Intel Core i7-3770 CPU.  
   - **Evaluation**: Models are trained and tested to assess the impact of batch size and architectural choices on classification performance.  

### Key Benefits of DenseNet:  
- Strengthened feature propagation by reusing features from previous layers.  
- Reduces network depth and complexity, minimizing overfitting risks.  
- Improves training efficiency.  

---

## Results  

| Model       | Batch Size | Training Accuracy | Validation Accuracy |  
|-------------|------------|--------------------|----------------------|  
| DenseNet121 | 8          | 97.1%             | 95.1%               |  
| DenseNet121 | 16         | 93.4%             | 81.2%               |  
| DenseNet169 | 8          | 97.9%             | 97.2%               |  
| DenseNet169 | 16         | 98.0%             | 98.3%               |  

From the results, DenseNet-169 with a batch size of 16 demonstrates the best performance, achieving a validation accuracy of 98.3%.

---

## User Interface  

The final trained model is integrated into a GUI for practical use, allowing users to classify images easily.  

### Features:  
- **Choose Image**: Select an image for classification.  
- **Start Classification**: Run the model to classify the uploaded image.  
- **Show Category**: Display the classification result.  
- **Clean Current Image**: Reset the interface for a new task.


---

## Conclusion  
This project successfully applies deep learning to waste classification, achieving high accuracy with DenseNet models. By integrating a user-friendly GUI, it is well-suited for real-world applications. Future developments aim to create a mobile application for broader accessibility, promoting efficient waste management and contributing to a cleaner environment.

