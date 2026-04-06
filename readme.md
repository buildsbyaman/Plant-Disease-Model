# Agriculture (Plant & Cassava Disease Detection)

This project implements robust disease detection systems for agricultural crops. 

I used advanced architectures (Transformers) to capture global pathological context better than standard CNN baselines.

---

### Project Structure

```bash
├── Plant&Disease/         # Contains New Plant Diseases Dataset (Augmented) with 38 classes
├── CassavaDisease/        # Full training and test data for Cassava leaf diseases 
├── graphs/                # All training plots, ROC curves, and EDA charts (updates dynamically)
├── requirements.txt       # Pinned library versions (Gradio/Pydantic/HF) for Python 3.9 stability
├── cmd.txt                # Quick execution commands to run the whole pipeline
│
├── train_plant.py         # [Task 1] Trains DeiT model with Focal Loss for plant imbalance
├── train_cassava.py       # [Task 2] Trains Swin Transformer for Cassava morphology 
├── test.py                # Interactive Gradio UI for inference & metrics visualization
│
├── plant_model.pth        # Saved weights for Plant Diease track
├── cassava_model.pth      # Saved weights for Cassava track
└── *_classes.txt          # Class label mapping files for the inference engine
```

---

### Functional Highlights

#### 1. Track One: New Plant Diseases
*   **Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (Augmented with 38 classes)
*   **EDA**: Class distribution quantification and environmental confound analysis.
*   **Architecture**: `DeiT` (Data-efficient Image Transformer) tiny patch16_224.
*   **Loss**: **Focal Loss** to counteract major class imbalance across 38 labels.

#### 2. Track Two: Cassava Leaf Disease
*   **Dataset**: [Kaggle Link](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) (2020 version)
*   **EDA**: Morphological frequency analysis and ROC curves for healthy-vs-diseased contrast.
*   **Architecture**: `Swin Transformer` for specialized region localization.
*   **Loss**: **Label Smoothing** loss to handle noisy competitive annotations.

#### 3. Inference Dashboard (`test.py`)
*   **Predict Tab**: Upload a leaf image, select your track, and get real-time health diagnostics.
*   **Metrics Tab**: Dynamically refreshes and displays training plots exported to the `graphs/` folder.

---

### Starting Up
Run the following to set up your environment and launch the app:
1. `pip3 install -r requirements.txt`
2. `python3 train_plant.py && python3 train_cassava.py`
3. `python3 test.py`

*Launch at: http://127.0.0.1:7860/ or use the public share link provided in terminal.*
