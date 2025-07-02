# 📝 Grammar Correction using FLAN-T5 with LoRA Fine-Tuning

This project implements a **grammar correction system** using the **FLAN-T5-base transformer** model, fine-tuned with **LoRA (Low-Rank Adaptation)**. It is built for efficiency and performance — covering data preprocessing, model fine-tuning, evaluation, deployment using Streamlit, and containerization with Docker.

---

## 📁 Dataset Preparation

The dataset consists of pairs of **ungrammatical input sentences** and their corresponding **grammatically correct outputs**.

**Example:**

| Input                                           | Output                                             |
|------------------------------------------------|----------------------------------------------------|
| correct grammar: double story bed curtain around | A double story bed with curtains all around.       |
| correct grammar: cat lay on floor paw in shoe   | A cat is laying on the floor with its paw in a shoe|

- The prefix `correct grammar:` is added to instruct the FLAN-T5 model.
- The dataset was cleaned and converted into a CSV format.
- Loaded using the Hugging Face `datasets` library and tokenized using the `AutoTokenizer` from `transformers`.

---

## 🧠 Model Architecture and Fine-Tuning

### ✅ Base Model:
- [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)

### ✅ Fine-Tuning Strategy:
- **LoRA (Low-Rank Adaptation)** is used to perform parameter-efficient fine-tuning.
- This avoids updating all weights, saving compute and memory while maintaining strong performance.

### ✅ Libraries Used:
- `transformers`, `peft`, `datasets`, `torch`, `evaluate`, `sentencepiece`, `scikit-learn`

### Training Summary:
- 2 epochs
- Batch size: 8
- Learning rate: 2e-4
- Trained using Hugging Face’s `Seq2SeqTrainer`
- Loss decreased across epochs indicating successful learning

---

## 📊 Evaluation Metrics

The model is evaluated on the following NLP metrics:

| Metric   | Score       |
|----------|-------------|
| **BLEU** | 0.71–0.72   |
| **ROUGE-L** | ~0.90   |
| **METEOR** | ~0.91     |
| **chrF++** | ~86.13    |

> These are **state-of-the-art level scores** for a lightweight model with just two epochs of training using LoRA!

---

## 🌐 Streamlit App Deployment

An interactive **Streamlit web app** is created for live grammar correction.

### Features:
- Enter any ungrammatical sentence
- Get the corrected version instantly
- Clean and responsive UI

### Run Locally:
```bash
# Activate your Python virtual environment
streamlit run app.py


🐳 Docker Containerization
This project includes Docker support for easy deployment.

Dockerfile highlights:
Uses python:3.10 as base image

Copies project files

Installs dependencies from requirements.txt

Exposes port 8501 for Streamlit

Build and Run:
bash
Copy
Edit
# Build the Docker image
docker build -t vardan201/grammarcorrection .

# Run the container
docker run -p 8501:8501 vardan201/grammarcorrection

grammar-correction/
├── app.py                    # Streamlit app
├── model/                    # LoRA fine-tuned model
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker setup
├── dataset.csv               # Input-output sentence pairs
├── Grammar_Correction_model.py            # Training script
└── README.md                 # Project documentation


📌 requirements.txt (with Pinned Versions)
ini
Copy
Edit
transformers==4.52.4
peft==0.15.2
accelerate==1.8.1
datasets==3.6.0
torch==2.7.1
sentencepiece==0.2.0
evaluate==0.4.1
scikit-learn==1.7.0
streamlit==1.45.1

🛠️ How to Rebuild This Project
Clone the repository

Prepare a dataset with ungrammatical → grammatical sentence pairs

Tokenize the data using FLAN-T5 tokenizer

Apply LoRA and fine-tune the FLAN-T5 model

Evaluate the model using BLEU, ROUGE-L, METEOR, and chrF++

Deploy with Streamlit locally or using Docker

🙌 Final Notes
This project combines efficient fine-tuning (LoRA) with powerful instruction-tuned models (FLAN-T5).

The system delivers high performance with low compute needs.

Fully deployable via Streamlit and Docker — portable across environments.
