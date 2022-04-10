import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForTokenClassification

from src import engine, utils
from src.dataset import AbbreviationDetectionDataset

TRAIN_FILEPATH = "data/AAAI-21-SDU-shared-task-1-AI/dataset/train.json"
VAL_FILEPATH = "data/AAAI-21-SDU-shared-task-1-AI/dataset/dev.json"
LABEL_TO_NUM_DICT = {
    "O": 0,
    "B-long": 1,
    "B-short": 2,
    "I-long": 3,
    "I-short": 4,
}
NUM_TO_LABEL_DICT = dict(((n, s) for s, n in LABEL_TO_NUM_DICT.items()))

MODEL_PATH = "allenai/scibert_scivocab_cased"
EPOCHS = 5
LR = 2e-5
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_raw = utils.load_json(TRAIN_FILEPATH)
val_raw = utils.load_json(VAL_FILEPATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, do_lower_case=False)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH, num_labels=len(LABEL_TO_NUM_DICT)
)
model.to(DEVICE)

train_dataset = AbbreviationDetectionDataset(train_raw, tokenizer, LABEL_TO_NUM_DICT)
val_dataset = AbbreviationDetectionDataset(val_raw, tokenizer, LABEL_TO_NUM_DICT)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR)

model, optimizer = engine.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
)
