import torch

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src import predict

MAX_LENGTH = 35
OVERLAP = 8

LABEL_TO_NUM_DICT = {
    "O": 0,
    "B-long": 1,
    "B-short": 2,
    "I-long": 3,
    "I-short": 4,
}
NUM_TO_LABEL_DICT = dict(((n, s) for s, n in LABEL_TO_NUM_DICT.items()))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_cased", do_lower_case=False
)
saved_model = AutoModelForTokenClassification.from_pretrained(
    "allenai/scibert_scivocab_cased", num_labels=len(LABEL_TO_NUM_DICT)
)
saved_model.load_state_dict(
    torch.load(
        "models/abbreviation_detector_v2.pt",
        map_location=torch.device(DEVICE),
    )
)


def run(text):
    batch = [sent for sent in sent_tokenize(text)]

    tokenized_inputs = TOKENIZER(batch, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = saved_model(**tokenized_inputs)

    logits = outputs.logits
    output = logits.argmax(-1).numpy()
    output = [[NUM_TO_LABEL_DICT[y] for y in x] for x in output]

    compiled_output = predict.compile_model_output(
        TOKENIZER,
        tokenized_inputs["input_ids"],
        output,
        OVERLAP,
    )
    print(compiled_output)
    predict.removing_single_letter_abbreviations(compiled_output)
    predict.remove_mislabeled_O_in_btwn_B_I(compiled_output)
    long_short_pairs = predict.get_long_short_pairs(compiled_output)
    short_forms = predict.get_short_forms(compiled_output)

    return long_short_pairs, short_forms


if __name__ == "__main__":
    text = "The 2002–3 pandemic caused by severe acute respiratory syndrome coronavirus (SARS-CoV) was one of the most significant public health events in recent history(1). An ongoing outbreak of Middle East respiratory syndrome coronavirus (MERS-CoV)(2) suggests that this group of viruses remains a major threat and that their distribution is wider than previously recognized. Although bats have been suggested as the natural reservoirs of both viruses(3–5), attempts to isolate the progenitor virus of SARS-CoV from bats have been unsuccessful. Diverse SARS-like coronaviruses (SL-CoVs) have now been reported from bats in China, Europe and Africa(5–8), but none are considered a direct progenitor of SARS-CoV because of their phylogenetic disparity from this virus and the inability of their spike proteins (S) to use the SARS-CoV cellular receptor molecule, the human angiotensin converting enzyme II (ACE2)(9,10). Here, we report whole genome sequences of two novel bat CoVs from Chinese horseshoe bats (Family: Rhinolophidae) in Yunnan, China; RsSHC014 and Rs3367. These viruses are far more closely related to SARS-CoV than any previously identified bat CoVs, particularly in the receptor binding domain (RDB) of the S protein. Most importantly, we report the first recorded isolation of a live SL-CoV (bat SL-CoV-WIV1) from bat fecal samples in Vero E6 cells, which has typical coronavirus morphology, 99.9% sequence identity to Rs3367 and uses the ACE2s from human, civet and Chinese horseshoe bat for cell entry. Preliminary in vitro testing indicates that WIV1 also has a broad species tropism. Our results provide the strongest evidence to date that Chinese horseshoe bats are natural reservoirs of SARS-CoV, and that intermediate hosts may not be necessary for direct human infection by some bat SL-CoVs. They also highlight the importance of pathogen discovery programs targeting high-risk wildlife groups in emerging disease hotspots as a strategy for pandemic preparedness."
    pairs, acronyms = run(text)
    print()
    print("Pairs")
    print(pairs)
    print()
    print("Acronyms")
    print(acronyms)
