import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import spacy

from src.detector import AbbreviationDetector


text = "The 2002–3 pandemic caused by severe acute respiratory syndrome coronavirus (SARS-CoV) was one of the most significant public health events in recent history(1). An ongoing outbreak of Middle East respiratory syndrome coronavirus (MERS-CoV)(2) suggests that this group of viruses remains a major threat and that their distribution is wider than previously recognized. Although bats have been suggested as the natural reservoirs of both viruses(3–5), attempts to isolate the progenitor virus of SARS-CoV from bats have been unsuccessful. Diverse SARS-like coronaviruses (SL-CoVs) have now been reported from bats in China, Europe and Africa(5–8), but none are considered a direct progenitor of SARS-CoV because of their phylogenetic disparity from this virus and the inability of their spike proteins (S) to use the SARS-CoV cellular receptor molecule, the human angiotensin converting enzyme II (ACE2)(9,10). Here, we report whole genome sequences of two novel bat CoVs from Chinese horseshoe bats (Family: Rhinolophidae) in Yunnan, China; RsSHC014 and Rs3367. These viruses are far more closely related to SARS-CoV than any previously identified bat CoVs, particularly in the receptor binding domain (RDB) of the S protein. Most importantly, we report the first recorded isolation of a live SL-CoV (bat SL-CoV-WIV1) from bat fecal samples in Vero E6 cells, which has typical coronavirus morphology, 99.9% sequence identity to Rs3367 and uses the ACE2s from human, civet and Chinese horseshoe bat for cell entry. Preliminary in vitro testing indicates that WIV1 also has a broad species tropism. Our results provide the strongest evidence to date that Chinese horseshoe bats are natural reservoirs of SARS-CoV, and that intermediate hosts may not be necessary for direct human infection by some bat SL-CoVs. They also highlight the importance of pathogen discovery programs targeting high-risk wildlife groups in emerging disease hotspots as a strategy for pandemic preparedness."
nlp = spacy.load('en_core_sci_sm')
nlp.add_pipe('abbreviation_detector', config={'config_path': 'models/config.json', 'tokenizer_path': 'allenai/scibert_scivocab_cased', 'model_path': 'models/abbreviation_detector_v2.pt'})
doc = nlp(text)
print(doc._.abbreviation_pairs)
