from pydantic import EnumMemberError
import torch

from spacy.language import Language
from spacy.tokens import Doc, Span
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

# Main Function imports
import spacy
from scispacy.custom_sentence_segmenter import pysbd_sentencizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = AutoConfig.from_pretrained("models/config.json")
TOKENIZER = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_cased", do_lower_case=False
)
MODEL = AutoModelForTokenClassification.from_config(CONFIG)
MODEL.load_state_dict(
    torch.load(
        "models/abbreviation_detector_v2.pt",
        map_location=torch.device(DEVICE),
    )
)


@Language.factory("abbreviation_detector")
class AbbreviationDetector:
    def __init__(
        self,
        nlp: Language,
        name: str = 'abbreivation_detector',
        make_serializable: bool = False,
    ) -> None:

        Doc.set_extension("abbreviations", default=[], force=True)
        Doc.set_extension("abbreviation_pairs", default=[], force=True)
        Doc.set_extension("long2short", default={}, force=True)
        Doc.set_extension("short2long", default={}, force=True)
        Span.set_extension("long_form", default=None, force=True)
        Span.set_extension("is_short_form", default=False, force=True)

        self.make_serializable = make_serializable

    def __call__(self, doc: Doc) -> Doc:
        sentences = self._get_sents(doc)
        text_sents =  [[x.text for x in sent] for sent in sentences]

        tokenized_inputs = TOKENIZER(text_sents, padding=True, return_tensors='pt', is_split_into_words=True)
        with torch.no_grad():
            outputs = MODEL(**tokenized_inputs)

        logits = outputs.logits
        output = logits.argmax(-1).numpy()
        predicted_token_classes = [[CONFIG.id2label[x] for x in t] for t in output]
        synced_predictions = self.sync_predictions(sentences, tokenized_inputs, predicted_token_classes)

        self._discard_single_letter_abbreviation(text_sents, synced_predictions)
        self._correct_mislabeled_O(synced_predictions)
        self.get_long_short_pairs(sentences, synced_predictions, doc)
        self._tag_short_forms(sentences, synced_predictions, doc)

        return doc

    def _get_sents(self, doc: Doc):
        sents = []
        for token in doc:
            if token.is_sent_start:
                sents.append([token])
            else:
                sents[-1].append(token)

        return sents

    def sync_predictions(self, sentences, tokenized_inputs, predicted_token_classes):
        synced_predictions = [['O' for _ in range(len(x))] for x in sentences]
        for batch_id in range(len(sentences)):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_id)
            tokens = predicted_token_classes[batch_id]
            prev_word_idx = None
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                elif wid != prev_word_idx:
                    synced_predictions[batch_id][wid] = tokens[i]

                prev_word_idx = wid

        return synced_predictions

    def _discard_single_letter_abbreviation(self, text_sents, synced_predictions):
        for i, (sent, labels) in enumerate(zip(text_sents, synced_predictions)):
            for j, (word, label) in enumerate(zip(sent, labels)):
                if label[2:] == 'short' and len(word) == 1:
                    synced_predictions[i][j] = 'O'
                    print('found single letter abbreviation. Discarding it!')

    def _correct_mislabeled_O(self, synced_predictions):
        for i, predictions in enumerate(synced_predictions):
            _tmp = []
            B_found_flag = False
            gap = 2

            for j, label in enumerate(predictions):
                if label == "O" and not B_found_flag:
                    continue

                elif label[0] == "B":
                    B_found_flag = True
                    gap = 2
                    _tmp = []
                    continue

                elif label == "O":
                    if gap > 0:
                        _tmp.append(j)
                        gap -= 1
                    else:
                        B_found_flag = False
                        gap = 2
                        _tmp = []
                elif label[0] == "I":
                    if gap == 2:
                        B_found_flag = False
                    else:
                        b_label = predictions[_tmp[0] - 1]
                        if b_label[2:] == label[2:]:
                            for k in _tmp:
                                synced_predictions[i][k] = label

                        B_found_flag = True
                        gap = 2
                        _tmp = []

    def _tag_short_forms(self, sentences, predictions, doc):
        short_form_list = []
        for sent, pred in zip(sentences, predictions):
            b_found = False
            _tmp = []
            for token, label in zip(sent, pred):
                if label == "B-short":
                    b_found = True
                    _tmp.append(token.i)
                if b_found and label == "O":
                    short_form = doc[_tmp[0]: _tmp[-1]+1]
                    short_form._.is_short_form = True
                    short_form._.long_form = doc._.short2long.get(short_form.text, None)
                    short_form_list.append(short_form)

                    b_found = False
                    _tmp = []

        doc._.abbreviations = [self._serialize_short_form(short_form) for short_form in short_form_list]

    def _serialize_short_form(self, short_form):
        return {
            'short_form': short_form.text,
            'start': short_form.start,
            'end': short_form.end
        }


    def get_long_short_pairs(self, sentences, predictions, doc):
        all_long_short_pairs = []

        for sent, pred in zip(sentences, predictions):
            long_form_found = False
            short_form_found = False

            long_form_start_end = [-1, -1]
            short_form_start_end = [-1, -1]

            gap = 3
            for token, label in zip(sent, pred):
                if label == "O" and not long_form_found:
                    continue
                elif label[2:] == "short" and not long_form_found:
                    continue
                elif label[2:] == "long":
                    if label[0] == "B":
                        long_form_found = True
                        long_form_start_end[0] = token.i
                    else:
                        long_form_start_end[1] = token.i
                elif label == "O" and not short_form_found:
                    if gap > 0:
                        gap -= 1
                    else:
                        long_form_found = False
                        long_form_start_end = [-1, -1]
                        gap = 3

                elif label[2:] == "short":
                    if label[0] == "B":
                        short_form_found = True
                        short_form_start_end[0] = token.i
                    else:
                        short_form_start_end[1] = token.i

                elif label == "O" and long_form_found and short_form_found:

                    # Long form str
                    start, end = long_form_start_end
                    if end == -1:
                        curr_long_form = doc[start: start+1]
                    else:
                        curr_long_form = doc[start: end+1]

                    # Short form str
                    start, end = short_form_start_end
                    if end == -1:
                        curr_short_form = doc[start: start+1]
                    else:
                        curr_short_form = doc[start: end+1]

                    # Add to the list
                    all_long_short_pairs.append((curr_long_form, curr_short_form))

                    # Reset
                    long_form_found = False
                    short_form_found = False

                    long_form_start_end = [-1, -1]
                    short_form_start_end = [-1, -1]

                    gap = 3

            if long_form_found and short_form_found:
                # Long form str
                start, end = long_form_start_end
                if end == -1:
                    curr_long_form = doc[start: start+1]
                else:
                    curr_long_form = doc[start: end+1]

                # Short form str
                start, end = short_form_start_end
                if end == -1:
                    curr_short_form = doc[start: start+1]
                else:
                    curr_short_form = doc[start: end+1]

                # Add to the list
                all_long_short_pairs.append((curr_long_form, curr_short_form))

        doc._.abbreviation_pairs = [self._serialize_pairs(pair) for pair in all_long_short_pairs]
        doc._.long2short = dict((lf.text, sf.text) for lf, sf in all_long_short_pairs)
        doc._.short2long = dict((sf.text, lf.text) for lf, sf in all_long_short_pairs)

        # check if multiple long forms exist for the same short form
        all_short = set()
        for _, sf in all_long_short_pairs:
            if sf.text in all_short:
                raise "Multiple short-forms present !"
            
            all_short.add(sf)

    def _serialize_pairs(self, pair):
        long_form, short_form = pair
        return {
            "short_form": short_form.text,
            "short_form_start": short_form.start,
            "short_form_end": short_form.end,
            "long_form": long_form.text,
            "long_form_start": long_form.start,
            "long_form_end": long_form.end,
        }




if __name__ == '__main__':
    nlp = spacy.load('en_core_sci_sm')
    nlp.add_pipe('pysbd_sentencizer', before='parser')
    nlp.add_pipe('abbreviation_detector')

    text = "The 2002-3 pandemic caused by severe acute respiratory syndrome coronavirus (SARS-CoV) was one of the most significant public health events in recent history(1). An ongoing outbreak of Middle East respiratory syndrome coronavirus (MERS-CoV)(2) suggests that this group of viruses remains a major threat and that their distribution is wider than previously recognized. Although bats have been suggested as the natural reservoirs of both viruses(3-5), attempts to isolate the progenitor virus of SARS-CoV from bats have been unsuccessful. Diverse SARS-like coronaviruses (SL-CoVs) have now been reported from bats in China, Europe and Africa(5-8), but none are considered a direct progenitor of SARS-CoV because of their phylogenetic disparity from this virus and the inability of their spike proteins (S) to use the SARS-CoV cellular receptor molecule, the human angiotensin converting enzyme II (ACE2)(9,10). Here, we report whole genome sequences of two novel bat CoVs from Chinese horseshoe bats (Family: Rhinolophidae) in Yunnan, China; RsSHC014 and Rs3367. These viruses are far more closely related to SARS-CoV than any previously identified bat CoVs, particularly in the receptor binding domain (RDB) of the S protein. Most importantly, we report the first recorded isolation of a live SL-CoV (bat SL-CoV-WIV1) from bat fecal samples in Vero E6 cells, which has typical coronavirus morphology, 99.9% sequence identity to Rs3367 and uses the ACE2s from human, civet and Chinese horseshoe bat for cell entry. Preliminary in vitro testing indicates that WIV1 also has a broad species tropism. Our results provide the strongest evidence to date that Chinese horseshoe bats are natural reservoirs of SARS-CoV, and that intermediate hosts may not be necessary for direct human infection by some bat SL-CoVs. They also highlight the importance of pathogen discovery programs targeting high-risk wildlife groups in emerging disease hotspots as a strategy for pandemic preparedness."
    #text = 'Although bats have been suggested as the natural reservoirs of both viruses(3-5), attempts to isolate the progenitor virus of SARS-CoV from bats have been unsuccessful. Diverse SARS-like coronaviruses (SL-CoVs) have now been reported from bats in China, Europe and Africa(5-8), but none are considered a direct progenitor of SARS-CoV because of their phylogenetic disparity from this virus and the inability of their spike proteins (S) to use the SARS-CoV cellular receptor molecule, the human angiotensin converting enzyme II (ACE2) (9,10)'
    doc = nlp(text)
    print(doc._.abbreviations)
    print(doc._.short2long)
    for short_form in doc._.abbreviations:
        start = short_form['start']
        end = short_form['end']
        long_form = doc[start:end]._.long_form
        print(short_form['short_form'], ':', long_form)