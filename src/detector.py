import logging
import torch

from spacy.language import Language
from spacy.tokens import Doc, Span
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)



@Language.factory("abbreviation_detector")
class AbbreviationDetector:
    pipe = None

    def __init__(
        self,
        nlp: Language,
        config_path: str,
        tokenizer_path: str,
        model_path:str,
        name: str = 'abbreviation_detector',
        make_serializable: bool = False,
    ) -> None:

        Doc.set_extension("abbreviations", default=[], force=True)
        Doc.set_extension("abbreviation_pairs", default=[], force=True)
        Doc.set_extension("long2short", default={}, force=True)
        Doc.set_extension("short2long", default={}, force=True)
        Span.set_extension("long_form", default=None, force=True)
        Span.set_extension("is_short_form", default=False, force=True)

        self.make_serializable = make_serializable
        self.init_pipeline(config_path, tokenizer_path, model_path)

    def init_pipeline(self, config_path, tokenizer_path, model_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = AutoConfig.from_pretrained(config_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False
        )
        model = AutoModelForTokenClassification.from_config(config)
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device),
            )
        )
        self.pipe = pipeline('token-classification', model=model, tokenizer=tokenizer, use_fast=True, aggregation_strategy='first')
        logging.info('transformers pipeline ready')

    def __call__(self, doc: Doc) -> Doc:
        output = self.pipe(doc.text)
        logger.debug(output)

        if len(output) == 0:
            return doc

        # get long-short pairs
        all_long_short_pairs = []
        prev_detection = output[0]
        for pred in output[1:]:
            if prev_detection['entity_group'] == 'Long' and pred['entity_group'] == 'Short':
                lf = doc.char_span(prev_detection['start'], prev_detection['end'], alignment_mode='expand')
                sf = doc.char_span(pred['start'], pred['end'], alignment_mode='expand')
                all_long_short_pairs.append((lf, sf))

            prev_detection = pred

        doc._.abbreviation_pairs = [self._serialize_pairs(pair) for pair in all_long_short_pairs]
        doc._.long2short = dict((lf.text, sf.text) for lf, sf in all_long_short_pairs)
        doc._.short2long = dict((sf.text, lf.text) for lf, sf in all_long_short_pairs)

        logger.info(f'got {len(doc._.abbreviation_pairs)} abbreviation pairs')

        # check if multiple long forms exist for the same short form
        all_short = set()
        for _, sf in all_long_short_pairs:
            if sf.text in all_short:
                raise "Multiple short-forms present !"

            all_short.add(sf)

        # tag short forms
        short_form_list = []
        for pred in output:
            if pred['entity_group'] == 'Long':
                continue

            short_form = doc.char_span(pred['start'], pred['end'], alignment_mode='expand')
            short_form._.is_short_form = True
            short_form._.long_form = doc._.short2long.get(short_form.text, None)
            short_form_list.append(short_form)

        doc._.abbreviations = [self._serialize_short_form(short_form) for short_form in short_form_list]
        logger.info(f'got {len(doc._.abbreviations)} short forms')

        return doc

    def _serialize_short_form(self, short_form):
        return {
            'short_form': short_form.text,
            'start': short_form.start,
            'end': short_form.end
        }

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
