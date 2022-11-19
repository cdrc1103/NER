from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from spacy import displacy

import copy

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def join_results(results):
    joined_results = []
    for result in results:
        if "##" in result["word"] and joined_results:
            joined_results[-1]["end"] = result["end"]
            joined_results[-1]["word"] += remove_prefix(result["word"], "##")
            joined_results[-1]["score"] = min(joined_results[-1]["score"], result["score"])
        else:
            joined_results.append(result)
    return joined_results


def clean_result(result):
    result["label"] = remove_prefix(result["entity"], "-")
    return result


def remove_prefix(word, prefix):
    if prefix in word:
        return word.split(prefix, 1)[1]
    return " " + word


def convert_to_displacy_format(example, ner_results, threshold=0.9):
    results = copy.deepcopy(ner_results)
    joined_results = join_results(results)
    filtered_results = [r for r in joined_results if r["score"] > threshold]
    cleaned_results = [clean_result(r) for r in filtered_results]
    return [{
        "text": example,
        "ents": cleaned_results,
        "title": None
    }]


def inference_pipeline(input_text: str) -> str:
    """Run NER model and return annotated text"""
    ner_results = pipeline(input_text)
    displacy_results = convert_to_displacy_format(input_text, ner_results)
    return displacy.render(displacy_results, style="ent", manual=True)