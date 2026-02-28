import json
import numpy as np
import re
import string
import spacy
import nltk
from rank_bm25 import BM25Okapi
import os
from nltk.tokenize import sent_tokenize

from factscore.ollama_lm import OllamaModel

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class AtomicFactGenerator(object):
    def __init__(self, model_name="llama3.2", demon_dir=None, cache_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True

        # Use ollama instead of OpenAI
        self.lm = OllamaModel(model_name, cache_file=cache_file)

        # Load demos if available
        self.demons = {}
        if demon_dir and os.path.exists(demon_dir):
            demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")
            if os.path.exists(demon_path):
                with open(demon_path, 'r') as f:
                    self.demons = json.load(f)

        if self.demons:
            tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def save_cache(self):
        self.lm.save_cache()

    def run(self, generation, cost_estimate=None):
        """Convert the generation into a set of atomic facts."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(paragraphs)

    def get_atomic_facts_from_paragraph(self, paragraphs):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)
            curr_sentences = sent_tokenize(paragraph)
            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            sentences += curr_sentences

        atoms = self.get_init_atomic_facts_from_sentence(sentences)

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if sent in atoms:
                atomic_facts_pairs.append((sent, atoms[sent]))
            else:
                atomic_facts_pairs.append((sent, []))

        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks

    def get_init_atomic_facts_from_sentence(self, sentences):
        """Get atomic facts from sentences using ollama."""
        atoms = {}

        for sentence in sentences:
            if sentence in atoms:
                continue

            # Build prompt
            prompt = "Please breakdown the following sentence into independent facts. Each fact should be a simple, standalone statement.\n\n"

            # Add demos if available
            if self.demons and self.bm25:
                top_matchings = best_demos(sentence, self.bm25, list(self.demons.keys()), k=1)
                for match in top_matchings:
                    prompt += f"Sentence: {match}\n"
                    prompt += "Facts:\n"
                    for fact in self.demons[match]:
                        prompt += f"- {fact}\n"
                    prompt += "\n"

            prompt += f"Sentence: {sentence}\nFacts:\n"

            output, _ = self.lm.generate(prompt)
            atoms[sentence] = text_to_sentences(output)

        return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_matchings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_matchings


def text_to_sentences(text):
    """Transform LLM output into list of facts."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("* "):
            line = line[2:]
        elif re.match(r'^\d+\.?\s', line):
            line = re.sub(r'^\d+\.?\s', '', line)

        if line:
            if line[-1] not in '.!?':
                line += '.'
            sentences.append(line)

    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        int(text)
        return True
    except:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'
    numeric_values = re.findall(pattern, text)
    return set([value for value in numeric_values])


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities


def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        new_facts = []
        for j, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for k, other_fact in enumerate(facts) if k != j]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks


def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]


def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break

    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            sentences.append(sent)
    return sentences


if __name__ == "__main__":
    generator = AtomicFactGenerator(model_name="llama3.2")
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.")
    print(atomic_facts)
    print(para_breaks)
