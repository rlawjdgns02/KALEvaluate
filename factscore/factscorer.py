import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.ollama_lm import OllamaModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="llama3.2",
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 cost_estimate=None,
                 abstain_detection_type=None,
                 batch_size=256):
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.batch_size = batch_size
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        # Use ollama instead of OpenAI
        self.lm = OllamaModel(model_name,
                              cache_file=os.path.join(cache_dir, "ollama.pkl"))

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(
                    model_name=self.model_name,
                    demon_dir=os.path.join(self.data_dir, "demos"),
                    cache_file=os.path.join(self.cache_dir, "ollama_af.pkl"))

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            for topic, gen in zip(topics, generations):
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                score = np.mean([d["is_supported"] for d in decision])

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score

                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores) if scores else 0,
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None]) if any(d is not None for d in decisions) else 0}

        if gamma and init_scores:
            out["init_score"] = np.mean(init_scores)

        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source):
        decisions = []
        for atom in atomic_facts:
            atom = atom.strip()
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            definition = "Answer the question about {} based on the given context.\n\n".format(topic)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())

            output = self.lm.generate(prompt)

            # Parse output
            generated_answer = output[0].lower()
            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            decisions.append({"atom": atom, "is_supported": is_supported})

        return decisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name', type=str, default="llama3.2")
    parser.add_argument('--gamma', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default=".cache/factscore/")
    parser.add_argument('--cache_dir', type=str, default=".cache/factscore/")
    parser.add_argument('--knowledge_source', type=str, default=None)
    parser.add_argument('--abstain_detection_type', type=str, default=None)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--n_samples', type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    abstain_detection_type=args.abstain_detection_type)

    topics, generations = [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            topics.append(dp["topic"])
            generations.append(dp["output"])
            if args.n_samples is not None and len(topics)==args.n_samples:
                break

    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)

    logging.info("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.info("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.info("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.info("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")
