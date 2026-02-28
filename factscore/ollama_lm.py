from factscore.lm import LM
import ollama

class OllamaModel(LM):
    def __init__(self, model_name="llama3.2", cache_file=None):
        self.model_name = model_name
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temp, "num_predict": max_output_length}
        )
        output = response['message']['content']
        return output, response
