from ..config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = None
        self.pipe = None

    def lazy_load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.pipe is None:
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
            self.pipe = pipeline("summarization", model=self.config.model_path, tokenizer=self.tokenizer)

    def predict(self, text):
        self.lazy_load()

        print("Dialogue:")
        print(text)

        output = self.pipe(text)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output
