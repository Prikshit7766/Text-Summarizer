import os
from ..logging import logger
from ..entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):

        prefix = "summarize: "
        # Tokenize the dialogue from the example batch
        inputs = [prefix + doc for doc in example_batch['dialogue']]
        model_inputs = self.tokenizer(
            inputs,                     # Extract dialogue from the example batch
            max_length=1024,             # Set maximum sequence length for input
            truncation=True             # Truncate sequences that exceed max_length
        )

        # Tokenize the summary from the example batch as target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example_batch['summary'],  # Extract summary from the example batch
                max_length=128,             # Set maximum sequence length for summary
                truncation=True            # Truncate sequences that exceed max_length
            )

        
        model_inputs["labels"] = labels["input_ids"]
        # Return a dictionary containing input and target encodings
        return model_inputs
    
    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset.map(
            self.convert_examples_to_features,
            batched=True,
        )

        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum"))



