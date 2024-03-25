
# t5-small-samsum
This model is a fine-tuned version of [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) on an [samsum](https://huggingface.co/datasets/samsum) dataset. It achieves the following results on the evaluation set:

- Loss: 1.6507

## Usage

## Use a pipeline

```python
from transformers import pipeline

pipe = pipeline("summarization", model="Prikshit7766/t5-small-samsum")
```

### Load model directly

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Prikshit7766/t5-small-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("Prikshit7766/t5-small-samsum")
```