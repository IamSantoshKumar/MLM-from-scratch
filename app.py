import gradio as gr
import torch
import transformers
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device('cpu')

NUM_CLASSES=5

model=BertForMaskedLM.from_pretrained("./")
tokenizer=BertTokenizer.from_pretrained("./")


def predict(text=None) -> dict:  
    model.eval()
    inputs = tokenizer(str(text), return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    model.to(device)
    token_logits = model(input_ids, attention_mask=attention_mask).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, NUM_CLASSES, dim=1).indices[0].tolist()
    score = torch.nn.functional.softmax(mask_token_logits)[0]
    top_5_score = torch.topk(score, NUM_CLASSES).values.tolist()
    return {tokenizer.decode([tok]): float(score) for tok, score in zip(top_5_tokens, top_5_score)}
    
gr.Interface(fn=predict, 
             inputs=gr.inputs.Textbox(lines=2, placeholder="Your Text… "),
             title="Mask Language Modeling",
             outputs=gr.outputs.Label(num_top_classes=NUM_CLASSES),
             description="Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks",
             examples=['A Good Man Is Hard to Find [MASK].', 'Some stories have a [MASK] kind of message called a moral.'],
             interpretation='default').launch()