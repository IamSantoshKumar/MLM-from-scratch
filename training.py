import os
import gc  
import torch
import config
import dataset
import argparse
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM, AdamW

mlm_parser = argparse.ArgumentParser(description='MLM input file')

tokenizer = BertTokenizer.from_pretrained(config.CONFIG.model_name)
model = BertForMaskedLM.from_pretrained(config.CONFIG.model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_file(filename: str=None):
    ABSOLUTE_PATH = os.path.join(args.HOME_DIR, filename)
    with open(ABSOLUTE_PATH) as f:
        text = f.read().split("\n")
    return text


def mlm_preprocessing(text):
    selection =list()
    inputs = tokenizer(text, return_tensors="pt", max_length=config.CONFIG.max_length, padding="max_length", truncation=True)
    inputs['labels'] = inputs.input_ids.detach().clone()
    rand = (torch.rand(inputs.input_ids.shape)<config.CONFIG.mask_proba)
    mask_arr = rand * (inputs.input_ids!=101) * (inputs.input_ids!=102) * (inputs.input_ids!=0)
    for i in range(mask_arr.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(mask_arr.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    return inputs


def train(inputs):
    model.to(device)
    model.train()
    mlm_dataset = dataset.MLMDataset(inputs)
    dataloader = torch.utils.data.DataLoader(mlm_dataset, batch_size=config.CONFIG.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.CONFIG.lr)

    for epoch in range(config.CONFIG.epochs):
        loop = tqdm(dataloader, leave=True)
        scaler =  torch.cuda.amp.GradScaler()
        train_loss=0
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            train_loss +=loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description(f"EPOCH: {epoch}")
            loop.set_postfix(loss=train_loss/len(dataloader))
        gc.collect()
        torch.cuda.empty_cache()

    return model


if __name__=="__main__":

    mlm_parser.add_argument('filename',
                       metavar='filename',
                       type=str,
                       help='the name of the file')


    mlm_parser.add_argument('HOME_DIR',
                       metavar='HOME_DIR',
                       type=str,
                       help='home directory')

    args = mlm_parser.parse_args()

    text = read_file(args.filename)
    input_text = mlm_preprocessing(text)
    model = train(input_text)
    model.save_pretrained(args.HOME_DIR)
    tokenizer.save_pretrained(args.HOME_DIR)
