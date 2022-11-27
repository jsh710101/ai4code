from tqdm import tqdm
from time import perf_counter
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertModel
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_


def train():
    cuda = 0
    model_dir = 'model'
    model_type = {'markdown': 'ttc', 'code': 'ctt'}
    batch_size = 32
    learning_rate = 5e-5
    weight_decay = 1e-7
    max_norm = 1.0
    num_epochs = 100
    eval_interval = 10
    save_interval = 10

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    def get_elapsed_time(start, end):
        seconds = int(end - start)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}'

    class MyDataset(Dataset):
        def __init__(self, files):
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            return torch.load(self.files[index], map_location='cpu')

    def collate_fn(batch):
        model_data, eval_data = {}, {}

        for cell_type in ['markdown', 'code']:
            model_data[cell_type] = [data[f'{model_type[cell_type]}_{cell_type}_embeds'] for data in batch]
            model_data[f'{cell_type}_mask'] = [torch.ones(len(cell)) for cell in model_data[cell_type]]
        model_data['labels'] = [data['labels'] for data in batch]

        for key in ['markdown', 'markdown_mask', 'code', 'code_mask', 'labels']:
            padding_value = -100 if key == 'labels' else 0
            model_data[key] = pad_sequence(model_data[key], batch_first=True, padding_value=padding_value)

        for key in ['markdown_order_i', 'code_order_i']:
            eval_data[key] = [data[key] for data in batch]

        return model_data, eval_data

    files = list(Path('train').iterdir())
    train_files, valid_files = train_test_split(files, test_size=0.2, random_state=0)

    train_dataset, valid_dataset = MyDataset(train_files), MyDataset(valid_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)

    markdown_config = BertConfig(
        vocab_size=4, hidden_size=768, num_hidden_layers=3, num_attention_heads=12, intermediate_size=1536,
        max_position_embeddings=2048, position_embedding_type='none')
    code_config = BertConfig(
        vocab_size=4, hidden_size=768, num_hidden_layers=3, num_attention_heads=12, intermediate_size=1536,
        max_position_embeddings=2048, position_embedding_type='absolute')

    markdown_model = BertModel(markdown_config, add_pooling_layer=False).to(device)
    code_model = BertModel(code_config, add_pooling_layer=False).to(device)

    parameters = list(markdown_model.parameters()) + list(code_model.parameters())

    criterion = CrossEntropyLoss()
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(1, num_epochs + 1):
        start_time = perf_counter()

        markdown_model.train(), code_model.train()
        train_loss = 0

        for model_data, _ in tqdm(train_dataloader):
            train_keys = ['markdown', 'markdown_mask', 'code', 'code_mask', 'labels']
            markdown, markdown_mask, code, code_mask, labels = [model_data[key].to(device) for key in train_keys]

            markdown = markdown_model(attention_mask=markdown_mask, inputs_embeds=markdown).last_hidden_state
            code = code_model(attention_mask=code_mask, inputs_embeds=code).last_hidden_state

            similarity = torch.bmm(markdown, code.transpose(1, 2))
            similarity = similarity + 10000 * (code_mask[:, None] - 1.0)

            predictions = similarity.view(-1, similarity.shape[-1])
            loss = criterion(predictions, labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, max_norm)
            optimizer.step()

            train_loss += loss.item() * len(labels) / len(train_dataset)

        valid_metric = -1
        if epoch % eval_interval == 0:
            markdown_model.eval(), code_model.eval()
            numerator, denominator = 0, 0

            with torch.no_grad():
                for model_data, eval_data in tqdm(valid_dataloader):
                    valid_keys = ['markdown', 'markdown_mask', 'code', 'code_mask']
                    markdown, markdown_mask, code, code_mask = [model_data[key].to(device) for key in valid_keys]

                    markdown = markdown_model(attention_mask=markdown_mask, inputs_embeds=markdown).last_hidden_state
                    code = code_model(attention_mask=code_mask, inputs_embeds=code).last_hidden_state

                    similarity = torch.bmm(markdown, code.transpose(1, 2))
                    similarity = similarity + 10000 * (code_mask[:, None] - 1.0)

                    predictions = similarity.argmax(-1).tolist()

                    markdown_order_i, code_order_i = eval_data['markdown_order_i'], eval_data['code_order_i']
                    for prediction, markdown_order, code_order in zip(predictions, markdown_order_i, code_order_i):
                        prediction = prediction[:len(markdown_order)]

                        cell_order = []
                        for i, code in enumerate(code_order):
                            cell_order += [markdown for markdown, pred in zip(markdown_order, prediction) if pred == i]
                            cell_order.append(code)
                        cell_order += [markdown for markdown, pred in zip(markdown_order, prediction) if pred == len(code_order)]

                        num_swaps, num_cells = 0, len(cell_order)
                        for i in range(num_cells):
                            for j in range(i + 1, num_cells):
                                if cell_order[i] > cell_order[j]:
                                    num_swaps += 1

                        numerator += num_swaps
                        denominator += num_cells * (num_cells - 1)

                valid_metric = 1 - 4 * (numerator / denominator)

        if epoch % save_interval == 0:
            model_prefix = f'{model_type["markdown"]}_{model_type["code"]}'
            markdown_model.save_pretrained(f'{model_dir}/{model_prefix}_markdown')
            code_model.save_pretrained(f'{model_dir}/{model_prefix}_code')

        scheduler.step()
        end_time = perf_counter()
        elapsed_time = get_elapsed_time(start_time, end_time)

        print(f'[Epoch {epoch:3}/{num_epochs}] Loss: {train_loss:6.4f} | Metric: {valid_metric:6.4f} | {elapsed_time}')


train()
