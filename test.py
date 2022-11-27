from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel


def test():
    cuda = 0
    model_dir = 'model'
    model_type = {'markdown': 'ttc', 'code': 'ctt'}
    batch_size = 32

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

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

        for key in ['markdown', 'markdown_mask', 'code', 'code_mask']:
            model_data[key] = pad_sequence(model_data[key], batch_first=True, padding_value=0)

        for key in ['markdown_order', 'code_order']:
            eval_data[key] = [data[key] for data in batch]

        return model_data, eval_data

    test_files = list(Path('test').iterdir())

    test_dataset = MyDataset(test_files)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model_prefix = f'{model_type["markdown"]}_{model_type["code"]}'
    markdown_model = BertModel.from_pretrained(f'{model_dir}/{model_prefix}_markdown', add_pooling_layer=False).to(device)
    code_model = BertModel.from_pretrained(f'{model_dir}/{model_prefix}_code', add_pooling_layer=False).to(device)

    markdown_model.eval(), code_model.eval()

    cell_orders = []
    with torch.no_grad():
        for model_data, eval_data in tqdm(test_dataloader):
            test_keys = ['markdown', 'markdown_mask', 'code', 'code_mask']
            markdown, markdown_mask, code, code_mask = [model_data[key].to(device) for key in test_keys]

            markdown = markdown_model(attention_mask=markdown_mask, inputs_embeds=markdown).last_hidden_state
            code = code_model(attention_mask=code_mask, inputs_embeds=code).last_hidden_state

            similarity = torch.bmm(markdown, code.transpose(1, 2))
            similarity = similarity + 10000 * (code_mask[:, None] - 1.0)

            predictions = similarity.argmax(-1).tolist()

            markdown_orders, code_orders = eval_data['markdown_order'], eval_data['code_order']
            for prediction, markdown_order, code_order in zip(predictions, markdown_orders, code_orders):
                prediction = prediction[:len(markdown_order)]

                cell_order = []
                for i, code in enumerate(code_order):
                    cell_order += [markdown for markdown, pred in zip(markdown_order, prediction) if pred == i]
                    cell_order.append(code)
                cell_order += [markdown for markdown, pred in zip(markdown_order, prediction) if pred == len(code_order)]

                cell_orders.append(' '.join(cell_order))

    submission = {'cell_order': {file.stem: cell_order for file, cell_order in zip(test_files, cell_orders)}}
    submission = pd.DataFrame.from_dict(submission)
    submission.to_csv('submission.csv', index_label='id')


test()
