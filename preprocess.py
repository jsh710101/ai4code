from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
from transformers import GPT2TokenizerFast, GPT2Model


def preprocess():
    cuda = 0
    data_dir = 'data'
    model_dir = 'model'
    split = 'test'
    file_batch_size = 16
    batch_size = 8

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

    files = list(Path(f'{data_dir}/{split}').iterdir())
    if split == 'train':
        cell_orders = pd.read_csv(f'{data_dir}/train_orders.csv', index_col='id')

    Path(split).mkdir(parents=True, exist_ok=True)

    tokenizer = {
        'ttc': GPT2TokenizerFast.from_pretrained(f'{model_dir}/codeparrot-small-text-to-code'),
        'ctt': GPT2TokenizerFast.from_pretrained(f'{model_dir}/codeparrot-small-code-to-text')}
    model = {
        'ttc': GPT2Model.from_pretrained(f'{model_dir}/codeparrot-small-text-to-code').to(device),
        'ctt': GPT2Model.from_pretrained(f'{model_dir}/codeparrot-small-code-to-text').to(device)}

    for model_type in ['ttc', 'ctt']:
        tokenizer[model_type].pad_token = tokenizer[model_type].eos_token
        model[model_type].eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(files), file_batch_size)):
            file_batch = files[i:i + file_batch_size]
            df_batch = [pd.read_json(file) for file in file_batch]

            sources = {'markdown': [], 'code': []}
            for df in df_batch:
                for cell_type in ['markdown', 'code']:
                    sources[cell_type] += df[df['cell_type'] == cell_type]['source'].tolist()
                sources['code'].append('exit()  # Final Conclusion End')

            embeds = {
                'ttc': {'markdown': [], 'code': []},
                'ctt': {'markdown': [], 'code': []}}

            for model_type in ['ttc', 'ctt']:
                for cell_type in ['markdown', 'code']:
                    for j in range(0, len(sources[cell_type]), batch_size):
                        source_batch = sources[cell_type][j:j + batch_size]

                        if cell_type == 'markdown':
                            if model_type == 'ttc':
                                source_batch = [f'"""\n{source.strip()}\n"""' for source in source_batch]
                            else:
                                source_batch = [f'"""\nExplanation: {source.strip()}\nEnd of explanation\n"""' for source in source_batch]

                        inputs = tokenizer[model_type](source_batch, padding=True, truncation=True,
                                                       max_length=1024, return_tensors='pt').to(device)
                        outputs = model[model_type](**inputs).last_hidden_state

                        embed = (outputs * inputs.attention_mask[..., None]).sum(1) / inputs.attention_mask.sum(-1)[:, None]
                        embeds[model_type][cell_type].append(embed)
                    embeds[model_type][cell_type] = torch.cat(embeds[model_type][cell_type])

            for file, df in zip(file_batch, df_batch):
                data = {}
                for cell_type in ['markdown', 'code']:
                    cells = df[df['cell_type'] == cell_type]
                    num_cells = len(cells) if cell_type == 'markdown' else len(cells) + 1

                    data[f'{cell_type}_order'] = cells.index.tolist()

                    for model_type in ['ttc', 'ctt']:
                        data[f'{model_type}_{cell_type}_embeds'] = embeds[model_type][cell_type][:num_cells].clone()
                        embeds[model_type][cell_type] = embeds[model_type][cell_type][num_cells:]

                if split == 'train':
                    data['cell_order'] = cell_orders.loc[file.stem, 'cell_order'].split()

                    num_codes, labels = 0, {}
                    for cell_id in data['cell_order']:
                        if df.loc[cell_id, 'cell_type'] == 'markdown':
                            labels[cell_id] = num_codes
                        else:
                            num_codes += 1
                    data['labels'] = torch.tensor([labels[markdown_id] for markdown_id in data['markdown_order']], device=device)

                    for cell_type in ['markdown', 'code']:
                        data[f'{cell_type}_order_i'] = [data['cell_order'].index(cell_id) for cell_id in data[f'{cell_type}_order']]

                torch.save(data, f'{split}/{file.stem}.pt')


preprocess()
