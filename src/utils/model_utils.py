from transformers import (
    LlamaConfig, 
    LlamaForCausalLM,
    GPT2TokenizerFast,
)
from pathlib import Path


def trainable_model_params(model):
    '''
    Method to get number of trainable model params

    Input:
        - model: pytorch model

    Output:
        - sum of trainable parameters of model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_default_model(config, tokenizer_path, init):
    # print(f"Loading model {config['model']['name']} with config: {config['model']}")
    # print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    if config['model']['type'] == 'Llama':
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=256,
            hidden_size=config['model']['hidden_size'],
            intermediate_size=config['model']['intermediate_size'],
            num_hidden_layers=config['model']['n_layer'],
            num_attention_heads=config['model']['n_head'],
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        if init:
            model = LlamaForCausalLM(model_config)
        else:
            model_path = Path(f"./output/models/random") / config['model']['name'].lower()
            print(f"Loading model from {config['model']['name']} and weights from {model_path}")
            model = LlamaForCausalLM.from_pretrained(f'{model_path}')
    else:
        raise ValueError('Model type not supported!')

    return model, tokenizer


from transformers import LlamaForCausalLM

def get_model(model_config, dataset, checkpoint=False, epoch_to_load=None, init=False, tokenizer_vocab_size=None):
    model, tokenizer = get_default_model(
        model_config, 
        f'data/{dataset}/tokenizers/tokenizer_{tokenizer_vocab_size}.json',
        init=True  # Always create model fresh here
    )

    model_path = Path(f"./output/models/{dataset}") / model_config['model']['name'].lower()
    if init:
        model_path = Path(f"./output/models/random") / model_config['model']['name'].lower()
        model_path.mkdir(parents=True, exist_ok=True)
        # ✅ Save the randomly initialized model so it can be reloaded later
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"[INIT] Saved randomly initialized model to {model_path}")

    if checkpoint:
        model = LlamaForCausalLM.from_pretrained(f'{model_path}-{epoch_to_load}e')

    # Print number of trainable params
    print(f"Number of trainable params: {trainable_model_params(model) / 1e6:.2f}M")

    return model, tokenizer, model_path
