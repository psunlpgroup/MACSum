prompt_checkpoint_path = "/data/yfz5488/PCS/output/BART-large_cnndm_prompt_2022-07-12/last_checkpoint/pytorch_model.bin"
prefix_checkpoint_path = "/data/yfz5488/PCS/output/BART-large_cnndm_freeze_2022-7-20/checkpoint-89000/pytorch_model.bin"
output_path = "/data/yfz5488/PCS/output/checkpoints/prompt+prefix/pytorch_model.bin"

prefix_keywords = ['wte', 'control_trans', 'input_tokens']
import torch

prompt_parameters = torch.load(prompt_checkpoint_path)
prefix_parameters = torch.load(prefix_checkpoint_path)


for name, tensor in prompt_parameters.items():
    if any(x in name for x in prefix_keywords):
        prompt_parameters[name] = prefix_parameters[name]

for name, tensor in prefix_parameters.items():
    print(name, tensor.shape)

torch.save(prompt_parameters, output_path)




