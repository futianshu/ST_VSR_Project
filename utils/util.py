import torch
from diffusers import SD3Transformer2DModel
from peft import LoraConfig

def show_param(model ,static_dict, print_param=False):
    for key in static_dict:
        frozen_layers = list(filter(lambda param : key in param[0], model.named_parameters()))
        for name, param in frozen_layers:
            print(name,param.requires_grad)
            if print_param:
                print(param, param.dtype)
                print(param.grad)
            else:
                print(param[0])

def load_lora_state_dict(state_dict, model, adapter_name="default"):
    loaded_keys = 0
    for n, p in model.named_parameters():
        if adapter_name in n:
            target_name = n.replace(f".{adapter_name}", "")
            # 智能匹配：在 state_dict 寻找最匹配的 key 
            match_key = next((k for k in state_dict.keys() if target_name in k), None)
            if match_key:
                p.data.copy_(state_dict[match_key])
                state_dict.pop(match_key)
                loaded_keys += 1
    print(f"✅ 成功智能匹配并加载了 {loaded_keys} 个 LoRA 权重！")
    if len(state_dict) > 0:
        print(f"Warning: {len(state_dict)} keys not loaded")
        print(state_dict.keys())
        
def get_trainable_param(model):
    train_param = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            train_param.append(n)
    return train_param
                