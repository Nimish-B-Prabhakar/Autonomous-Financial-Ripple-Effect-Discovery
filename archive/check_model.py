import torch

state_dict = torch.load(
    "../models/finbert-event-classifier/model.pt", map_location="cpu", weights_only=True
)

print("Keys in model.pt:")
for key in state_dict.keys():
    print(f"  {key}: {state_dict[key].shape}")
