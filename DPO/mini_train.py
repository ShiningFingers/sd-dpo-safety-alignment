import torch
from train import setup_lora_model, train

pipe, unet, ref_unet = setup_lora_model()

unet, loss_history, acc_history = train(
    unet, ref_unet, pipe,
    num_epochs=50,
    checkpoint_interval=5
)

print("loss:", loss_history)
print("acc :", acc_history)