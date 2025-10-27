import torch


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Exponential Moving Average of model parameters.
    Improves inference stability and audio quality.
    """
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()

    for key in ema_state.keys():
        ema_state[key].copy_(
            decay * ema_state[key] + (1 - decay) * model_state[key]
        )
