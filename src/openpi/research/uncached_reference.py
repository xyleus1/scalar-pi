# ruff: noqa: PLC0415, SLF001, I001 -- The independent oracle uses the audited private Gemma primitives lazily.
"""Independent PI05 prefix/expert arithmetic for numerical qualification only.

No DynamicCache or joint attention forward is used. Each branch performs its
own attention and output casts; prefix keys/values are widened for the expert.
This is an oracle, not an alternate production training or serving path.
"""


def require(value, message):
    if not value:
        raise ValueError(message)


def reference_inputs(model, observation):
    """Embedding-only preparation; no prefix transformer pass or cached K/V."""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    prefix, padding, ar = model.embed_prefix(images, image_masks, tokens, token_masks)
    return {
        "prefix": prefix,
        "padding": padding,
        "ar": ar,
        "state": state,
        "prefix_mask": model._prepare_attention_masks_4d(make_att_2d_masks(padding, ar)),
        "prefix_positions": torch.cumsum(padding, dim=1) - 1,
    }


def uncached_reference(model, inputs, actions, time_, *, trace=False):
    """Interleave native layers without using model.forward(), cache APIs or shared joint attention.

    A prefix token cannot attend to an action token. Its branch computes only
    prefix attention in its own dtype; action attention explicitly widens the
    prefix keys/values to the expert dtype. This defines the hybrid arithmetic.
    """
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from transformers.models.gemma import modeling_gemma as gemma

    vlm = model.paligemma_with_expert.paligemma.language_model
    expert = model.paligemma_with_expert.gemma_expert.model
    require(len(vlm.layers) == len(expert.layers), "Branches have different depth")
    prefix = inputs["prefix"].to(vlm.layers[0].self_attn.q_proj.weight.dtype)
    suffix, suffix_padding, suffix_ar, condition = model.embed_suffix(inputs["state"], actions, time_)
    suffix = suffix.to(expert.layers[0].self_attn.q_proj.weight.dtype)
    prefix_len = prefix.shape[1]
    padding = torch.cat((inputs["padding"], suffix_padding), dim=1)
    ar = torch.cat((inputs["ar"], suffix_ar), dim=1)
    full_mask = model._prepare_attention_masks_4d(make_att_2d_masks(padding, ar))
    suffix_mask = full_mask[:, :, prefix_len:, :]
    suffix_positions = inputs["padding"].sum(dim=-1)[:, None] + torch.cumsum(suffix_padding, dim=1) - 1
    rotations = (vlm.rotary_emb(prefix, inputs["prefix_positions"]), expert.rotary_emb(suffix, suffix_positions))
    prefix_trace = []

    for prefix_layer, suffix_layer in zip(vlm.layers, expert.layers, strict=True):
        projected = []
        gates = []
        for hidden, layer, cond, rotation in zip(
            (prefix, suffix), (prefix_layer, suffix_layer), (None, condition), rotations, strict=True
        ):
            normalized, gate = layer.input_layernorm(hidden, cond=cond)
            shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2)
            k = layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2)
            v = layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2)
            q, k = gemma.apply_rotary_pos_emb(q, k, *rotation)
            projected.append((q, k, v))
            gates.append(gate)
        pq, pk, pv = projected[0]
        sq, sk, sv = projected[1]
        if trace:
            prefix_trace.append((pk.detach().clone(), pv.detach().clone()))
        p_attention, _ = gemma.eager_attention_forward(
            prefix_layer.self_attn, pq, pk, pv, inputs["prefix_mask"], prefix_layer.self_attn.scaling
        )
        s_attention, _ = gemma.eager_attention_forward(
            suffix_layer.self_attn,
            sq,
            torch.cat((pk.to(sk.dtype), sk), dim=2),
            torch.cat((pv.to(sv.dtype), sv), dim=2),
            suffix_mask,
            suffix_layer.self_attn.scaling,
        )
        updated = []
        for hidden, attention, layer, gate, cond in zip(
            (prefix, suffix),
            (p_attention, s_attention),
            (prefix_layer, suffix_layer),
            gates,
            (None, condition),
            strict=True,
        ):
            attention = attention.reshape(*hidden.shape[:-1], -1).to(layer.self_attn.o_proj.weight.dtype)  # noqa: PLW2901
            residual = gemma._gated_residual(hidden, layer.self_attn.o_proj(attention), gate)
            normalized, mlp_gate = layer.post_attention_layernorm(residual, cond=cond)
            mlp = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
            updated.append(gemma._gated_residual(residual, mlp, mlp_gate))
        prefix, suffix = updated
    suffix, _ = expert.norm(suffix, cond=condition)
    output = model.action_out_proj(suffix[:, -model.config.action_horizon :].float())
    return (output, prefix_trace) if trace else output


def trajectory(velocity, noise, steps):
    import torch

    x = noise.clone()
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=x.device)
    t = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    values = [x.detach().cpu().numpy()]
    for _ in range(steps):
        x = x + dt * velocity(x, t.expand(x.shape[0]))
        values.append(x.detach().cpu().numpy())
        t = t + dt
    import numpy as np

    return np.stack(values, axis=1)


def torch_joint_velocity(model, observation, actions, time):
    """Native joint attention path on fixed inference inputs, before scalarization."""
    import torch

    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    prefix, prefix_masks, prefix_ar = model.embed_prefix(images, image_masks, tokens, token_masks)
    suffix, suffix_masks, suffix_ar, condition = model.embed_suffix(state, actions, time)
    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype != torch.float32:
        raise ValueError("Native joint parity requires the selected all-FP32 runtime")
    masks = torch.cat((prefix_masks, suffix_masks), dim=1)
    autoregressive = torch.cat((prefix_ar, suffix_ar), dim=1)
    (_, output), _ = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(masks, autoregressive)),
        position_ids=torch.cumsum(masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, condition],
    )
    return model.action_out_proj(output[:, -model.config.action_horizon :].float())
