import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class KeyStateTokenParameters(nnx.Module):
    """Small field-factorized vocabulary used by the opt-in key-state path."""

    def __init__(self, num_fields: int, max_num_values: int, width: int, rngs: nnx.Rngs):
        init = nnx.initializers.normal(stddev=width**-0.5)
        self.query_embeddings = nnx.Param(init(rngs.params(), (num_fields, width)))
        self.field_embeddings = nnx.Param(init(rngs.params(), (num_fields, width)))
        self.value_embeddings = nnx.Param(init(rngs.params(), (num_fields, max_num_values, width)))
        self.segment_embeddings = nnx.Param(init(rngs.params(), (2, width)))
        self.logit_bias = nnx.Param(jnp.zeros((num_fields, max_num_values), dtype=jnp.float32))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pi05_state_sequence_in_suffix = config.pi05_state_sequence_in_suffix
        self.key_state_token_mode = config.key_state_token_mode
        self.key_state_num_values = config.key_state_num_values
        self.key_state_loss_weight = config.key_state_loss_weight
        self.key_state_allowed_transitions = config.key_state_allowed_transitions
        self.key_state_initial_ids = config.key_state_initial_ids
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.pi05_state_sequence_in_suffix:
                self.state_sequence_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.key_state_token_mode != "disabled":
            self.key_state_token = KeyStateTokenParameters(
                len(self.key_state_num_values), max(self.key_state_num_values), paligemma_config.width, rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if self.key_state_token_mode != "disabled":
            self._validate_key_state_observation(obs, require_targets=False)
            previous_state_tokens = self._embed_key_state_values(obs.key_state_input_ids, segment_index=0)
            tokens.append(previous_state_tokens)
            input_mask.append(jnp.ones(previous_state_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * previous_state_tokens.shape[1]

            query_tokens = self._embed_key_state_queries(obs.state.shape[0])
            tokens.append(query_tokens)
            input_mask.append(jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [True] + [False] * (query_tokens.shape[1] - 1)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _validate_key_state_observation(self, obs: _model.Observation, *, require_targets: bool) -> None:
        expected_fields = len(self.key_state_num_values)
        if obs.key_state_input_ids is None:
            raise ValueError("key-state token mode requires key_state_input_ids")
        if obs.key_state_input_ids.shape[-1] != expected_fields:
            raise ValueError(
                f"expected {expected_fields} key-state input fields, got {obs.key_state_input_ids.shape[-1]}"
            )
        if require_targets:
            if obs.key_state_target_ids is None or obs.key_state_target_mask is None:
                raise ValueError("training key-state token mode requires target ids and target mask")
            if obs.key_state_target_ids.shape[-1] != expected_fields:
                raise ValueError("key_state_target_ids field count does not match schema")

    def _embed_key_state_queries(self, batch_size: int) -> jax.Array:
        params = self.key_state_token
        tokens = params.query_embeddings.value + params.field_embeddings.value
        return jnp.broadcast_to(tokens[None, ...], (batch_size, *tokens.shape))

    def _embed_key_state_values(self, ids: jax.Array, *, segment_index: int) -> jax.Array:
        ids = jnp.asarray(ids, dtype=jnp.int32)
        params = self.key_state_token
        field_indices = jnp.arange(len(self.key_state_num_values))[None, :]
        values = params.value_embeddings.value[field_indices, ids]
        return values + params.field_embeddings.value[None, ...] + params.segment_embeddings.value[segment_index]

    def _key_state_logits(self, query_hidden: jax.Array) -> jax.Array:
        hidden = query_hidden.astype(jnp.float32)
        hidden = hidden * jax.lax.rsqrt(jnp.mean(jnp.square(hidden), axis=-1, keepdims=True) + 1e-6)
        logits = jnp.einsum(
            "bfd,fkd->bfk", hidden, self.key_state_token.value_embeddings.value.astype(jnp.float32)
        )
        logits = logits + self.key_state_token.logit_bias.value[None, ...]
        valid = jnp.arange(logits.shape[-1])[None, :] < jnp.asarray(self.key_state_num_values)[:, None]
        return jnp.where(valid[None, ...], logits, -jnp.inf)

    def _select_key_state(self, logits: jax.Array, previous_ids: jax.Array) -> jax.Array:
        schema = tuple(self.key_state_num_values)
        if not schema:
            raise ValueError("key-state schema must contain at least one field")
        previous_ids = jnp.asarray(previous_ids, dtype=jnp.int32)
        class_ids = jnp.arange(logits.shape[-1])[None, :]

        if self.key_state_allowed_transitions is not None:
            selected = []
            for field_index, (_field_size, rows) in enumerate(
                zip(schema, self.key_state_allowed_transitions, strict=True)
            ):
                transition_table = [
                    [class_id in allowed_values for class_id in range(logits.shape[-1])]
                    for allowed_values in rows
                ]
                legal = jnp.asarray(transition_table, dtype=jnp.bool_)[previous_ids[:, field_index]]
                selected.append(jnp.argmax(jnp.where(legal, logits[:, field_index], -jnp.inf), axis=-1))
            return jnp.stack(selected, axis=-1).astype(jnp.int32)

        phase_size = schema[0]
        previous_phase = previous_ids[:, 0]
        next_phase = jnp.minimum(previous_phase + 1, phase_size - 1)
        phase_valid = class_ids < phase_size
        phase_legal = phase_valid & ((class_ids == previous_phase[:, None]) | (class_ids == next_phase[:, None]))
        phase = jnp.argmax(jnp.where(phase_legal, logits[:, 0], -jnp.inf), axis=-1)
        selected = [phase]
        for field_index, field_size in enumerate(schema[1:], start=1):
            if schema == (3, 3, 3) and field_index == 2:
                previous_button = previous_ids[:, field_index]
                button_legal_p1 = jnp.array(
                    [[False, True, False], [False, True, True], [False, False, True]], dtype=jnp.bool_
                )[previous_button]
                legal = jnp.where(
                    (phase == 1)[:, None], button_legal_p1, jnp.array([True, False, False])[None, :]
                )
            else:
                previous_value = previous_ids[:, field_index]
                valid = class_ids < field_size
                legal = valid & ((previous_value == 0)[:, None] | (class_ids == previous_value[:, None]))
            selected.append(jnp.argmax(jnp.where(legal, logits[:, field_index], -jnp.inf), axis=-1))
        return jnp.stack(selected, axis=-1).astype(jnp.int32)

    def _key_state_cross_entropy(self, logits: jax.Array, obs: _model.Observation) -> jax.Array:
        targets = jnp.asarray(obs.key_state_target_ids, dtype=jnp.int32)
        mask = jnp.asarray(obs.key_state_target_mask, dtype=jnp.float32)
        per_field = -jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1
        )[..., 0]
        return jnp.sum(per_field * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # State: (batch, state_dim) or (batch, seq_len, state_dim)
            state = obs.state
            if state.ndim == 2:
                state = state[:, None, :]
            num_state_tokens = state.shape[1]

            state_tokens = self.state_proj(state)
            tokens.append(state_tokens)
            input_mask.append(jnp.ones((state.shape[0], num_state_tokens), dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (num_state_tokens - 1))
        elif self.pi05_state_sequence_in_suffix:
            state = obs.state
            if state.ndim != 3:
                raise ValueError(
                    "PI0.5 suffix state conditioning requires state shape "
                    f"(batch, sequence_length, state_dim), got {state.shape}"
                )
            num_state_tokens = state.shape[1]
            state_tokens = self.state_sequence_proj(state)
            tokens.append(state_tokens)
            input_mask.append(jnp.ones((state.shape[0], num_state_tokens), dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (num_state_tokens - 1))

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if self.key_state_token_mode != "disabled":
            actions, _, _ = self.sample_actions_with_key_state(rng, observation, num_steps=num_steps, noise=noise)
            return actions
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @staticmethod
    def _resolve_action_condition_state_ids(selected_ids, action_condition_state_ids):
        if action_condition_state_ids is None:
            return selected_ids
        condition_ids = jnp.asarray(action_condition_state_ids, dtype=jnp.int32)
        if condition_ids.shape != selected_ids.shape:
            raise ValueError(
                "action_condition_state_ids must match predicted state shape: "
                f"expected {selected_ids.shape}, got {condition_ids.shape}"
            )
        return condition_ids

    def sample_actions_with_key_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_condition_state_ids: at.Int[at.Array, "b f"] | None = None,
    ) -> tuple[_model.Actions, jax.Array, jax.Array]:
        """Sample actions and explicitly return the predicted structured state."""
        if self.key_state_token_mode == "disabled":
            raise ValueError("sample_actions_with_key_state requires an enabled key-state token mode")
        observation = _model.preprocess_observation(None, observation, train=False)
        self._validate_key_state_observation(observation, require_targets=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        num_fields = len(self.key_state_num_values)
        state_logits = self._key_state_logits(prefix_out[:, -num_fields:])
        selected_ids = self._select_key_state(state_logits, observation.key_state_input_ids)

        if self.key_state_token_mode == "serial":
            condition_ids = self._resolve_action_condition_state_ids(selected_ids, action_condition_state_ids)
            current_tokens = self._embed_key_state_values(condition_ids, segment_index=1)
            current_mask = jnp.ones(current_tokens.shape[:2], dtype=jnp.bool_)
            current_ar = jnp.array([True] + [False] * (current_tokens.shape[1] - 1))
            local_attn = make_attn_mask(current_mask, current_ar)
            cached_attn = einops.repeat(prefix_mask, "b p -> b s p", s=current_tokens.shape[1])
            full_attn = jnp.concatenate([cached_attn, local_attn], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(current_mask, axis=-1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [current_tokens, None], mask=full_attn, positions=positions, kv_cache=kv_cache
            )
            prefix_mask = jnp.concatenate([prefix_mask, current_mask], axis=1)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cached_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cached_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions, selected_ids, state_logits
