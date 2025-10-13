import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from PIL import Image
import random

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


@jax.jit
def print_f(x):
  jax.debug.print("🤯 {x} 🤯", x=x)
  return x

class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config, text_encoder=None):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, text_encoder=text_encoder, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}


    # det_size = config.dyn[config.dyn.typ].deter  # e.g. 200
    # sto_size = np.prod(config.dyn[config.dyn.typ].stoch)  # e.g. 30
    # wm_feat_dim = det_size + sto_size               # e.g. 230
    # # 2) add on your text MLP output, if using VLM:
    # combined_dim = wm_feat_dim + (proj_dim if config.use_vlm else 0)
    # # 3) create the proper `elements.Space` for your policy input:
    # feat_space = elements.Space(np.float32, (combined_dim,))
    # self.pol = embodied.jax.MLPHead(
    #   feat_space,   # now matches jnp.concatenate([feat_tensor, proj_text], -1)
    #   outs,         # same output spec as before
    #   **config.policy,
    #   name='pol'
    # )
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    
    stop_cfg = getattr(config, 'stophead', config.conhead)
    stop_space = elements.Space(np.int32, (), 0, 2)
    self.stop = embodied.jax.MLPHead(stop_space, **stop_cfg, name='stop')
    self.modules.append(self.stop)
    self.config.setdefault('stop_threshold', 0.5)  # for hard gating if you want

    # self.null = NullVec(name='instr_null')
    # self.modules.append(self.null)  # so its params get optimized

    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    scales['bc'] = 1.0
    scales.setdefault('stop', 1.0)  # tune 0.3–2.0
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol|stop|instr_null)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    carry = self.init_policy(batch_size)
    return carry

  def init_report(self, batch_size):
    return self.init_policy(batch_size)
  
  def sample_with_vlm(self, captions, batch_size, time, dtype) -> jnp.ndarray:
    """
    Given a list of PIL frames and corresponding low-level action strings,
    returns a JAX array of shape (batch, hidden_dim) as the frozen text embedding.
    """
    if captions is None:
        captions = ['dummy'] * batch_size
     # 2. tokenize
    inputs = self.tokenizer(
        captions,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    # 3. encode (frozen)
    outputs = self.text_encoder(
        **inputs,
        params=self.text_encoder.params,
        train=False,
    )
    # shape: (batch, seq_len, hidden_dim)
    hidden = outputs.last_hidden_state
    # 4. mean-pool across sequence length
    pooled = jnp.mean(hidden, axis=1).astype(dtype)
    return pooled  # (batch, hidden_dim)

  def policy(self, carry, obs, mode='train', return_stop_token=False):
    
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)

    out = {}
    feat_vec_base = self.feat2tensor(feat)

    instr = enc_entry['instr']
    instr = instr * (1.0 / jnp.sqrt(instr.shape[-1]).astype(instr.dtype))
    stop_inp = jnp.concatenate([feat_vec_base, nn.cast(instr)], -1)
    p_stop = self.stop(stop_inp, 1).prob(1)  # [B]
    # Learned null vector (shared across batch/time at eval)
    # null_vec = self.null(instr)
    # gate = (1.0 - sg(p_stop))[..., None]  # [B,1]
    # blended = gate * instr + (1.0 - gate) * nn.cast(null_vec)  # [B,D]
    # feat_vec = jnp.concatenate([feat_vec_base, nn.cast(blended)], -1)
    feat_vec = jnp.concatenate([feat_vec_base, nn.cast(instr)], -1)

    policy = self.pol(feat_vec, bdims=1)
    # act = sample(policy)

    if False:
       # Generate uniform random actions for each action key
        uniform_actions = {}
        for key, space in self.act_space.items():
            batch_shape = feat_vec.shape[:-1]  # Get batch dimensions
            action_shape = (*batch_shape, *space.shape)  # Combine batch + action dims
            
            if space.discrete:
                # For discrete actions, sample uniformly from the action range
                # Use the space bounds directly without trying to extract concrete values
                uniform_actions[key] = jax.random.randint(
                    nj.seed(), 
                    shape=action_shape, 
                    minval=space.low, 
                    maxval=space.high
                )
            else:
                # For continuous actions, sample uniformly from the action range
                uniform_actions[key] = jax.random.uniform(
                    nj.seed(),
                    shape=action_shape,
                    minval=space.low,
                    maxval=space.high
                )
        
        # Sample from policy
        policy_actions = sample(policy)
        
        # Decide whether to use uniform or policy action
        use_uniform = jax.random.uniform(nj.seed(), shape=feat_vec.shape[:-1]) < 0.1
        
        # Mix uniform and policy actions
        act = {}
        for key in self.act_space.keys():
            # Expand use_uniform to match action dimensions if needed
            if len(self.act_space[key].shape) > 0:
                use_uniform_expanded = jnp.expand_dims(use_uniform, axis=-1)
                for _ in range(len(self.act_space[key].shape) - 1):
                    use_uniform_expanded = jnp.expand_dims(use_uniform_expanded, axis=-1)
            else:
                use_uniform_expanded = use_uniform
                
            act[key] = jnp.where(
                use_uniform_expanded,
                uniform_actions[key],
                policy_actions[key]
            )

    else:
        # Original behavior - just sample from policy
        act = sample(policy)
    
    p_stop = self.stop(jnp.concatenate([feat_vec_base, nn.cast(instr)], -1), 1).prob(1)
    act = dict(act)
    act['_stop_prev'] = (p_stop > 0.5).astype(jnp.float32)  # or use a cfg threshold
    
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry,  dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    if return_stop_token:
      return carry, act, out, p_stop
    else:
      return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training, captions=None):

    # inside Agent.loss
    # if jax.process_index() == 0:
    # jax.debug.print("row0: {}", obs['action_ids'][0,])
    # f(obs['action_ids'][0,])

    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    # enc_carry, enc_entries, tokens = self.enc(
    #     enc_carry, obs, reset, training)
    enc_carry, enc_entries, tokens = self.enc(
       enc_carry, obs, reset, training)

    # Build shifted stop token (prev) from BC window ends; 1 at last step, else 0; shift by 1
    demo_ids = obs.get('action_ids', None)
    prevact_aug = prevact
    if demo_ids is not None:
      mask_bool = (demo_ids != 0)
      if mask_bool.ndim == 3:
        mask_bool = mask_bool.any(axis=-1)
      next_mask_bool = jnp.concatenate(
          [mask_bool[:, 1:], jnp.zeros_like(mask_bool[:, :1])], 1)
      stop_label = (mask_bool & (~next_mask_bool)).astype(jnp.float32)
      stop_prev = jnp.concatenate(
          [jnp.zeros_like(stop_label[:, :1]), stop_label[:, :-1]], 1)
      prevact_aug = dict(prevact)
      prevact_aug['_stop_prev'] = stop_prev

    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact_aug, reset, training)


    # dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
    #     dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)

    instr = enc_entries['instr']
    instr = instr * (1.0 / jnp.sqrt(instr.shape[-1]).astype(instr.dtype))

    demo_ids = obs['action_ids']
    mask_bool = (demo_ids != 0)                               # (B,T) bool
    mask_f32  = mask_bool.astype(jnp.float32)                    # for weighting

    oh = jnp.clip(demo_ids, 0)

    feat_vec_base = self.feat2tensor(repfeat)

    # mask: [B,T] bool where BC actions are provided
    next_mask_bool = jnp.concatenate(
        [mask_bool[:, 1:], jnp.zeros_like(mask_bool[:, :1])], 1  # (B,T) bool
    )
    stop_label = (mask_bool & (~next_mask_bool)).astype(jnp.float32)   # 1 at last BC step
    # stop_weight = mask_f32
    stop_weight = stop_label
    if stop_label.ndim > 2:
      stop_label = stop_label[:, :, 0]
      stop_weight = stop_weight[:, :, 0]
    # stop_inp_train = jnp.concatenate([feat_vec_base, nn.cast(instr)], -1)  # in loss()
    # stop_pred = self.stop(stop_inp_train, 2)
    # losses['stop'] = stop_weight * stop_pred.loss(stop_label)

    # p_stop_train = stop_pred.prob(1)                # [B,T]
    # null_vec = self.null(instr)
    # gate = (1.0 - sg(p_stop_train))[..., None]
    # blended_instr = gate * instr + (1.0 - gate) * nn.cast(null_vec)
    # feat_vec = jnp.concatenate([feat_vec_base, nn.cast(blended_instr)], -1)
    stop_inp_train = jnp.concatenate([feat_vec_base, nn.cast(instr)], -1)  # in loss()
    stop_pred = self.stop(stop_inp_train, 2)
    print(stop_label.shape)
    losses['stop'] = stop_weight * stop_pred.loss(stop_label)
    # Directly use instruction embeddings during the action sequence; null otherwise.
    feat_vec = jnp.concatenate([feat_vec_base, nn.cast(instr)], -1)

    action_policy = self.pol(feat_vec, 2)['action']
    if 'Categorical' in action_policy.__class__.__name__:
      logp = action_policy.logp(oh)
      losses['bc'] = -(logp * mask_f32) / jnp.maximum(mask_f32.sum(axis=1, keepdims=True), 1.0)
    else:
      logp = action_policy.output.logp(oh)
      losses['bc'] = -(logp * mask_f32).sum(-1) / jnp.maximum(mask_f32.sum(-1).sum(axis=1, keepdims=True), 1.0)

    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    
    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # has_supervised_data = jnp.any(mask > 0)
    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    instrK     = instr[:, -K:]
    instr_flat = instrK.reshape((B * K, -1))

    def policyfn(feat):                                     # feat: (B*K, F)
      f = self.feat2tensor(feat)                          # (B*K, F)
      stop_inp = jnp.concatenate([f, nn.cast(instr_flat)], -1)   # [B*K,F+D]
      p_stop_im = self.stop(stop_inp, 1).prob(1)                 # [B*K]
      f = jnp.concatenate([f, nn.cast(instr_flat)], -1)
      act = sample(self.pol(f, 1))
      # Feed previous-step stop (hard 0/1) into RSSM
      act = dict(act)
      act['_stop_prev'] = (p_stop_im > 0.5).astype(jnp.float32)
      return act
    
    # policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp_wm = self.feat2tensor(imgfeat)

    instr_time = jnp.repeat(instr_flat[:, None, :], inp_wm.shape[1], axis=1)  # [B*K, H+1, D]
    inp = jnp.concatenate([inp_wm, nn.cast(instr_time)], -1)                     # [B*K,H+1,F+D]


    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp_wm, 2).pred(),
        self.con(inp_wm, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    
    # losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    demo_mask = demo_ids != 0
    if demo_mask.ndim == 3:
      demo_mask = demo_mask.any(axis=-1)

    win_mask = demo_mask[:, -K:]                   # [B, K]
    rl_step_weight = (~win_mask).astype(instr.dtype)       # 1.0 where unlabeled, else 0.0
    # keep RL active; downweight if a start window is mostly supervised
    rl_losses = {k: v.mean(1).reshape((B, K)) * rl_step_weight for k, v in los.items()}
    losses.update(rl_losses)


    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      # inp_wm = self.feat2tensor(feat)

      p_stop_train = stop_pred.prob(1)[:, -K:]
      inp_aug = jnp.concatenate([self.feat2tensor(feat), nn.cast(instrK)], -1)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp_aug, 2),
          self.slowval(inp_aug, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    # assert set(losses.keys()) == set(self.scales.keys()), (
    #     sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
  
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  # for k in act:
  #   metrics[f'ent/{k}'] = ents[k].mean()
  #   if hasattr(policy[k], 'minent'):
  #     lo, hi = policy[k].minent, policy[k].maxent
  #     metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  for k, head in policy.items():
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(head, 'minent'):
      lo, hi = head.minent, head.maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)


  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
