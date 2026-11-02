"""Resolve public backend identity and execution interval before any motion."""


def resolve_policy_execution(metadata, action_steps=None, expected_policy=None, default_steps=8):
    backend = metadata.get('policy', 'unknown')
    family = {'gr00t_n1.5': 'gr00t'}.get(backend, backend)
    if expected_policy is not None and family != expected_policy:
        raise ValueError(f'Expected policy {expected_policy!r}, but inference reports {backend!r}; '
                         'check the running inference service and Compose overlay')
    horizon = metadata.get('action_horizon')
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('Policy action horizon must be a positive integer')
    steps = (metadata.get('recommended_action_steps', min(default_steps, horizon))
             if action_steps is None else action_steps)
    if type(steps) is not int or not 1 <= steps <= horizon:
        raise ValueError('Action steps must be a positive integer within the policy horizon')
    if metadata.get('required_action_steps') is not None and steps != metadata['required_action_steps']:
        raise ValueError(f'{backend} requires exactly {metadata["required_action_steps"]} action steps per prediction')
    return steps


def policy_family(metadata):
    return {'gr00t_n1.5': 'gr00t'}.get(metadata.get('policy'), metadata.get('policy'))


def validate_policy_interval(metadata, step, steps, max_steps):
    quantum = metadata.get('required_action_steps')
    if quantum is not None and (step % quantum or (steps % quantum and step + steps < max_steps)):
        raise ValueError(f'{metadata["policy"]} requires {quantum}-step aligned requests, except the final episode remainder')
