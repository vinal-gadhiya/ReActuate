"""Backend language boundary. Explicit instructions pass through verbatim."""

from core.goals import ACTION_SCHEMA, normalized


class PolicyInstructionAdapter:
    VERSION = 'semantic-action-v1'

    def __init__(self, backend):
        if backend not in ('gr00t', 'patchwam', 'awomo-0.5i'):
            raise ValueError(f'Unsupported policy instruction backend: {backend}')
        self.backend = backend
        # Independent profiles can be evaluated without changing either policy.
        # These basic fallbacks are reference phrasing, not verified competence.
        self.templates = {
            'gr00t': {'navigate': 'Navigate to the {destination}.', 'pickup': 'Pick up the {object}.',
                      'place': 'Place the {object} on the {destination}.'},
            'patchwam': {'navigate': 'Navigate to the {destination}.', 'pickup': 'Pick up the {object}.',
                        'place': 'Place the {object} on the {destination}.'},
            'awomo-0.5i': {'navigate': 'Navigate to the {destination}.', 'pickup': 'Pick up the {object}.',
                          'place': 'Place the {object} on the {destination}.'},
        }[backend]

    def render(self, action):
        if not isinstance(action, dict) or set(action) != set(ACTION_SCHEMA['properties']):
            raise ValueError('Action needs intent, object, destination and instruction')
        if any(not isinstance(value, str) or len(value) > 1000 for value in action.values()) or not action['intent'].strip():
            raise ValueError('Invalid semantic action fields')
        if action['instruction'].strip():
            return action['instruction']
        template = self.templates.get(normalized(action['intent']))
        if template is None:
            raise ValueError('This action requires an explicit backend instruction')
        if any(not action[field].strip() for field in ('object', 'destination') if '{' + field + '}' in template):
            raise ValueError('Missing semantic action arguments for instruction rendering')
        return template.format(**action)

    def validate_command(self, action, name, arguments):
        rendered = self.render(action)
        actual = arguments.get('instruction', f'Navigate to the {arguments.get("fixture", "")}.')
        if name not in ('run_groot', 'navigate') or actual != rendered:
            raise ValueError('Executable command must match the rendered semantic action verbatim')
        if name == 'navigate' and (normalized(action['intent']) != 'navigate'
                                  or normalized(action['destination']) != normalized(arguments['fixture'])):
            raise ValueError('Navigation fixture must match the semantic destination')
        return rendered
