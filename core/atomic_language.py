"""Validate policy language against versioned atomic references without rewriting it."""

from functools import lru_cache
import re

from core.simulator_context import load_atomic_task_catalog


# Slots contain entity descriptions, never another action or control program.
SLOT = r"[\w][\w '\-/]*?"
ACTION_CLAUSE = re.compile(
    r'\b(?:and|then|before|after)\s+(?:(?:carefully|gently)\s+)?'
    r'(?:pick|place|put|open|close|navigate|turn|press|push|pull|move|lift|release|grasp|pour|wash|stir)\b',
    re.IGNORECASE)


def gripper_release_instruction(instruction):
    return isinstance(instruction, str) and instruction.strip().casefold().rstrip('.') == 'open the gripper'


RELEASE_EXTENSION = {'id': 'controlled_gripper_release', 'template': 'Open the gripper.',
    'source_kind': 'operator_requested_harness_extension', 'source_tasks': [],
    'meaning': 'A language request to the frozen policy, not a numeric gripper override. '
               'Only as a verified low-height release intervention during placement.'}


def _pattern(template):
    """Instantiate documented alternatives, named slots, and door(s) notation."""
    parts, cursor = [], 0
    for match in re.finditer(r'\{([^{}]+)\}|\[([^\[\]]+)\]|\(s\)', template):
        parts.append(re.escape(template[cursor:match.start()]))
        token = match[1] or match[2]
        if match[0] == '(s)':
            parts.append(r'(?:s|\(s\))?')
        elif '/' in token:
            parts.append('(?:' + '|'.join(re.escape(option) for option in token.split('/')) + ')')
        else:
            parts.append('(' + SLOT + ')')
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    expression = ''.join(parts)
    # Case and final punctuation do not change the action vocabulary or format.
    if expression.endswith(r'\.'):
        expression = expression[:-2] + r'\.?'
    return re.compile(expression, re.IGNORECASE)


class AtomicInstructionGrounder:
    mode = 'atomic'
    VERSION = 'atomic-catalog-language-v2'

    def __init__(self, catalog=None):
        self.catalog = catalog if catalog is not None else load_atomic_task_catalog()
        self.tasks = {task['name']: task for task in self.catalog['tasks']}
        self.forms = self.catalog['instruction_forms']
        for sources in self.catalog.get('skill_reference_tasks', {}).values():
            if not sources or any(name not in self.tasks for name in sources):
                raise ValueError('Invalid atomic skill reference provenance')
        identifiers = set()
        self.patterns = []
        for form in self.forms:
            if form['id'] in identifiers or not form['source_tasks'] or any(
                    name not in self.tasks for name in form['source_tasks']):
                raise ValueError('Invalid atomic instruction form provenance')
            identifiers.add(form['id'])
            self.patterns.append((form['id'], form['source_tasks'], form['template'], _pattern(form['template'])))
        # Preserve specialized wording (buttons, levers, racks, temperatures, etc.)
        # directly from all atomic descriptions, rather than inventing synonyms.
        self.patterns.extend((task['name'], [task['name']], task['description'], _pattern(task['description']))
                             for task in self.tasks.values())

    def validate(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ValueError('Atomic instruction must be nonempty text of at most 1000 characters')
        if gripper_release_instruction(instruction):
            return {'version': self.VERSION, 'form': RELEASE_EXTENSION['id'],
                    'template': RELEASE_EXTENSION['template'], 'source_tasks': [], 'upstream_ref': None,
                    'source_kind': RELEASE_EXTENSION['source_kind']}
        if re.match(r'^\s*(?:fully\s+)?open\b.*\bgripper\b', instruction, re.IGNORECASE):
            raise ValueError('Use exactly Open the gripper. for the verified placement release extension')
        for identifier, sources, template, pattern in self.patterns:
            match = pattern.fullmatch(instruction.strip())
            if match is None or any(ACTION_CLAUSE.search(slot) for slot in match.groups()):
                continue
            return {'version': self.VERSION, 'form': identifier, 'template': template,
                    'source_tasks': list(sources), 'upstream_ref': self.catalog['upstream_ref']}
        raise ValueError(
            'Instruction is not grounded in an atomic description/form. Use an atomic_instruction_forms '
            'template with scene entity names, e.g. "Pick up the mug.", "Place the mug on the tray.", '
            '"Open the cabinet." or "Navigate to the dining counter." For other skills use the '
            'documented atomic description with its placeholders filled. Keep explanations in evidence; '
            'do not invent free-form motor instructions or append another action inside an entity slot.')

    def validate_plan(self, steps):
        if any(gripper_release_instruction(step['instruction']) for step in steps):
            raise ValueError('Open the gripper is a verified temporary placement intervention, not a standalone planned stage')
        return [self.validate(step['instruction']) for step in steps]

    def validate_generated(self, instruction, committed_instruction=None):
        """Apply catalog vocabulary/format equally to new and committed text."""
        return self.validate(instruction)

    def validate_generated_plan(self, steps, committed_instruction=None):
        if any(re.match(r'^\s*pick\b.*\band\s+(?:place|put)\b', step['instruction'], re.IGNORECASE) for step in steps):
            raise ValueError('Generated pick-and-place plans must start decomposed: pick, optional navigation, place')
        return self.validate_plan(steps)


@lru_cache(maxsize=1)
def atomic_instruction_grounder():
    return AtomicInstructionGrounder()
