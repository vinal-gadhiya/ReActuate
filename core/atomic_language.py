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
        return [self.validate(step['instruction']) for step in steps]


@lru_cache(maxsize=1)
def atomic_instruction_grounder():
    return AtomicInstructionGrounder()
