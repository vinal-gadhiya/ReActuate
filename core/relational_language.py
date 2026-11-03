"""Awomo language experiment: broad action frames with grounded referents."""

from functools import lru_cache
import re

from core.atomic_language import AtomicInstructionGrounder, atomic_instruction_grounder, ACTION_CLAUSE, gripper_release_instruction


RELATIONAL_POLICY = '''RELATIONAL INSTRUCTION MODE (Awomo): Use a clear action,
intended object/component, and source/destination or reference when needed. The
atomic catalog is vocabulary guidance, not an exact sentence/verb whitelist.
Use concise natural language with observed color, shape, size, support, spatial
relation or neighboring fixture references that distinguish the intended entity.
Prefer a stable named anchor: the mug to the right of the brown mug; the left
cabinet beside the microwave; the plate on the neighboring counter. Resolve
which camera/view supports left/right and use scene/fixture-relative references
where possible. Avoid unexplained this/that/next/right; explicitly name the anchor
and intended support. A reference object is an anchor, not automatically the
support: placing beside a mug usually requires the counter underneath, not the
mug itself. Never invent attributes, unseen neighbors or coordinates.
Cite current camera/step evidence identifying target, source and destination.
Illustrations, NOT observations: Pick the mug from the cabinet beside the sink.;
Place the mug to the left of the brown mug.; Pick the mug from the neighboring
cabinet and place it on the counter beside the brown mug. Use such wording ONLY
when the corresponding entities/relation are actually observed and task-eligible.
Keep original goal identities, containment/support relations and exclusions.
Near an object is not on a required tray, and beside a bowl is not inside it.
No new object or destination may be substituted for convenience. Use separate
pick, optional navigation and place by default; a concise combined transfer is
allowed when the observed references make it clearer. One executable tool call
per decision remains mandatory. No procedural chains, code, arbitrary motor
coordinates or commentary in the instruction. Specialized controls may still use
their documented appliance mechanism. Prefer the least wording that resolves
ambiguity; verbose wording is not evidence of greater competence.
When useful, refine the same pending action with more precise current references
using reference_refinement recovery and dated evidence. This is not a claim of
failure. Preserve the original requirement/destination and achieved grasp, and
explicitly replan for actual changes of objective or strategy.
The ONLY gripper-opening language extension is exactly Open the gripper., a
temporary controlled_release intervention during placement with supported hold,
verified destination alignment and low height, limited to release_step_limit.
Do not use synonyms to evade that independent release review. Completion still
needs verified destination support, release and clearly withdrawn gripper.
All temporal cadence, fixture-opening checks and physical execution gates remain.'''

# Basic action frames, not task/object-specific recipes. Clauses are checked
# separately so a spatial description cannot conceal additional motor actions.
DESTINATION = r'(?:on|onto|in|into|inside|under|near|beside|by|next to|to the (?:left|right) of|on the (?:left|right) side of|in front of|behind)'
TRANSFER = re.compile(r'^(?:pick(?: up)?|take|grasp)\s+(.+?)\s+and\s+(?:place|put|set)\s+(?:it|them)\s+(' + DESTINATION + r')\s+(.+?)\.?$', re.IGNORECASE)
ACQUIRE = re.compile(r'^(?:pick(?: up)?|take|grasp)\s+(.+?)\.?$', re.IGNORECASE)
PLACE = re.compile(r'^(?:place|put|set)\s+(.+?)\s+(' + DESTINATION + r')\s+(.+?)\.?$', re.IGNORECASE)
EXTRA_ACTION = re.compile(r'(?:\b(?:and|then|before|after)\s+|,\s*)(?:pick|take|grasp|place|put|set|open|close|navigate|move|turn|press|release)\b', re.IGNORECASE)


class RelationalInstructionGrounder(AtomicInstructionGrounder):
    mode = 'relational'

    def validate(self, instruction):
        if (not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000
                or any(char in instruction for char in '\n\r;`{}') or '<|' in instruction):
            raise ValueError('Relational instruction must be one plain action sentence')
        # Never turn unreviewed hand-actuator synonyms into fixture instructions.
        if (re.match(r'^(?:fully\s+)?(?:open|release)\b.*\b(?:gripper|fingers|hand)\b', instruction.strip(), re.IGNORECASE)
                and not gripper_release_instruction(instruction)):
            raise ValueError('Use the exact verified Open the gripper. extension for hand release')
        if re.match(r'^(?:pick|take|grasp|place|put|set)\b', instruction.strip(), re.IGNORECASE) and EXTRA_ACTION.search(instruction):
            transfer = TRANSFER.fullmatch(instruction.strip())
            if transfer is None or any(EXTRA_ACTION.search(part) for part in transfer.groups()):
                raise ValueError('Only one pickup-plus-placement transfer is allowed; no procedural action chains')
        try:
            result = super().validate(instruction)
            return {**result, 'instruction_mode': self.mode}
        except ValueError:
            pass
        for name, pattern in (('relational_transfer', TRANSFER), ('relational_place', PLACE), ('relational_pick', ACQUIRE)):
            match = pattern.fullmatch(instruction.strip())
            if match is not None and not any(ACTION_CLAUSE.search(part) or EXTRA_ACTION.search(part) or re.search(r'[.!?]', part.rstrip('.'))
                                            for part in match.groups()):
                return {'version': 'relational-language-v1', 'form': name,
                        'template': 'Action + observed target/source/destination or spatial reference',
                        'source_tasks': [], 'upstream_ref': None, 'instruction_mode': self.mode,
                        'source_kind': 'harness_relational_language_experiment'}
        raise ValueError('Use a basic action frame with grounded target/source/destination references; no action chains')

    def validate_generated_plan(self, steps, committed_instruction=None):
        return self.validate_plan(steps)


@lru_cache(maxsize=4)
def instruction_grounder(backend, mode='auto'):
    if mode not in ('auto', 'atomic', 'relational'):
        raise ValueError('Unknown instruction mode')
    if mode == 'relational' and backend != 'awomo-0.5i':
        raise ValueError('Relational instruction mode is currently an Awomo experiment')
    if backend == 'awomo-0.5i' and mode in ('auto', 'relational'):
        return RelationalInstructionGrounder()
    return atomic_instruction_grounder()


def instruction_policy(base, grounder):
    if getattr(grounder, 'mode', 'atomic') != 'relational':
        return base
    prefix, marker, rest = base.partition('ATOMIC LANGUAGE CONTRACT:')
    _, end, suffix = rest.partition('\nDo not invent manual motor controls.')
    if not marker or not end:
        raise ValueError('Instruction policy section markers changed')
    rendered = prefix + RELATIONAL_POLICY + end + suffix
    tail_start = 'Ground generated instructions in the supplied atomic task descriptions and\n'
    before, marker, tail = rendered.partition(tail_start)
    if marker:
        _, keep, after = tail.partition('Never silently rewrite an explicit operator instruction.')
        if not keep:
            raise ValueError('Atomic language appendix marker changed')
        rendered = before + 'Use observed references with the selected relational action frames.\n' + keep + after
    return rendered.replace('Keep the decomposed phases and preserve achieved progress;',
                            "Preserve the chosen action's scope and achieved progress;")
