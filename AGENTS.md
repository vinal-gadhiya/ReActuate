# Project requirements

- Optimize the shared harness for all RoboCasa tasks, not the task currently being demonstrated. Do not add task-name branches, object-specific recovery logic, task-specific thresholds, or special-case action wording to the controller.
- Public, versioned task catalogs may describe individual tasks. They are reference data, not controller rules or evidence of policy competence.
- Keep the frozen policy's observation contract and numerical actions intact. Preserve explicitly supplied policy instruction text. Simple instructions are preferred; backend wording changes require measured evidence.
- Keep original requirements separate from mutable execution plans. Treat visual outcomes and memories as fallible claims with dated observation references.
- Evaluate changes across task families and paired scenes. Report baseline-success regressions and operational failures alongside improvements. Mock tests prove contracts and plumbing, not live model quality.
- This workspace is for source work. Run Python tests, builds, simulation, inference, and installation in the Thor containers, not on the local host. Read-only inspection and source/data edits are allowed locally.
