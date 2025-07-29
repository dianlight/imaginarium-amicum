---
name: SOLID Rules
---

# SOLID Design Principles - Coding Assistant Guidelines

When generating, reviewing, or modifying code, follow these guidelines to ensure adherence to the SOLID principles.

## Implementation Guidelines

1. **Explicitly Identify Single Responsibility**
   - When starting a new class, explicitly identify its single responsibility.
   - Document extension points and expected subclassing behavior.
   - Write interface contracts with clear expectations and invariants.
   - Question any class that depends on many concrete implementations.
   - Use design patterns (Strategy, Decorator, Factory, Observer, etc.) to facilitate SOLID adherence.

2. **Document Extension Points and Expected Subclassing Behavior**
   - Ensure extension points are clearly documented.
   - Document expected subclassing behavior.
   - Write interface contracts with clear expectations and invariants.
   - Question any class that depends on many concrete implementations.
   - Use design patterns (Strategy, Decorator, Factory, Observer, etc.) to facilitate SOLID adherence.

3. **Regular Refactoring Toward SOLID**
   - Especially when extending functionality.
   - Use design patterns (Strategy, Decorator, Factory, Observer, etc.) to facilitate SOLID adherence.

4. **Warning Signs**
   - God classes that do "everything"
   - Methods with boolean parameters that radically change behavior
   - Deep inheritance hierarchies
   - Classes that need to know about implementation details of their dependencies
   - Circular dependencies between modules
   - High coupling between unrelated components
   - Classes that grow rapidly in size with new features
   - Methods with many parameters

@diff

Follow the Conventional Commits format strictly for commit messages. Use the structure below:\n\n```\n<type>[optional scope]: <gitmoji> <description>\n\n[optional body]\n```\n\nGuidelines:\n\n1. **Type and Scope**: Choose an appropriate type (e.g., `feat`, `fix`) and optional scope to describe the affected module or feature.\n2. **Gitmoji**: Include a relevant `gitmoji` that best represents the nature of the change.\n3. **Description**: Write a concise, informative description in the header; use backticks if referencing code or specific terms.\n4. **Body**: For additional details, use a well-structured body section:\n   - Use bullet points (`*`) for clarity.\n   - Clearly describe the motivation, context, or technical details behind the change, if applicable.\n\nCommit messages should be clear, informative, and professional, aiding readability and project tracking.