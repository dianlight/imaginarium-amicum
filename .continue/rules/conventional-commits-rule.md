---
name: Conventional Commits Rule
---

# Conventional Commits - Commit Message Formatting Guidelines

When committing changes, adhere to the Conventional Commits format to ensure consistency and clarity in your commit messages. This rule will help maintain a uniform and understandable commit history.

## Guidelines

1. **Type and Scope**: Choose an appropriate type (e.g., `feat`, `fix`) and optional scope to describe the affected module or feature.
2. **Gitmoji**: Include a relevant `gitmoji` that best represents the nature of the change.
3. **Description**: Write a concise, informative description in the header; use backticks if referencing code or specific terms.
4. **Body**: For additional details, use a well-structured body section:
   - Use bullet points (`*`) for clarity.
   - Clearly describe the motivation, context, or technical details behind the change, if applicable.

## Example Commit Message

```markdown
feat[optional scope]: <gitmoji> <description>

[optional body]
```

### Example

```markdown
feat(ui): 🚀 Add new feature to display user profiles

- Add a new endpoint to fetch user profiles
- Update the UI to display user information
- Refactor existing code to handle new data structures
```

### Warning Signs

- Commit messages that do not follow the Conventional Commits format.
- Commit messages that are unclear or lack the necessary context.

## Implementation Guidelines

1. **Type and Scope**: 
   - `feat`: A new feature for the user-facing application
   - `fix`: A bug fix for the application
   - `docs`: Documentation only changes
   - `refactor`: A code change that neither fixes a bug nor adds a feature
   - `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
   - `test`: Adding missing tests or correcting existing tests
   - `chore`: Changes to the build process or auxiliary tools and libraries such as documentation, etc.

2. **Gitmoji**:
   - Use relevant Gitmojis to represent the nature of the change.

3. **Description**:
   - Write a concise, informative description in the header.
   - Use backticks if referencing code or specific terms.

4. **Body**:
   - Provide additional details in a well-structured body section.
   - Use bullet points (`*`) for clarity.
   - Clearly describe the motivation, context, or technical details behind the change.

## Example Rules

- Ensure every commit message follows the Conventional Commits format.
- Use appropriate types and scopes.
- Include relevant Gitmojis.
- Provide clear and concise descriptions.
- Include additional details in the body if necessary.

@diff

Follow the Conventional Commits format strictly for commit messages. Use the structure below:\n\n```\n<type>[optional scope]: <gitmoji> <description>\n\n[optional body]\n```\n\nGuidelines:\n\n1. **Type and Scope**: Choose an appropriate type (e.g., `feat`, `fix`) and optional scope to describe the affected module or feature.\n2. **Gitmoji**: Include a relevant `gitmoji` that best represents the nature of the change.\n3. **Description**: Write a concise, informative description in the header; use backticks if referencing code or specific terms.\n4. **Body**: For additional details, use a well-structured body section:\n   - Use bullet points (`*`) for clarity.\n   - Clearly describe the motivation, context, or technical details behind the change, if applicable.\n\nCommit messages should be clear, informative, and professional, aiding readability and project tracking.