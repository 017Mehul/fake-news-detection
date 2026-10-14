# Fake News Detection — Web UI

A responsive white-and-gray 3D/neumorphic interface for the Fake News Detection Python project.

## Preview locally

No build step is required.

1. Download or clone the repository.
2. Open `web/index.html` in a browser, or serve the folder with a local static server:

   ```bash
   python -m http.server 8000
   ```

   Then visit `http://localhost:8000/web/` from the repository root.

## Current scope

- Responsive navigation and page sections
- News article text area with character counter and sample text
- Result panel with an explicit demo state
- Grayscale, raised-card, and 3D illustration styling
- GitHub repository links

**Important:** this is currently a frontend-only experience. It does not claim to produce real predictions. To activate classification, connect the form submission in `script.js` to a backend API that loads the project's trained model and returns a documented prediction response. Do not expose model files or secrets in frontend code.
